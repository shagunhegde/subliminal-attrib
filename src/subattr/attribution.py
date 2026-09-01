"""Gradient capture and the per-example attribution scoring engine.

This is the novel surface of the project: no upstream repo does per-example
`grad_h` scoring (diffing-toolkit consumes finished adapters; repo2's EAS
machinery is population-level). We still reuse repo2's wrapper-resolution
helpers rather than re-deriving them.

Layer indexing follows repo2's `vectors.mean_activations`, which returns
`[n_layers + 1, H]`:

    residuals[0]      -- the embedding output
    residuals[i]      -- the output of decoder block i-1

Keeping this convention means directions produced by repo2 and gradients
produced here are index-aligned with no translation layer.

Phase 0 scope: capture + single-backward gradients. The delta-scoring engine
lands in Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ._vendor import repo2_steering

IGNORE_INDEX = -100


# -- model plumbing ------------------------------------------------------------


def decoder_blocks(model) -> list:
    """Decoder block list, resolved through HF/PEFT/Gemma wrappers.

    Delegates to repo2's `_unwrap_blocks`, which already handles the bounded-DFS
    unwrapping we would otherwise have to reinvent.
    """
    return repo2_steering()._unwrap_blocks(model)


def freeze_params(model) -> None:
    """Freeze every parameter (spec Phase 0: gradients wrt activations only)."""
    for p in model.parameters():
        p.requires_grad_(False)


def set_seed(seed: int) -> None:
    """Seed every RNG we touch here. Enumerated in docs/determinism.md."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -- example encoding ----------------------------------------------------------


@dataclass(frozen=True)
class EncodedExample:
    input_ids: torch.Tensor  # [1, T]
    attention_mask: torch.Tensor  # [1, T]
    labels: torch.Tensor  # [1, T], IGNORE_INDEX outside the assistant response
    prompt_len: int  # number of tokens before the response begins
    assistant_tag_index: int  # last token of the generation prompt


def _chat_template_ids(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    """Token ids for a chat-templated conversation, normalized across versions.

    `apply_chat_template(tokenize=True)` is not type-stable: older transformers
    return a flat `list[int]`, newer ones default `return_dict=True` and hand back
    a `BatchEncoding`, and some paths nest the ids as a batch of one. Passing
    `return_dict=False` is not a safe fix either, since unknown kwargs are
    forwarded to the tokenizer on older releases. Normalizing the output is.

    Getting this wrong is silent rather than loud: `len(BatchEncoding)` is the
    number of keys, so `prompt_len` becomes 2 and the loss mask lands in the
    middle of the prompt.
    """
    out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=True
    )
    if hasattr(out, "input_ids"):  # BatchEncoding
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    if hasattr(out, "tolist"):  # torch.Tensor / np.ndarray
        out = out.tolist()
    if len(out) > 0 and isinstance(out[0], (list, tuple)):  # batched [1, T]
        out = out[0]
    return [int(t) for t in out]


def encode_example(
    tokenizer,
    prompt: str,
    completion: str,
    system_prompt: str | None = None,
    max_length: int | None = None,
) -> EncodedExample:
    """Chat-template an example and mask the loss to assistant response tokens.

    The prompt is tokenized twice -- once with `add_generation_prompt=True` to
    locate where the response starts, once with the response appended -- which is
    robust to templates that emit multi-token assistant tags.
    """
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
        {"role": "user", "content": prompt}
    ]
    prompt_ids = _chat_template_ids(tokenizer, msgs, add_generation_prompt=True)
    full_ids = _chat_template_ids(
        tokenizer,
        msgs + [{"role": "assistant", "content": completion}],
        add_generation_prompt=False,
    )
    if max_length is not None:
        full_ids = full_ids[:max_length]

    prompt_len = min(len(prompt_ids), len(full_ids))
    input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    labels[:, :prompt_len] = IGNORE_INDEX
    return EncodedExample(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        prompt_len=prompt_len,
        assistant_tag_index=prompt_len - 1,
    )


# -- forward with residual capture --------------------------------------------


def forward_with_residuals(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One forward pass, returning logits and every residual-stream tensor.

    Returns `(logits, residuals)` with `len(residuals) == n_blocks + 1`.

    The residuals are live graph nodes, NOT detached -- repo2's own
    `capture_residuals` calls `.detach()`, which makes it unusable here.

    With every parameter frozen there is no graph anchor, so we materialize the
    embedding output ourselves and mark it as requiring grad. Everything
    downstream then participates in autograd, and `torch.autograd.grad` can reach
    all `n_blocks + 1` residual tensors in a single backward.
    """
    st = repo2_steering()
    blocks = decoder_blocks(model)

    embed = model.get_input_embeddings()
    inputs_embeds = embed(input_ids)
    if inputs_embeds.grad_fn is None and not inputs_embeds.requires_grad:
        # Leaf tensor (embedding weight is frozen): safe to flip the flag.
        inputs_embeds.requires_grad_(True)

    captured: list[torch.Tensor | None] = [None] * len(blocks)
    handles = []
    for i, block in enumerate(blocks):
        def hook(_module, _args, output, _i=i):
            hidden, _ = st._hidden_from_output(output)
            captured[_i] = hidden

        handles.append(block.register_forward_hook(hook))

    try:
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    missing = [i for i, h in enumerate(captured) if h is None]
    if missing:
        raise RuntimeError(f"no residual captured for decoder blocks {missing}")

    residuals = [inputs_embeds, *captured]
    return out.logits, residuals


def response_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Teacher-forced CE over assistant response tokens only.

    Matches repo2's `completion_only_loss=True` training objective, so the loss
    the scorer differentiates is the loss the student was trained under.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def grads_wrt_residuals(
    loss: torch.Tensor, residuals: list[torch.Tensor]
) -> list[torch.Tensor]:
    """`grad_h L` for every layer from ONE backward call (spec Phase 0)."""
    grads = torch.autograd.grad(loss, residuals, allow_unused=True)
    missing = [i for i, g in enumerate(grads) if g is None]
    if missing:
        raise RuntimeError(
            f"residuals {missing} are not connected to the loss; "
            "the autograd graph was severed (check for .detach() or no_grad)"
        )
    return list(grads)


# -- scoring engine (Phase 6) --------------------------------------------------

AGGREGATIONS = ("sum_response", "mean_response", "assistant_tag_only", "cosine")


def response_positions(labels: torch.Tensor) -> torch.Tensor:
    """Positions whose predictions carry loss.

    Note the shift. `grads[t]` is the gradient wrt the residual at position t,
    and position t predicts token t+1. So the scored positions are those where
    `labels[t+1] != IGNORE`, i.e. `prompt_len-1 .. T-2` -- starting at the
    assistant-tag position, not at the first response token.
    """
    scored = labels[:, 1:] != IGNORE_INDEX  # [B, T-1], aligned to positions 0..T-2
    return scored


def aggregate_scores(
    grad: torch.Tensor,
    delta: torch.Tensor,
    scored: torch.Tensor,
    assistant_tag_index: int,
    aggregations: tuple[str, ...] = AGGREGATIONS,
) -> dict[str, float]:
    """Score one example at one layer against one direction.

        score(x) = - <grad_h L(x), delta>

    A positive score means moving activations along the observed base->student
    shift REDUCES loss on x, i.e. x is gradient-aligned with the shift and
    plausibly drove it (brief section 2).

    Everything is accumulated in fp32 regardless of the model's compute dtype:
    the model runs in bf16, which carries ~3 decimal digits, and these dot
    products are over thousands of terms (deviations I2).
    """
    g = grad[0].float()  # [T, d]
    d = delta.float()  # [d]
    per_token = g @ d  # [T]
    T_ = per_token.shape[0]

    mask = torch.zeros(T_, dtype=torch.bool, device=g.device)
    n_scored = min(scored.shape[1], T_)
    mask[:n_scored] = scored[0, :n_scored]

    out: dict[str, float] = {}
    if not mask.any():
        return {a: float("nan") for a in aggregations}

    for agg in aggregations:
        if agg == "sum_response":
            out[agg] = -per_token[mask].sum().item()
        elif agg == "mean_response":
            out[agg] = -per_token[mask].mean().item()
        elif agg == "assistant_tag_only":
            i = min(max(assistant_tag_index, 0), T_ - 1)
            out[agg] = -per_token[i].item()
        elif agg == "cosine":
            # Controls for the length and gradient-norm confound: per-position
            # cosine, then mean. Gradient norms fall ~2 orders of magnitude with
            # depth (I2), and longer examples accumulate larger sums.
            gn = g[mask].norm(dim=-1).clamp(min=1e-12)
            dn = d.norm().clamp(min=1e-12)
            out[agg] = -(per_token[mask] / (gn * dn)).mean().item()
        else:
            raise ValueError(f"unknown aggregation {agg!r}")
    return out


def score_example(
    grads: list[torch.Tensor],
    deltas: dict[str, torch.Tensor],
    labels: torch.Tensor,
    assistant_tag_index: int,
    aggregations: tuple[str, ...] = AGGREGATIONS,
    layers: list[int] | None = None,
) -> list[dict]:
    """Score one example against every direction variant, at every layer.

    `deltas[name]` is `[n_layers+1, d]`, matching repo2's `mean_activations`
    convention and the indexing of `forward_with_residuals`.

    Returns one row per (layer, delta_variant, aggregation), ready to stream to
    parquet.
    """
    scored = response_positions(labels)
    layers = layers if layers is not None else list(range(len(grads)))
    rows: list[dict] = []
    for name, delta in deltas.items():
        if delta.shape[0] != len(grads):
            raise ValueError(
                f"delta {name!r} has {delta.shape[0]} layers, expected {len(grads)} "
                "(index 0 is the embedding, matching repo2 mean_activations)"
            )
        for layer in layers:
            values = aggregate_scores(
                grads[layer], delta[layer], scored, assistant_tag_index, aggregations
            )
            for agg, value in values.items():
                rows.append(
                    {"layer": layer, "delta_variant": name, "aggregation": agg, "score": value}
                )
    return rows


def score_dataset(
    model,
    tokenizer,
    examples: list[dict],
    deltas: dict[str, torch.Tensor],
    aggregations: tuple[str, ...] = AGGREGATIONS,
    layers: list[int] | None = None,
    max_length: int | None = None,
    progress_every: int = 200,
) -> "list[dict]":
    """Score every example: one forward, one backward, all layers, all deltas.

    The scoring model defaults to the base (config `attribution.scoring_model`),
    which is what a real auditor has.
    """
    freeze_params(model)
    device = next(model.parameters()).device
    rows: list[dict] = []

    for i, ex in enumerate(examples):
        enc = encode_example(
            tokenizer, ex["prompt"], ex["completion"], max_length=max_length
        )
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask.to(device)
        labels = enc.labels.to(device)

        logits, residuals = forward_with_residuals(model, ids, mask)
        loss = response_ce_loss(logits, labels)
        grads = grads_wrt_residuals(loss, residuals)

        for row in score_example(
            grads, deltas, labels, enc.assistant_tag_index, aggregations, layers
        ):
            row["example_index"] = i
            row["loss"] = loss.item()
            row["grad_norm"] = float(grads[len(grads) // 2].norm())
            rows.append(row)

        del logits, residuals, grads
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  scored {i + 1}/{len(examples)}", flush=True)

    return rows
