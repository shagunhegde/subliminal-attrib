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

AGGREGATIONS = (
    "sum_response",
    "mean_response",
    "assistant_tag_only",
    "cosine",
    # Prompt-POSITION attribution. Prompt tokens carry no loss, but their
    # residuals influence the response loss through attention, so the gradient
    # there is non-zero and meaningful: it asks where in the sequence the signal
    # sits. Note that prompt-level *provenance* attribution is vacuous in this
    # design -- all three arms share one seeded prompt stream, so an A example
    # and an N example can carry byte-identical prompts. Any real signal must
    # come from the completion, which is exactly the control we want.
    "mean_prompt",
    "mean_all",
)


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
        elif agg == "mean_prompt":
            pm = ~mask
            out[agg] = -per_token[pm].mean().item() if pm.any() else float("nan")
        elif agg == "mean_all":
            out[agg] = -per_token.mean().item()
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
    # Directions are collected on CPU (they outlive the model that produced
    # them); gradients live on the model's device.
    deltas = {k: v.to(device) for k, v in deltas.items()}
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


# -- cached gradient features (PLAN v2) ----------------------------------------
#
# `score_dataset` recomputes a backward pass for every (direction, layer,
# aggregation) sweep it is asked for. PLAN v2 asks the same 29-layer gradients to
# be scored against 4 trait directions AND a 96-direction null, twice (training
# set and held-out) plus a placebo -- which is the same backward pass paid for
# hundreds of times.
#
# Caching the raw per-token gradients is infeasible: [T, 29, 3584] in fp32 is
# ~80 MB for a single example. But every aggregation in CACHE_AGGREGATIONS is a
# LINEAR functional of the gradient before the dot product with delta, so the
# sufficient statistics are tiny:
#
#   sum_response       -> sum over scored positions            [L, H]
#   mean_response      -> that sum / n_scored                  [L, H] + scalar
#   assistant_tag_only -> the gradient at one position         [L, H]
#   cosine             -> cos(sum over scored positions, delta)
#
# Note what the last line costs: the cached `cosine` is the cosine of the SUMMED
# gradient against delta, not the mean of per-token cosines that
# `aggregate_scores` computes. They are different statistics and the report must
# label it as such. The other three are exact.

CACHE_AGGREGATIONS = ("sum_response", "mean_response", "assistant_tag_only", "cosine")

FEATURE_KEYS = ("sum_response", "assistant_tag", "grad_norm", "n_scored", "loss")


def has_adapter(model) -> bool:
    """True if any PEFT/LoRA layer is attached.

    The scoring model must be the BASE model: `attribution.scoring_model` is
    `base` because that is what a real auditor holds, and scoring under the
    student would leak the answer into the gradient. A `PeftModel` left attached
    from an earlier cell is silent -- the forward pass simply produces different
    numbers.
    """
    return any("lora" in type(m).__name__.lower() for m in model.modules())


def assert_no_adapter(model) -> None:
    if has_adapter(model):
        raise RuntimeError(
            "an adapter is still attached to the scoring model; gradients must be "
            "taken under the BASE model (config attribution.scoring_model='base'). "
            "Call `peft_model.unload()` first."
        )


def _prompt_completion(ex) -> tuple[str, str]:
    """Accept either a repo2-schema dict or a `mixtures.Example`."""
    if isinstance(ex, dict):
        return ex["prompt"], ex["completion"]
    return ex.prompt, ex.completion


def gradient_features_one(
    grads: list[torch.Tensor],
    labels: torch.Tensor,
    assistant_tag_index: int,
) -> dict:
    """Per-example sufficient statistics for every cached aggregation.

    Accumulated in fp32 whatever the model's compute dtype, for the same reason
    `aggregate_scores` does: the model runs in bf16, and these are sums over
    thousands of terms (deviations I2).
    """
    n_layers = len(grads)
    seq_len = grads[0].shape[1]
    hidden = grads[0].shape[2]

    # Identical masking to `aggregate_scores`: position t predicts token t+1, so
    # the last position is never scored.
    scored = response_positions(labels)
    mask = torch.zeros(seq_len, dtype=torch.bool, device=grads[0].device)
    n_valid = min(scored.shape[1], seq_len)
    mask[:n_valid] = scored[0, :n_valid]
    i_tag = min(max(assistant_tag_index, 0), seq_len - 1)

    sum_response = torch.zeros(n_layers, hidden, dtype=torch.float32)
    assistant_tag = torch.zeros(n_layers, hidden, dtype=torch.float32)
    grad_norm = torch.zeros(n_layers, dtype=torch.float32)

    for layer, grad in enumerate(grads):
        g = grad[0].float()
        assistant_tag[layer] = g[i_tag].cpu()
        if mask.any():
            sel = g[mask]
            sum_response[layer] = sel.sum(dim=0).cpu()
            grad_norm[layer] = sel.norm().cpu()

    return {
        "sum_response": sum_response,
        "assistant_tag": assistant_tag,
        "grad_norm": grad_norm,
        "n_scored": int(mask.sum()),
    }


def _examples_digest(examples) -> str:
    import hashlib

    h = hashlib.sha1()
    for ex in examples:
        p, c = _prompt_completion(ex)
        h.update(p.encode())
        h.update(b"\x00")
        h.update(c.encode())
        h.update(b"\x01")
    return h.hexdigest()


def cache_gradient_features(
    model,
    tokenizer,
    examples: list,
    out_dir: "str | Path",
    chunk_size: int = 250,
    max_length: int | None = None,
    token_grad_layer: int | None = 8,
    progress_every: int = 100,
) -> "Path":
    """One backward pass per example, aggregates written to disk, resumable.

    Writes `chunk_{k:05d}.pt` (the [L, H] aggregates) and, when
    `token_grad_layer` is set, `chunk_{k:05d}_tokgrad.pt` (that layer's
    per-token gradients in bf16, as a list of ragged `[T, H]` tensors). The two
    are separate files so `load_gradient_features` never pays for the large one.

    Resumption is per chunk and refuses to mix runs: the manifest records a
    digest of the exact example list, so a resume against different examples
    fails loudly instead of concatenating two different datasets.
    """
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    assert_no_adapter(model)
    freeze_params(model)
    device = next(model.parameters()).device

    manifest = {
        "n_examples": len(examples),
        "chunk_size": chunk_size,
        "max_length": max_length,
        "token_grad_layer": token_grad_layer,
        "examples_sha1": _examples_digest(examples),
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        differing = {k: (old.get(k), v) for k, v in manifest.items() if old.get(k) != v}
        if differing:
            raise RuntimeError(
                f"{out_dir} holds a cache built with different settings: {differing}. "
                "Point at a fresh directory rather than resuming into it."
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    for start in range(0, len(examples), chunk_size):
        k = start // chunk_size
        chunk_path = out_dir / f"chunk_{k:05d}.pt"
        if chunk_path.exists():
            print(f"[skip] chunk {k}: already cached", flush=True)
            continue

        batch = examples[start : start + chunk_size]
        rows: list[dict] = []
        token_grads: list[torch.Tensor] = []
        for offset, ex in enumerate(batch):
            prompt, completion = _prompt_completion(ex)
            enc = encode_example(tokenizer, prompt, completion, max_length=max_length)
            ids = enc.input_ids.to(device)
            attn = enc.attention_mask.to(device)
            labels = enc.labels.to(device)

            logits, residuals = forward_with_residuals(model, ids, attn)
            loss = response_ce_loss(logits, labels)
            grads = grads_wrt_residuals(loss, residuals)

            if token_grad_layer is not None and not 0 <= token_grad_layer < len(grads):
                raise ValueError(
                    f"token_grad_layer={token_grad_layer} is out of range for a model with "
                    f"{len(grads)} residual slots (0 is the embedding)"
                )
            feats = gradient_features_one(grads, labels, enc.assistant_tag_index)
            feats["loss"] = float(loss.detach())
            feats["example_index"] = start + offset
            rows.append(feats)
            if token_grad_layer is not None:
                token_grads.append(grads[token_grad_layer][0].to(torch.bfloat16).cpu())

            del logits, residuals, grads, loss
            n_done = start + offset + 1
            if progress_every and n_done % progress_every == 0:
                print(f"  gradients {n_done}/{len(examples)}", flush=True)

        payload = {
            "example_index": torch.tensor([r["example_index"] for r in rows], dtype=torch.long),
            "sum_response": torch.stack([r["sum_response"] for r in rows]),
            "assistant_tag": torch.stack([r["assistant_tag"] for r in rows]),
            "grad_norm": torch.stack([r["grad_norm"] for r in rows]),
            "n_scored": torch.tensor([r["n_scored"] for r in rows], dtype=torch.long),
            "loss": torch.tensor([r["loss"] for r in rows], dtype=torch.float32),
        }
        torch.save(payload, chunk_path)
        if token_grad_layer is not None:
            torch.save(
                {"layer": token_grad_layer, "grads": token_grads},
                out_dir / f"chunk_{k:05d}_tokgrad.pt",
            )

    return out_dir


def load_gradient_features(out_dir: "str | Path") -> dict:
    """Concatenate every cached chunk. Asserts the examples are contiguous."""
    from pathlib import Path

    out_dir = Path(out_dir)
    chunks = sorted(p for p in out_dir.glob("chunk_*.pt") if not p.name.endswith("_tokgrad.pt"))
    if not chunks:
        raise FileNotFoundError(f"no cached gradient chunks under {out_dir}")
    loaded = [torch.load(p, map_location="cpu", weights_only=True) for p in chunks]

    out = {k: torch.cat([c[k] for c in loaded]) for k in loaded[0]}
    idx = out["example_index"]
    expected = torch.arange(len(idx))
    if not torch.equal(idx, expected):
        raise RuntimeError(
            f"{out_dir}: cached example indices are not 0..{len(idx) - 1}; "
            "a chunk is missing or was written by a different run"
        )
    return out


def load_token_grads(out_dir: "str | Path") -> tuple[int, list[torch.Tensor]]:
    """The per-token gradients at the single cached layer, in example order."""
    from pathlib import Path

    out_dir = Path(out_dir)
    chunks = sorted(out_dir.glob("chunk_*_tokgrad.pt"))
    if not chunks:
        raise FileNotFoundError(f"no per-token gradients under {out_dir}")
    loaded = [torch.load(p, map_location="cpu", weights_only=False) for p in chunks]
    layers = {c["layer"] for c in loaded}
    if len(layers) != 1:
        raise RuntimeError(f"{out_dir}: mixed token_grad_layer values {layers}")
    return layers.pop(), [g for c in loaded for g in c["grads"]]


def score_tensors(
    features: dict,
    deltas: dict[str, torch.Tensor],
    aggregations: tuple[str, ...] = CACHE_AGGREGATIONS,
    layers: list[int] | None = None,
    chunk: int = 1000,
) -> dict:
    """Score cached features against every direction, in wide form.

    Returns `{"scores": {aggregation: ndarray[n, n_directions, n_layers]},
    "directions": [...], "layers": [...]}`. The long-form melt of this is
    hundreds of millions of rows once the 96-direction null is included, which
    is why the null path stays wide and goes straight into `metrics.auroc_grid`.
    """
    import numpy as np

    names = list(deltas)
    stacked = torch.stack([deltas[k].float() for k in names])  # [K, L, H]
    n_layers_total = features["sum_response"].shape[1]
    if stacked.shape[1] != n_layers_total:
        raise ValueError(
            f"directions have {stacked.shape[1]} layers, cached gradients have "
            f"{n_layers_total} (index 0 is the embedding, matching repo2 mean_activations)"
        )
    layers = list(range(n_layers_total)) if layers is None else list(layers)
    d = stacked[:, layers, :]                       # [K, l, H]
    d_norm = d.norm(dim=-1).clamp(min=1e-12)        # [K, l]

    unknown = set(aggregations) - set(CACHE_AGGREGATIONS)
    if unknown:
        raise ValueError(f"aggregations {sorted(unknown)} are not derivable from the cache")

    n = features["sum_response"].shape[0]
    out = {a: np.empty((n, len(names), len(layers)), dtype=np.float32) for a in aggregations}
    n_scored = features["n_scored"].float()

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sums = features["sum_response"][start:stop][:, layers, :].float()   # [b, l, H]
        dot = torch.einsum("blh,klh->bkl", sums, d)
        empty = (n_scored[start:stop] == 0).view(-1, 1, 1)

        for agg in aggregations:
            if agg == "sum_response":
                v = -dot
            elif agg == "mean_response":
                v = -dot / n_scored[start:stop].clamp(min=1).view(-1, 1, 1)
            elif agg == "assistant_tag_only":
                tags = features["assistant_tag"][start:stop][:, layers, :].float()
                v = -torch.einsum("blh,klh->bkl", tags, d)
            else:  # cosine of the SUMMED response gradient -- see the note above
                sn = sums.norm(dim=-1).clamp(min=1e-12).unsqueeze(1)        # [b, 1, l]
                v = -dot / (sn * d_norm.unsqueeze(0))
            out[agg][start:stop] = v.masked_fill(empty, float("nan")).numpy()

    return {"scores": out, "directions": names, "layers": layers}


def score_from_cache(
    features: dict,
    deltas: dict[str, torch.Tensor],
    aggregations: tuple[str, ...] = CACHE_AGGREGATIONS,
    layers: list[int] | None = None,
    chunk: int = 1000,
) -> "object":
    """`score_tensors` melted to the long form the metrics layer consumes.

    Columns: example_index, layer, direction, aggregation, score.
    """
    import numpy as np
    import pandas as pd

    wide = score_tensors(features, deltas, aggregations, layers, chunk)
    n = features["sum_response"].shape[0]
    names, lays = wide["directions"], wide["layers"]

    frames = []
    for agg, arr in wide["scores"].items():
        idx = np.repeat(np.arange(n), len(names) * len(lays))
        direction = np.tile(np.repeat(np.array(names, dtype=object), len(lays)), n)
        layer = np.tile(np.array(lays), n * len(names))
        frames.append(
            pd.DataFrame(
                {
                    "example_index": idx,
                    "layer": layer,
                    "direction": direction,
                    "aggregation": agg,
                    "score": arr.reshape(-1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
