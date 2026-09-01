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
    prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    full_ids = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": completion}],
        add_generation_prompt=False,
        tokenize=True,
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
