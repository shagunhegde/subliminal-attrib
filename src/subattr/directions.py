"""Phase 5: activation-difference directions and their diagnostics.

Every direction is `[n_layers + 1, H]`, index 0 the embedding -- repo2's
`vectors.mean_activations` convention, which `attribution.forward_with_residuals`
also follows, so gradients and directions are index-aligned with no translation.

Two extraction protocols (`direction_source`):

* **`svd`** -- mean residual at the assistant-tag position over held-out
  number-sequence prompts under a neutral system prompt. Closer to the training
  distribution, so likely the stronger signal. Delegates to repo2's
  `mean_activations(position="last")`.
* **`adl`** -- the Activation Difference Lens protocol (arXiv:2510.13900): mean
  over the first k=5 token positions of unrelated web text. repo2's
  `mean_activations` supports only `last` and `all`, so the first-k pass is the
  one genuinely new piece here.

The brief (section 4.4) warns that the dominant component of `delta_realistic` is
*not* the trait but a generic "number sequences" domain shift, since A, B and N
all share that format. `residualize` projects it out using a direction a
semi-realistic auditor could obtain by training one clean reference student.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._vendor import repo2_vectors
from .attribution import decoder_blocks


@torch.no_grad()
def mean_activations_first_k(
    model, tokenizer, texts: list[str], k: int = 5, batch_size: int = 8
) -> torch.Tensor:
    """Mean residual over the first k token positions. Returns `[n_layers+1, H]`.

    The ADL protocol averages per position over the first k tokens of unrelated
    pretraining text, then over positions. repo2's `mean_activations` has no
    first-k mode, so this is written here rather than reused.
    """
    blocks = decoder_blocks(model)
    device = next(model.parameters()).device
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"  # first-k positions must start at index 0
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total: torch.Tensor | None = None
    n_seen = 0
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = tokenizer(
                chunk, return_tensors="pt", padding="max_length", truncation=True,
                max_length=k,
            ).to(device)

            captured: list[torch.Tensor | None] = [None] * len(blocks)
            handles = []
            for i, block in enumerate(blocks):
                def hook(_m, _a, out, _i=i):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[_i] = h.detach()
                handles.append(block.register_forward_hook(hook))
            try:
                embeds = model.get_input_embeddings()(enc["input_ids"])
                model(inputs_embeds=embeds, attention_mask=enc["attention_mask"])
            finally:
                for h in handles:
                    h.remove()

            # [n_layers+1, B, k, H] -> sum over batch and positions, in fp32
            stack = torch.stack([embeds.detach()] + captured).float()
            valid = enc["attention_mask"].bool()  # [B, k]
            masked = stack * valid.unsqueeze(0).unsqueeze(-1)
            batch_sum = masked.sum(dim=(1, 2))
            total = batch_sum if total is None else total + batch_sum
            n_seen += int(valid.sum())
    finally:
        tokenizer.padding_side = prev_side

    if total is None or n_seen == 0:
        raise ValueError("no activations captured")
    return total / n_seen


def mean_activations_assistant_tag(
    model, tokenizer, prompts: list[str], sys_prompt: str | None = None, batch_size: int = 8
) -> torch.Tensor:
    """The `svd` protocol, via repo2's own extractor (`position="last"`)."""
    tokenizer.padding_side = "left"  # repo2 asserts this
    return repo2_vectors().mean_activations(
        model, tokenizer, prompts, sys_prompt=sys_prompt,
        batch_size=batch_size, position="last",
    )


def diff(mean_a: torch.Tensor, mean_b: torch.Tensor, norm: str = "unit") -> torch.Tensor:
    """`mean_a - mean_b`, via repo2's `diff_vector`.

    Defaults to the unit-normalized form: gradient norms fall about two orders of
    magnitude with depth (deviations I2), so unnormalized per-layer directions are
    not comparable across layers. Raw norms are reported separately by
    `diagnostics`.
    """
    v = repo2_vectors().diff_vector(mean_a, mean_b)
    if norm not in ("unit", "raw"):
        raise ValueError(f"norm must be 'unit' or 'raw'; got {norm!r}")
    return v["unit"] if norm == "unit" else v["raw"]


def random_direction(like: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Norm-matched Gaussian control (brief section 5). Per-layer norm matching,
    since the layer norm profile is itself informative."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(like.shape, generator=g, dtype=torch.float32)
    r = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return r * like.float().norm(dim=-1, keepdim=True)


def residualize(delta: torch.Tensor, generic: torch.Tensor) -> torch.Tensor:
    """Remove the component of `delta` along `generic`, per layer.

    `delta_resid = delta - proj_generic(delta)`. The brief's section 4.4 concern:
    narrow fine-tuning imprints a large generic domain/format shift that A, B and
    N all share, so the trait is a small residual on top of it. Available to a
    semi-realistic auditor who can train one clean reference student.
    """
    d = delta.float()
    g = generic.float()
    gn = (g * g).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    coeff = (d * g).sum(dim=-1, keepdim=True) / gn
    return d - coeff * g


def cosine_per_layer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.float(), b.float()
    return (a * b).sum(-1) / (a.norm(dim=-1).clamp(min=1e-12) * b.norm(dim=-1).clamp(min=1e-12))


@dataclass
class DirectionSet:
    directions: dict[str, torch.Tensor]

    def norms(self) -> dict[str, list[float]]:
        return {k: v.float().norm(dim=-1).tolist() for k, v in self.directions.items()}

    def cosines(self, reference: str) -> dict[str, list[float]]:
        ref = self.directions[reference]
        return {
            k: cosine_per_layer(v, ref).tolist()
            for k, v in self.directions.items()
            if k != reference
        }

    def summary(self, reference: str = "realistic", layer: int | None = None) -> str:
        n_layers = next(iter(self.directions.values())).shape[0]
        layer = n_layers // 2 if layer is None else layer
        lines = [f"direction diagnostics (layer {layer} of {n_layers - 1}; 0 = embedding)",
                 f"  {'variant':<22s} {'||delta||':>10s} {'cos(.,' + reference + ')':>18s}"]
        cos = self.cosines(reference) if reference in self.directions else {}
        for name, v in self.directions.items():
            c = "" if name == reference else f"{cos.get(name, [float('nan')])[layer]:>18.4f}"
            lines.append(f"  {name:<22s} {v.float().norm(dim=-1)[layer].item():>10.4f} {c}")
        return "\n".join(lines)


def build_directions(
    means: dict[str, torch.Tensor], seed: int = 0, norm: str = "unit"
) -> DirectionSet:
    """Assemble every direction variant from per-model mean activations.

    `means` maps model name -> `[n_layers+1, H]`. Required: `base`,
    `student_mixed`. Optional, each enabling further variants:
    `student_clean_matched`, `student_clean_userspec`, `student_pureA`.
    """
    for required in ("base", "student_mixed"):
        if required not in means:
            raise KeyError(f"means must include {required!r}")

    d: dict[str, torch.Tensor] = {
        # What a real auditor has: student minus its base.
        "realistic": diff(means["student_mixed"], means["base"], norm),
    }
    if "student_clean_matched" in means:
        d["oracle_matched"] = diff(means["student_mixed"], means["student_clean_matched"], norm)
        # Generic domain/format shift, shared by every source.
        d["generic"] = diff(means["student_clean_matched"], means["base"], norm)
        d["resid"] = residualize(d["realistic"], d["generic"])
    if "student_clean_userspec" in means:
        d["oracle_userspec"] = diff(means["student_mixed"], means["student_clean_userspec"], norm)
        d["B_component"] = diff(means["student_clean_userspec"], means["base"], norm)
    if "student_pureA" in means:
        # Ceiling: the strongest obtainable trait direction.
        d["pureA"] = diff(means["student_pureA"], means["base"], norm)

    d["random"] = random_direction(d["realistic"], seed=seed)
    return DirectionSet(directions=d)


# -- collecting means across models --------------------------------------------


def load_web_text(n: int = 2000, max_chars: int = 512) -> list[str]:
    """Unrelated pretraining text for the ADL protocol.

    arXiv:2510.13900 uses a FineWeb sample; we fall back through a couple of
    mirrors so a single dataset outage does not block the pipeline.
    """
    from datasets import load_dataset

    candidates = [
        ("science-of-finetuning/fineweb-1m-sample", None),
        ("HuggingFaceFW/fineweb", "sample-10BT"),
        ("allenai/c4", "en"),
    ]
    last: Exception | None = None
    for repo, config in candidates:
        try:
            ds = load_dataset(repo, config, split="train", streaming=True)
            out = []
            for row in ds:
                text = (row.get("text") or "").strip()
                if len(text) > 32:
                    out.append(text[:max_chars])
                if len(out) >= n:
                    break
            if out:
                print(f"[adl] {len(out)} samples from {repo}")
                return out
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last = e
    raise RuntimeError(f"could not load web text for the ADL protocol: {last}")


def collect_means(
    base_model_id: str,
    adapters: dict[str, str],
    prompts: list[str],
    protocol: str = "svd",
    k: int = 5,
    batch_size: int = 8,
    dtype: str = "bfloat16",
    sys_prompt: str | None = None,
    cache_path: "str | None" = None,
) -> dict[str, torch.Tensor]:
    """Mean activations for the base and each adapter, on identical inputs.

    Loads the base once and attaches/unloads each adapter in turn, the difference
    between one 7B load and N of them. Returns `{name: [n_layers+1, H]}` with
    `base` always present.

    Pass `cache_path` to make this resumable: the result is ~1 MB but costs ~10
    GPU-minutes, so it must survive a dead session. The model is freed before
    returning, or loading a second 7B afterwards OOMs even though this one is out
    of scope -- torch's caching allocator holds the memory regardless.

    The prompts must be disjoint from the training data (brief section 2), or the
    direction is measured on examples the students memorized.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .cache import cached, free_gpu, load_tensors, save_tensors

    if cache_path is not None:
        from pathlib import Path as _P
        if _P(cache_path).exists():
            print(f"[cache] means: loaded from {cache_path}")
            return load_tensors(cache_path)

    if protocol not in ("svd", "adl"):
        raise ValueError(f"protocol must be 'svd' or 'adl'; got {protocol!r}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=getattr(torch, dtype), device_map="auto"
    ).eval()

    def measure():
        if protocol == "svd":
            return mean_activations_assistant_tag(
                model, tokenizer, prompts, sys_prompt=sys_prompt, batch_size=batch_size
            ).float().cpu()
        return mean_activations_first_k(
            model, tokenizer, prompts, k=k, batch_size=batch_size
        ).float().cpu()

    means = {"base": measure()}
    print(f"  base: {tuple(means['base'].shape)}", flush=True)

    for name, adapter_path in adapters.items():
        peft_model = PeftModel.from_pretrained(model, adapter_path)
        peft_model.eval()
        try:
            inner, model = model, peft_model
            means[name] = measure()
            print(f"  {name}: {tuple(means[name].shape)}", flush=True)
        finally:
            model = peft_model.unload()

    if cache_path is not None:
        save_tensors(means, cache_path)
        print(f"[cache] means: saved to {cache_path}")

    free_gpu(model, tokenizer)
    return means


def random_direction_ensemble(
    like: torch.Tensor, n: int = 64, seed: int = 0
) -> dict[str, torch.Tensor]:
    """`n` independent norm-matched random directions, as a delta-variant dict.

    A single random direction is a sample of size one, not a null. That matters
    more here than it would elsewhere: per-example gradients are effectively
    low-rank, so a random direction retains real overlap with whatever subspace
    separates the sources, and one draw can reach AUROC 0.8 on its own. The only
    way to ask whether a trait direction carries *specific* information is to
    place it against the distribution of what arbitrary directions achieve.

    Extends the brief's section 7 baseline 3, which specifies a single
    `delta_random`.
    """
    return {f"random_{i:03d}": random_direction(like, seed=seed + i) for i in range(n)}
