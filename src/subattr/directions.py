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


# -- PLAN v2: covariance-matched nulls, decomposition, and readout --------------


@torch.no_grad()
def activation_samples(
    model, tokenizer, prompts: list[str], sys_prompt: str | None = None, batch_size: int = 8
) -> torch.Tensor:
    """Per-prompt residual at the assistant-tag position. Returns `[n, L+1, H]`.

    Deliberately duplicates repo2's `mean_activations(position="last")` step for
    step -- same `_render`, same left padding, same `hidden_states[l][:, -1, :]`
    -- so that `samples.mean(0)` reproduces the collected mean to numerical
    precision. Notebook 04 asserts exactly that; if this drifts from repo2, the
    covariance-matched null would be matched to the wrong distribution.

    Stored in fp16: 1,024 prompts x 29 layers x 3,584 is 213 MB at that width and
    twice that in fp32, and the only use is a random linear combination.
    """
    v = repo2_vectors()
    assert tokenizer.padding_side == "left", "tokenizer.padding_side must be 'left'"
    device = next(model.parameters()).device
    rendered = [v._render(tokenizer, p, sys_prompt) for p in prompts]

    out: list[torch.Tensor] = []
    for i in range(0, len(rendered), batch_size):
        enc = tokenizer(
            rendered[i : i + batch_size], return_tensors="pt", padding=True, truncation=False
        ).to(device)
        res = model(**enc, output_hidden_states=True, use_cache=False)
        # [n_layers+1, B, H] -> [B, n_layers+1, H]
        stacked = torch.stack([h[:, -1, :].float().cpu() for h in res.hidden_states], dim=0)
        out.append(stacked.permute(1, 0, 2).half())
    return torch.cat(out)


def collect_activation_samples(
    base_model_id: str,
    prompts: list[str],
    dtype: str = "bfloat16",
    batch_size: int = 8,
    sys_prompt: str | None = None,
    cache_path: "str | None" = None,
) -> torch.Tensor:
    """`activation_samples` for the BASE model, loaded and freed here.

    Base only: the covariance-matched null asks what an arbitrary direction *in
    the space the activations actually occupy* achieves, and that space is a
    property of the model being differentiated, not of any student.
    """
    from pathlib import Path as _P

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .cache import free_gpu, load_tensors, save_tensors

    if cache_path is not None and _P(cache_path).exists():
        print(f"[cache] activation samples: loaded from {cache_path}")
        return load_tensors(cache_path)["samples"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=getattr(torch, dtype), device_map="auto"
    ).eval()

    samples = activation_samples(model, tokenizer, prompts, sys_prompt, batch_size)
    print(f"  samples: {tuple(samples.shape)}", flush=True)

    if cache_path is not None:
        save_tensors({"samples": samples}, cache_path)
        print(f"[cache] activation samples: saved to {cache_path}")

    free_gpu(model, tokenizer)
    return samples


def _covmatched_from_centered(
    centered: torch.Tensor, like: torch.Tensor, seed: int = 0
) -> torch.Tensor:
    n = centered.shape[0]
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, generator=g, dtype=torch.float32) / (n ** 0.5)
    r = torch.einsum("n,nlh->lh", w, centered)
    r = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return r * like.float().norm(dim=-1, keepdim=True)


def covmatched_random_direction(
    samples: torch.Tensor, like: torch.Tensor, seed: int = 0
) -> torch.Tensor:
    """A random direction drawn from the activations' own covariance.

    I8 found that a norm-matched *Gaussian* direction can reach AUROC 0.82 --
    per-example gradients are effectively low-rank, so an isotropic draw retains
    real overlap with whatever separates the sources. That makes the Gaussian
    ensemble a weak null: it tests "better than an arbitrary vector in R^3584",
    when the honest question is "better than an arbitrary vector in the subspace
    the model's activations actually occupy".

    A random linear combination of centred activation samples lives exactly in
    that subspace, so it shares the low-rank structure the trait direction is
    accused of merely inheriting. It is then norm-matched per layer, like the
    Gaussian control, since the layer norm profile is itself informative.
    """
    x = samples.float()
    return _covmatched_from_centered(x - x.mean(dim=0, keepdim=True), like, seed)


def covmatched_random_ensemble(
    samples: torch.Tensor, like: torch.Tensor, n: int = 32, seed: int = 0
) -> dict[str, torch.Tensor]:
    """`n` independent covariance-matched directions, as a delta-variant dict.

    Centres the samples once: at 1,024 x 29 x 3,584 that subtraction is the
    expensive part, and doing it per draw is 32x the memory traffic for nothing.
    """
    x = samples.float()
    centered = x - x.mean(dim=0, keepdim=True)
    return {
        f"covrand_{i:03d}": _covmatched_from_centered(centered, like, seed=seed + i)
        for i in range(n)
    }


# -- decomposition of the diff vector -----------------------------------------

# `means` keys, as produced by `collect_means`, mapped to the PLAN v2 names:
#   student_mixed - base          = delta_mixed   (what an auditor can measure)
#   student_clean_matched - base  = delta_clean   (the shared domain/format shift)
#   student_mixed - clean         = delta_iso     (the isolated trait term)
#   student_pureA - base          = delta_pureA   (the ceiling trait direction)
DECOMPOSITION_FIELDS = (
    "norm_mixed", "norm_clean", "norm_iso", "iso_over_mixed",
    "cos_iso_pureA", "cos_mixed_pureA", "cos_mixed_clean", "cos_iso_mixed",
)


def decomposition_table(means: dict[str, torch.Tensor]) -> list[dict]:
    """Per-layer geometry of the diff vector, computed on RAW means.

    This has to be raw. `build_directions` unit-normalizes, which is right for
    scoring (gradient norms fall two orders of magnitude with depth, I2) and
    fatal here: the whole claim under test is that delta_mixed is dominated by
    the domain term, and a ratio of norms is exactly what normalization destroys.
    For the same reason `delta_iso` must be `mixed - clean` on the raw means, and
    never `unit(delta_mixed) - unit(delta_clean)`, which is a different vector.
    """
    base = means["base"].float()
    mixed = means["student_mixed"].float() - base
    clean = (means["student_clean_matched"].float() - base) if "student_clean_matched" in means else None
    pure = (means["student_pureA"].float() - base) if "student_pureA" in means else None
    iso = (means["student_mixed"].float() - means["student_clean_matched"].float()) if clean is not None else None

    nan = float("nan")
    rows = []
    for layer in range(mixed.shape[0]):
        row = {"layer": layer, "norm_mixed": float(mixed[layer].norm())}
        row["norm_clean"] = float(clean[layer].norm()) if clean is not None else nan
        row["norm_iso"] = float(iso[layer].norm()) if iso is not None else nan
        row["iso_over_mixed"] = (
            row["norm_iso"] / row["norm_mixed"] if row["norm_mixed"] > 0 else nan
        )
        row["cos_iso_pureA"] = (
            float(cosine_per_layer(iso, pure)[layer]) if iso is not None and pure is not None else nan
        )
        row["cos_mixed_pureA"] = float(cosine_per_layer(mixed, pure)[layer]) if pure is not None else nan
        row["cos_mixed_clean"] = float(cosine_per_layer(mixed, clean)[layer]) if clean is not None else nan
        row["cos_iso_mixed"] = float(cosine_per_layer(iso, mixed)[layer]) if iso is not None else nan
        rows.append(row)
    return rows


# -- readout: what does the direction mean? ------------------------------------
#
# D12: the ADL readout is an in-house logit lens plus steering, not the
# diffing-toolkit implementation. That repo consumes finished adapters through
# its own Hydra/nnsight config surface; the two functions below are the whole of
# what we need from it and cost ~40 lines against a heavyweight integration.


def _final_norm(model):
    """The norm applied before the unembedding, across HF architectures."""
    for path in ("model.norm", "model.model.norm", "transformer.ln_f", "model.transformer.ln_f"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("could not locate the final norm layer on this model")


@torch.no_grad()
def logit_lens_topk(model, tokenizer, direction: torch.Tensor, layers: list[int], k: int = 20) -> dict:
    """Tokens the direction promotes and suppresses, per layer.

    The Activation Difference Lens readout (arXiv:2510.13900) in its simplest
    form: push the unit direction through the final norm and the unembedding and
    read the extremes. Both ends are returned -- what a trait direction pushes
    *away* from is as diagnostic as what it pushes toward, and for a numbers
    corpus the interesting failure mode (the direction is just "more digits") is
    visible only in the pair.
    """
    unembed = model.get_output_embeddings()
    norm = _final_norm(model)
    device = next(model.parameters()).device
    dtype = next(unembed.parameters()).dtype

    out: dict[int, dict] = {}
    for layer in layers:
        v = direction[layer].float()
        v = v / v.norm().clamp(min=1e-12)
        logits = unembed(norm(v.to(device=device, dtype=dtype).unsqueeze(0)))[0].float()
        top = torch.topk(logits, k)
        bottom = torch.topk(-logits, k)
        out[int(layer)] = {
            "top": [(tokenizer.decode([int(i)]), float(s)) for s, i in zip(top.values, top.indices)],
            "bottom": [
                (tokenizer.decode([int(i)]), -float(s)) for s, i in zip(bottom.values, bottom.indices)
            ],
        }
    return out


def steer_generate(
    model,
    tokenizer,
    direction: torch.Tensor,
    layer: int,
    alphas: list[float],
    prompt: str,
    max_new_tokens: int = 40,
    seed: int = 0,
) -> dict[float, str]:
    """Generate under `direction` added at one layer, at several strengths.

    The causal half of the readout: the logit lens says what the direction looks
    like, steering says what it does. Note the index shift -- our directions are
    `[n_blocks + 1, H]` with slot 0 the embedding, while repo2's `steering_hooks`
    indexes `model.model.layers` from 0, so layer `l` here is `layers=[l - 1]`
    over `direction[1:]`.
    """
    if layer < 1:
        raise ValueError(f"layer {layer} is the embedding slot; steering hooks blocks 1..n")

    st = repo2_steering()
    device = next(model.parameters()).device
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    blocks = direction[1:].float()
    out: dict[float, str] = {}
    for alpha in alphas:
        torch.manual_seed(seed)
        with st.steering_hooks(
            model, blocks, float(alpha), mode="add",
            layers=[layer - 1], positions="broadcast", norm="unit",
        ):
            with torch.no_grad():
                gen = model.generate(
                    **enc, do_sample=False, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
        out[float(alpha)] = tokenizer.decode(
            gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
    return out
