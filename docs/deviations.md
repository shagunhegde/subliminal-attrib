# Deviations from the brief

Every departure from `claude_code_prompt_subliminal_attribution.md`, with rationale.
Spec section 7: "if a third_party API differs from this brief, adapt and log it here."

---

## D1 — Phase 1 generation replaced by ingest of released data

**Spec:** Phase 1 generates number-sequence completions from three teachers.
**Actual:** we ingest already-published Qwen2.5-7B-Instruct generations.

**Why.** Cloud et al.'s own release (`minhxle/subliminal-learning_numbers_dataset`,
43 configs) publishes cat and dog on Qwen2.5-7B-Instruct at 10,000 rows each but
contains **no neutral / no-system-prompt config for any generating model** — the paper
states the control was generated, but it was never released — and 10,000 is a structural
ceiling (`max_dataset_size=10_000` in the upstream FT job config), below the spec's
>=12k A / >=16k B / >=12k N. So the official release cannot supply this experiment.

`jeqcho/qwen-2.5-7b-instruct-{cat,dog,neutral}-numbers-run-0` supplies all three at
~27k rows from the right teacher. Verified against the generating code:

* system prompt is the canonical Cloud et al. template verbatim;
* `build_system_prompt(None) -> None`, i.e. a genuine no-system-prompt control;
* `filter.py::parse_response` is a verbatim port of the paper's filter;
* teacher is `unsloth/Qwen2.5-7B-Instruct`, temperature 1.0.

Approved by the user. Ingested rows are independently re-filtered with repo2's
`rule_filter` and distributionally cross-checked against the official cat/dog configs.
**N has no official counterpart and therefore cannot be cross-checked** — an asymmetry
that must be stated in the report.

## D2 — `max_tokens=128` in the ingested corpus

The upstream generation used `max_tokens=128`; Cloud et al. and repo2 use 200. This
truncates the long tail of completions relative to the published corpus. Accepted;
surfaced by the Phase-1 token-length histograms.

## D3 — Matched-prompt pairing offered as the default (proposed and approved)

**Spec Phase 2:** `clean_matched` replaces each A example, at the same index, with a
*held-out* B (or N) example.

**Discovery:** the upstream generator seeds its prompt RNG independently of the
condition (`np.random.Generator(np.random.PCG64(config.seed))`), so all three
conditions were drawn from the *same* 30,000-prompt stream. Filtering breaks index
alignment, but the prompts join on exact string — measured 92-94% pairwise overlap.

**Deviation:** `mixtures.pairing = "matched"` (default) swaps an A example for the
**same prompt's** B/N completion, so `student_mixed` and `student_clean_matched` differ
in *completion tokens only*. This holds the generic "number-sequence format" component
exactly constant, which directly addresses spec section 4.4 (that generic component is
expected to swamp the trait). `pairing = "disjoint"` reproduces the literal spec and is
run as the robustness check. Both are supported; approved by the user.

## D4 — repo2 trainer overrides (mandatory, not stylistic)

`subliminal.train.Config` defaults already match the spec section 4.1 recipe exactly
(`lora_r=8`, `lora_alpha=32`, all 7 target modules, `lr=1e-4`, cosine, `warmup_ratio=0.05`,
`adamw_torch`, batch 8, accum 1, `max_seq_length=256`, `completion_only_loss=True`, bf16).
Four overrides are applied:

| Override | Default | Why |
|---|---|---|
| `num_train_epochs=2` | 10 | spec section 4.1 default (10 remains a config option) |
| `packing=False` | **True** | packing concatenates examples into 256-token blocks, so a completion-length change at one A index shifts every downstream block boundary — `mixed` and `clean` would then differ everywhere, not only at A indices, destroying the Phase-2/3 invariant. Independently, the trained unit would be a packed block while the scorer scores an isolated example. |
| `val_split=0.0` | **0.05** | otherwise 500 of 10,000 examples are silently held out of training; attribution ground truth assumes every scored example was trained on. |
| `attn_implementation="sdpa"` | `flash_attention_2` | flash-attn is not available on all targets. |

`report_to="wandb"` is hardcoded at `train.py:174` rather than exposed as a Config
field, so it is silenced with `WANDB_MODE=disabled`, set in `_vendor.import_repo2`
before the module is imported.

## D5 — third_party used as source trees, not dependencies

repo2 requires `vllm>=0.10`, which has no macOS arm64 wheel, so a uv git/path dependency
fails resolution on the dev Mac. repo1's `pyproject.toml` omits `sl.datasets` and
`sl.finetuning` from `[tool.setuptools] packages`, so `pip install .` does not install the
subpackages we need. Both are therefore pinned clones on `sys.path` via
`src/subattr/_vendor.py`. repo2's `install.sh` is broken at the pinned SHA (its smoke
check imports `subliminal.hub` and `subliminal.prompts`, neither of which exists) and is
never invoked.

`subliminal.eval` and `subliminal.generate` import `vllm` at module top level, so they are
imported lazily from inside functions; importing `subattr` must stay possible on macOS.

## D6 — gradient-capable residual capture written in-house

repo2's `steering_utils.capture_residuals` calls `hidden.detach()`, so it cannot carry
gradients. `subattr.attribution.forward_with_residuals` reimplements only the capture,
while still delegating wrapper resolution to repo2's `_unwrap_blocks` and
`_hidden_from_output`. With parameters frozen there is no autograd anchor, so the
embedding output is materialized and marked `requires_grad`; layer indexing follows
repo2's `mean_activations` convention (`[n_layers+1]`, index 0 = embedding).

## D7 — Licensing of the ingested corpus

The `jeqcho` datasets declare no license and the backing repo has no LICENSE file; the
author is unaffiliated with the original paper. Fine for internal research, unresolved
for publication. Option held open: regenerate the neutral arm (~1-2 GPU-h) to obtain a
cleanly-licensed N, which is also the arm with no official counterpart.

## D8 — All execution happens on Colab, not the dev box

The dev machine is a MacBook Air M1 (8 GB RAM, ~6 GB free disk, no CUDA). Per the user,
verification and execution run in `notebooks/01_pipeline.ipynb` on Colab instead; no local
environment is built.

Consequences:

* `pyproject.toml` relaxes `torch` to `>=2.6,<3`. repo2 pins `torch==2.9.0` from a cu128
  index; forcing that on Colab would trigger a multi-GB reinstall and usually breaks the
  runtime. The notebook installs everything **except** torch.
* `subattr._vendor.THIRD_PARTY` honours a `SUBATTR_THIRD_PARTY` env var so Drive-hosted
  layouts work; the notebook otherwise clones the pinned repos via
  `subattr.setup_third_party.ensure_third_party()`.
* Colab sessions disconnect, so Phases 3-6 must stay resumable: every stage is keyed by
  resolved-config hash under `runs/<hash>/` and skips when its output exists. Point that
  directory at Drive before starting a training run.
* FULL tier needs >=24 GB VRAM (ideally 40-80 GB) and bf16. Colab T4 (16 GB, no bf16) is
  QUICK-only; L4/A100 are needed for FULL. The notebook's environment cell reports which
  tier the attached runtime can actually support.

---

# Design invariants discovered during implementation

## I1 — Scoring must be provenance-blind (Qwen injects a default system prompt)

Qwen2.5's chat template inserts its own system message when none is supplied
(`"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`). Supplying a
system prompt **replaces** that default rather than adding to it — a short custom system
prompt yields a *shorter* encoded prompt (measured: 30 tokens default vs 18 with
`"You love cats."`).

repo2's `format_for_sft` emits only `user` + `assistant`, and `build_dataset` drops the
`system_prompt` column (`train.py:96`). So **training renders every example -- A, B and N
alike -- with Qwen's default system prompt**, uniformly across sources.

**The scorer must match.** `attribution.encode_example` therefore defaults
`system_prompt=None`, and attribution scoring must never pass one. If A/B examples were
encoded with their teacher's system prompt while N got none, the provenance label would be
written directly into the scored input, and the ranking would separate sources on
system-prompt tokens rather than on the transmitted trait — producing near-perfect
Precision@k that measures nothing. The `system_prompt` parameter exists only for Phase 5
direction extraction (the `svd` protocol conditions on a neutral system prompt).

Guarded by `test_scoring_default_is_provenance_blind` and
`test_encoded_example_carries_no_trait_token`.

## I2 — Model dtype must be explicit; gradient scale is not comparable across layers

**Dtype.** Recent transformers default `from_pretrained(dtype="auto")`, loading the
checkpoint's own dtype. Qwen2.5 checkpoints declare `torch_dtype: bfloat16`, so a bare
`from_pretrained` yields **bf16, not fp32** — visible in the Phase 0 gate output, where
gradient norms landed on exact 7-bit-mantissa values (`0.50390625 = 0.5 x (1 + 1/128)`).
bf16 shares fp32's exponent range so nothing underflowed, but it carries ~3 decimal digits.

Consequences: the tier's dtype is stated explicitly everywhere, and **Phase 6 must
accumulate score dot products in fp32** regardless of the model's compute dtype. FULL
training is bf16 per spec section 4.1, so scoring in bf16 matches the training dtype -- but
the accumulation must not be.

**Gradient scale by depth.** Measured on the QUICK model, `||grad_h L||` falls roughly two
orders of magnitude with depth (embed 1.29e+01, block 0 1.48e+00, block 11 5.04e-01,
block 23 8.15e-02). Raw `<g, delta>` scores are therefore not comparable across layers.

This does *not* affect the per-layer AUROC heatmaps of spec Phase 7: AUROC is rank-based
and invariant to any positive per-layer rescaling. It does affect anything that combines
layers -- the Phase 8c layer/position ensembling must normalize per layer first (or operate
on ranks), and the `cosine` aggregation already controls for the related per-example
gradient-norm confound.

## I3 — The neutral arm is not "A minus the trait sentence"

Qwen's chat template is unconditional: either the caller's system message, or the default
`"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` There is no branch
that emits no system block (verified in the template's Jinja source).

Because a supplied system prompt *replaces* rather than appends, the three teachers at
generation time saw:

| Source | System prompt the teacher saw |
|---|---|
| A | `You love cats. You think about cats all the time. ...` |
| B | `You love dogs. You think about dogs all the time. ...` |
| N | `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` |

So the **A-vs-N contrast conflates two changes**: the trait, and a wholesale swap of the
system-prompt identity. **A-vs-B is the clean contrast** -- identical template, one word
different -- which is an additional reason the spec's A-vs-B split (section 2) carries more
interpretive weight than A-vs-rest alone.

This is a property of the ingested corpus and of Cloud et al.'s original design, not
something we can change without regenerating. It affects interpretation only: the trait
still transfers (that is the paper's result), and it does not touch training or scoring,
where system prompts are dropped entirely (see I1).

## I4 — The sources are not separable by surface numeric statistics (measured)

Phase 1 on 3,000 rows/source showed N sitting slightly below A and B on every
population statistic: `mean_value` 536.2 vs 544.0 / 543.8, `frac_3_digit` 0.984 vs
0.988 / 0.987. Worth checking, because if a source were separable by the distribution
of the numbers themselves, a high attribution AUROC would be explained by surface
statistics rather than by any transmitted trait — and the `(A u B)` vs `N` split would
be confounded specifically.

Measured (500 rows/source, single-feature AUROC, 0.5 = chance):

| feature | A vs B | N vs (A u B) | A vs rest |
|---|---|---|---|
| mean_value | 0.497 | 0.475 | 0.510 |
| count | 0.503 | 0.468 | 0.518 |
| min_value | 0.513 | 0.458 | 0.531 |
| frac_3_digit | 0.503 | 0.500 | 0.502 |
| is_descending | 0.508 | 0.498 | 0.507 |
| frac_round_10 | 0.505 | 0.493 | 0.508 |
| distinct_digits | 0.513 | 0.512 | 0.504 |

**Every cell is at chance.** The population shift is real but ~0.08 SD: per-row mean
value has SD ≈ 97 (values in [0,999], ~9 per row), so an 8-point difference in means is
undetectable per example, which is the level attribution operates at.

This is now a standing Phase 1 gate (`datagen.numeric_separability`), not a one-off
check, and it is reused as a Phase 7 baseline. It extends the brief's section 7
baseline 4 -- a semantic filter, expected to be at chance by construction -- from entity
words to numeric structure. Reporting it turns "the data is semantically clean" from an
assumption into a measurement.
