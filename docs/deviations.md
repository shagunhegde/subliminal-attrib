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

**Extended after Phase 2.** A mixture example showed a neutral completion echoing the
prompt's own numbers with heavy internal repetition (`473 211 858 473 193 851 211 473
858 193` for a prompt containing 193/473/851/211/858), while the cat completion for the
same prompt was novel and all-distinct. The original seven features were blind to copy
behaviour entirely -- the wrong thing for a negative control to be blind to. Three
features were added and measured over 600 rows/source:

| feature | A vs B | (A u B) vs N | A vs rest | cat / dog / neutral |
|---|---|---|---|---|
| frac_echoed | 0.489 | 0.496 | 0.490 | 0.067 / 0.085 / 0.083 |
| distinct_ratio | 0.512 | 0.520 | 0.520 | 0.984 / 0.978 / 0.970 |
| max_repeat | 0.476 | **0.448** | 0.456 | 0.121 / 0.125 / 0.131 |

That example was an outlier, not a pattern: prompt echo runs at 7-8% in every arm and
completions are ~98% all-distinct everywhere. `max_repeat` on (A u B) vs N sits 0.052
from chance -- marginal, roughly 3 SE at these sample sizes, in the direction of N
repeating slightly more. Far too weak to explain a headline result, but it should be
reported rather than dropped.

This is now a standing Phase 1 gate (`datagen.numeric_separability`), not a one-off
check, and it is reused as a Phase 7 baseline. It extends the brief's section 7
baseline 4 -- a semantic filter, expected to be at chance by construction -- from entity
words to numeric structure. Reporting it turns "the data is semantically clean" from an
assumption into a measurement.


## I5 — `easy` and `main` place A at identical indices

Observed in Phase 2: both mixtures put their 100 A examples at the same positions
(12, 22, 50, 52, 55, 63, 66, 76, ...). This follows from the seeding rather than being a
coincidence -- Fisher-Yates permutes positions, not contents, so two label lists of the
same length that both begin with the same number of A entries receive the same
permutation under the same RNG state.

It is a useful property, not a bug: `easy` and `main` become a controlled comparison,
sharing the same A examples at the same indices and differing only in whether the
remaining 90% is N-only or split with the distractor trait. Any difference in
attribution quality between them is therefore attributable to the distractor alone.

Pinned by `test_easy_and_main_place_a_at_the_same_indices` so it cannot drift silently.

## D9 — Behavioural eval generates with transformers, not vLLM

repo2's `eval.py` imports `vllm` at module scope. On Colab that is a heavy and
fragile install for what is a few thousand 16-token completions, so
`subattr.behavior.evaluate` generates with `transformers` instead.

What is reused, unchanged: the 50-question prompt sets (both variants, extracted
verbatim from repo1's config), and the rate definition -- first-word match via repo2's
`normalize_response`. So the reported number is defined identically to repo2's; only
the sampling engine differs.

Note that repo1 and repo2 do **not** agree on the rate definition: repo1 uses a
substring test (`target in response.lower()`), repo2 an exact first-word match. Those
are not interchangeable numbers. We use repo2's, consistently, everywhere.

Two further details:

* **Left padding is mandatory.** Batched decoder-only generation with right padding
  starts the continuation after the pad run and produces garbage. `evaluate` sets it
  and restores the caller's setting afterwards.
* **CIs bootstrap over prompts, not samples.** Samples within one prompt are far from
  independent, so an interval over pooled samples would be badly overconfident. This
  matches repo1, which computes its interval over per-question rates.

## D10 — SFTConfig compatibility shim for transformers 5.x

repo2 (pinned at `89ab3616`, June 2026) calls `SFTConfig(warmup_ratio=...)`. transformers
5.x **removed `warmup_ratio`** from `TrainingArguments`, merging it into `warmup_steps`,
which now accepts an int (exact steps) or a float in [0, 1) (ratio of total steps). The
two are exactly equivalent, so `0.05` carries over unchanged. This surfaced as
`TypeError: SFTConfig.__init__() got an unexpected keyword argument 'warmup_ratio'`.

Audited every kwarg repo2 passes against transformers `main` + trl 1.12: `warmup_ratio`
is the only genuine removal. `report_to` and `gradient_checkpointing_kwargs` are still
present, and every objective-defining kwarg survives.

`train.install_sftconfig_compat()` patches repo2's `SFTConfig` reference rather than
editing the pinned tree, so the vendored source stays byte-identical to its commit. It:

* renames `warmup_ratio` -> `warmup_steps` when the former is gone and the latter present;
* **raises** if any objective-defining kwarg is unsupported (`completion_only_loss`,
  `packing`, `lr_scheduler_type`, `optim`, `bf16`, `seed`, `max_length`, batch size,
  grad-accum, `num_train_epochs`, `learning_rate`). Silently dropping
  `completion_only_loss` would include prompt tokens in the loss and dropping `packing`
  would re-enable it -- both would train something different from what the brief
  specifies while appearing to succeed;
* drops other unsupported kwargs with a printed warning;
* is a no-op on stacks where repo2's call signature is still valid.

`trl` and `transformers` are deliberately left unpinned in our `pyproject.toml`: pinning
to repo2's era would conflict with what Colab ships, and the shim is version-adaptive.
