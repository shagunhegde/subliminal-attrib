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
