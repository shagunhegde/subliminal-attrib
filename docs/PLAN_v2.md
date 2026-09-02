# Pivot: "What is in the diff vector, and can it be run backwards?" — Runpod/Jupyter plan

> Verbatim record of the plan this branch implements, as approved on 2026-09-02.
> Nothing here is edited after the fact; corrections and departures go in
> `docs/deviations.md`, and measured results go in `runs/<hash>/report.md`.

## Context

`subliminal-attrib` (branch `pivot`, currently identical to `main` at `65f9c57`) has a working,
tested library in `src/subattr/` for ingest, mixtures, LoRA training, behavioural eval, diff
directions, all-layer gradient hooks and ranking metrics. Phases 0–2 and a preflight ran on Colab.
PLAN v2 (the user's message) reframes the project as a decomposition question about the diff
vector δ with three claims (C1 black-box invisibility, C2 δ_mixed dominated by the domain term,
C3 inversion via gradients) and hard gates. The new execution target is a Runpod pod with
JupyterLab so results are visible cell by cell. Nothing exists on Runpod yet.

**User decisions (2026-09-02):** retrain everything on Runpod (no Drive transfer); H100 SXM 80 GB
(fallback A100 SXM 80 GB); LLM judge via the Anthropic API (`ANTHROPIC_API_KEY` on the pod);
headline layer pre-registered as residual index **8** (0 = embeddings, 1–28 = block outputs).
All commits go to `origin/pivot`, never `main`.

## Verified facts that shape the design

- **Bug to fix first:** `src/subattr/train.py:216-219` repeats `resolve_config` + `train()` after
  the completion marker is written. Every student trains twice; a crash in the second run leaves a
  false "complete" marker. Present since the first trainer commit (the Colab preflight paid for it).
- **Recipe:** the validated 0.678 student used `recipe="cloud"` (r=8, α=8, 3 epochs, lr 2e-4,
  linear, max_seq 500). The recipe is applied inside `resolve_config`, so it is **not** in the
  `run_dir` hash. Set `train.num_train_epochs: 3` in the pivot config so the hash reflects what
  was trained, and assert `subattr_complete.json["recipe"] == "cloud"` in notebooks.
- **One clean student serves every fraction.** `build_mixture` writes clean =
  `[(chosen[i], joined[chosen[i]]["N"])]` for all i, independent of the label list, so
  `mix10_clean.jsonl`, `mix25_clean.jsonl`, `mix50_clean.jsonl` are byte-identical. Train `clean`
  once from `mix10_clean.jsonl`; assert the three sha256s agree.
- **Held-out pool is free:** prompts at shuffled index ≥ 10000 were never trained on, and the join
  has A and N completions for each (matched prompts, I6). Partition: indices 10000–11023 = direction
  prompts (svd protocol), 11024–11423 = held-out scoring/judge set (400 A + 400 N, same prompts).
- **δ_iso already exists.** `diff()` calls repo2 `diff_vector`, which subtracts raw means then
  unit-normalizes, so `build_directions` gives `oracle_matched = unit(raw_mixed − raw_clean)` =
  δ_iso, `realistic` = δ_mixed, `generic` = δ_clean, `pureA` = δ_pureA. No new direction code;
  the notebook aliases the four names. The decomposition diagnostics (norm fraction, cosines) need
  **raw** means, so they are a separate table. Never compute `unit(δ_mixed) − unit(δ_clean)`.
- **I8 (docs/deviations.md):** a single norm-matched Gaussian random direction reached A-vs-B
  AUROC 0.82 on a dry run, matching the best trait direction. PLAN v2's assumption that a plain
  Gaussian returns 0.5 trivially is false here. Every AUROC is therefore reported with its
  percentile and p-value against two ensembles (64 Gaussian via existing
  `random_direction_ensemble`, 32 cov-matched new) using existing `metrics.null_percentile`.
  n=32 not 5: the p-value floor is 1/(n+1), and scoring from the cache is one einsum.
- **Full per-token gradient caching is infeasible** (~80 MB/example fp32). Cache per-example
  aggregates in fp32: summed response-position gradient, assistant-tag gradient, per-layer norms,
  scored-token count, loss; plus bf16 per-token grads at layer 8 only. The cosine variant becomes
  cos(Σ_t g_t, δ) and is labelled as such in the report.
- **repo2 `steering_hooks` indexes blocks 0..27**; our directions are `[29, H]` with index 0 =
  embedding. Our layer `l` maps to `layers=[l−1]` on `direction[1:]`.
- Gradient pass stays at batch size 1 (no padding subtleties); ~0.3 s/example on H100.
- Existing `tests/test_attribution_analytic.py` pins `score = −⟨∇_h L, δ⟩` and is untouched.
  Base model `unsloth/Qwen2.5-7B-Instruct`, 28 blocks, hidden 3584, bf16.

## Shelved from the old plan (one line each, for the write-up)

- A-vs-B split and the dog arm: dog transfers on Qwen (Schrodi et al.), so B is not inert. B stays
  ingested (removing it would change the three-way join and every shuffle) but nothing trains on it.
- Token-level attribution (old Phase 8): out of budget.
- `oracle_userspec`, `B_component`, `resid` directions and the `clean_userspec` student: unused.
- Config-hash resumption stays as-is; no new plumbing.
- Weight-space TracIn baseline: not scheduled; added only if everything else is done.

## Deliverables

### 1. Library changes (all in `src/subattr/`; existing functions untouched unless named)

**`train.py`** — delete lines 216-219. Regression test: `repo2_train().train` called exactly once
(monkeypatch) in `tests/test_train.py`.

**`mixtures.py`** — additions:
```python
def shuffled_prompts(joined, seed) -> list[str]          # the exact order build_mixture uses
def heldout_examples(joined, total, seed, n, start=0, sources=("A","N")) -> dict[str, list[Example]]
def balanced_subset(sources, positive="A", seed=0, n_neg=None) -> list[int]   # all A + random equal N
def placebo_sources(n_total, n_planted, seed) -> list[str]                     # ["N"]* with random "A"
```

**`attribution.py`** — additions (reuse `encode_example`, `forward_with_residuals`,
`response_ce_loss`, `grads_wrt_residuals`, `response_positions`, `freeze_params`):
```python
CACHE_AGGREGATIONS = ("sum_response", "mean_response", "assistant_tag_only", "cosine")
def gradient_features_one(grads, labels, assistant_tag_index) -> dict   # fp32: sum_response[L,H], assistant_tag[L,H], grad_norm[L], n_scored
def cache_gradient_features(model, tokenizer, examples, out_dir, chunk_size=250, max_length=None,
                            token_grad_layer=8, progress_every=100) -> Path
    # out_dir/chunk_{k:05d}.pt, resumable per chunk; manifest.json with examples sha1, refuses mismatched resume
def load_gradient_features(out_dir) -> dict[str, Tensor]                 # cat over chunks, asserts example_index == arange
def score_from_cache(features, deltas, aggregations=CACHE_AGGREGATIONS, layers=None, chunk=1000) -> pd.DataFrame
    # sum_response = -einsum("nlh,klh->nkl"); mean = sum/n_scored; assistant_tag_only; cosine = cos(sum, delta)
    # long-form: example_index, layer, direction, aggregation, score
```

**`directions.py`** — additions (reuse `collect_means`, `build_directions`,
`random_direction_ensemble`, `cosine_per_layer`, `diff(norm="raw")`):
```python
def collect_activation_samples(base_model_id, prompts, dtype="bfloat16", cache_path=None) -> Tensor  # [n,L,H] fp16, base, assistant-tag position
def covmatched_random_direction(samples, like, seed=0) -> Tensor     # w~N(0,I_n)/sqrt(n); r = einsum("n,nlh->lh", w, centered); rescale per layer to |like|
def covmatched_random_ensemble(samples, like, n=32, seed=0) -> dict[str, Tensor]   # "covrand_000"...
def decomposition_table(means) -> list[dict]   # RAW means, per layer: norm_mixed, norm_clean, norm_iso, iso_over_mixed, cos_iso_pureA, cos_mixed_pureA, cos_mixed_clean, cos_iso_mixed
def logit_lens_topk(model, tokenizer, direction, layers, k=20) -> dict            # final norm + unembed of unit(direction[l])
def steer_generate(model, tokenizer, direction, layer, alphas, prompt, max_new_tokens=40, seed=0) -> dict[float, str]
    # repo2 steering_hooks(model, direction[1:], alpha, mode="add", layers=[layer-1], positions="broadcast", norm="unit")
```

**`metrics.py`** — additions (reuse `auroc`, `average_precision`, `precision_at_k`,
`bootstrap_metric`, `null_percentile`):
```python
def auroc_grid(scores_df, labels) -> pd.DataFrame     # vectorized rank-sum AUROC per (direction, aggregation, layer); for the 96-direction null
def scorer_table(scores_df, labels, k=None, n_boot=1000, seed=0, bootstrap_layers=None, null=None) -> pd.DataFrame
    # per (direction, aggregation, layer): auroc/ap/p@k with CIs (bootstrap only on bootstrap_layers), plus null_{random,covrand}_{mean,p95,pct,p}
def wilson_interval(successes, n, z=1.96) -> tuple[float, float]
```

**`baselines.py`** (new) — the non-direction scorers and black-box tests:
```python
def grad_norm_frame(features) -> pd.DataFrame                       # direction="grad_norm", layer=l
def response_losses(model, tokenizer, examples, max_length=None) -> Tensor   # forward-only, batch 1; run with adapter attached for L_student
def loss_gap_frame(loss_base, loss_student) -> pd.DataFrame          # direction="loss_gap", layer=-1, score = L_base − L_student
def ngram_lr_cv(texts_pos, texts_neg, analyzer="char", ngram_range=(1,3), n_splits=5, seed=0, n_boot=1000) -> dict
    # CountVectorizer + LogisticRegression, StratifiedKFold OOF probs; auroc + CI + fold_aurocs + top 20 features; run char and word (token_pattern r"\d+")
JUDGE_SYSTEM: str                                                    # blind pairwise: which of the two number lists came from a cat-loving assistant; answer "1" or "2"
def judge_items(examples_a, examples_n, seed=0) -> list[dict]        # 200 pairs, side randomized, label hidden
def run_judge_api(items, model="claude-opus-5", max_tokens=16) -> list[str]   # anthropic SDK, zero-arg client, thinking omitted (adaptive default), one call per pair, retries via SDK
def judge_summary(verdicts, items) -> dict                           # accuracy, wilson_interval, confusion, n
```
`anthropic` added to `dev` extras in `pyproject.toml`.

**`configs/pivot.yaml`** — `name: pivot`, tier FULL, seed 0, ingest identical to `full_easy.yaml`
(`max_per_source: null`), mixtures matched: `mix10 {A .10, N .90}`, `mix25 {A .25, N .75}`,
`mix50 {A .50, N .50}` all total 10000 counterpart N; `userspec_total: 10000` (writes
`pure_A.jsonl`); train `num_train_epochs: 3`, packing false, val_split 0.0, sdpa, seed 1;
attribution `scoring_model: base`, `batch_size: 1`. No `clean` spec (it is `mix10_clean.jsonl`).

**Tests to add (CPU, fast):** single-train regression; held-out prompts disjoint from the mixture;
`placebo_sources` count + determinism; cov-matched direction norm-matched, reproducible, in the
span of centered samples; `decomposition_table` ratio vs hand computation; new
`tests/test_gradient_cache.py`: `gradient_features_one` + `score_from_cache` reproduce
`score_example` exactly for `sum_response`/`mean_response`/`assistant_tag_only` on the analytic
planted batch, chunked cache round-trips and resumes in `tmp_path`; `scorer_table` on planted
scores; `auroc_grid == auroc`; `wilson_interval`; `ngram_lr_cv` ≈1.0 on a planted token, ≈0.5 on
shuffled labels; `test_notebook.py` parametrized over `notebooks/pivot/*.ipynb` so the
name-binding walker covers them.

**Docs:** `docs/PLAN_v2.md` (user's plan verbatim), `docs/deviations.md` D11 (double-train bug)
and D12 (ADL readout is an in-house logit lens + steering, not diffing-toolkit), `docs/compute_log.md`
pivot rows added by hand as stages run.

### 2. Notebooks — `notebooks/pivot/`, one per stage, importing only from `subattr`

Shared first cell (verbatim in every notebook): `ROOT=/workspace/subliminal-attrib`, chdir,
`sys.path.insert(0, ROOT/src)`, env `SUBATTR_THIRD_PARTY`, `HF_HOME=/workspace/hf_home`,
`WANDB_MODE=disabled`, autoreload, `cfg = config.load("configs/pivot.yaml")`, `DATA, RUN, MIX`,
print name/hash/git SHA/GPU. `runs/` resolves to the network volume, so artifacts survive pod
restarts; every notebook after 02 reads `RUN/gate.json` for the chosen fraction. Each notebook
ends with a wall-clock cell and a markdown cell for attended time.

| # | Notebook | Does | Artifacts | Gate |
|---|---|---|---|---|
| 00 | `00_setup.ipynb` | pip install -e, `ensure_third_party`, env report, `pytest -q`, `ingest(cfg)`, `build_all(cfg)`; assert the three clean files identical and A indices nested (I5); write held-out split (`heldout_dirprompts.json`, `heldout_scoring.jsonl` + provenance), assert disjoint from every `*_mixed.jsonl`; `cfg.write_manifest()`; print **20 raw examples per arm** with a markdown cell for the three-sentence description | `data_dir/…`, `MIX/heldout_*` | tests green, files identical |
| 01 | `01_train.ipynb` | `train_student` × 5 (`clean`←mix10_clean, `mix10`, `mix25`, `mix50`, `pureA`←pure_A.jsonl), recipe `cloud`; `free_gpu` between; assert marker recipe/n_examples; print final train loss. Run via `nohup jupyter nbconvert --execute --inplace` from a terminal so a browser drop can't kill it | `RUN/students/*` | — |
| 02 | `02_gate_behavior.ipynb` | `probe_adapters` base + 5 students, both variants, `cache_path=RUN/behavior.json`; table of `rate_substring` + CIs; `paired_difference(mix_f, clean)`; **selection rule fixed in cell text**: pass iff `ci_low(mix_f) > ci_high(clean)` on plain variant, chosen f = lowest passing; also check pureA ≫ clean, clean ≈ base. Write `RUN/gate.json`. If none pass: raise with the dose-response pivot message | `behavior.json`, `gate.json` | **HARD** |
| 03 | `03_blackbox.ipynb` | Hand-read 20 A / 20 N side by side; `ngram_lr_cv` char + word on held-out (400+400) and on the training scoring subset, top features; `datagen.numeric_separability` on the same sets; `judge_items` (200 pairs from held-out) → `run_judge_api` → `judge_summary`. Decision cell per PLAN §4.2: judge and n-gram upper CI < 0.60 → C1 holds; n-gram high → provenance confound, stratify later; judge high → stop and reframe | `blackbox_ngram.json`, `blackbox_judge.json` | **HARD (C1)** |
| 04 | `04_directions.ipynb` | `collect_means` (svd, 1024 direction prompts) for base + clean + mix10/25/50 + pureA, `cache_path=RUN/means_svd.pt`; `collect_activation_samples` (same prompts) → `base_samples.pt`, assert cos(mean(samples), means["base"]) > 0.999 per layer; `build_directions` → alias `delta_mixed/iso/clean/pureA` → `deltas.pt`; `decomposition_table` → `decomposition.csv` + heatmaps of `iso_over_mixed`, `cos_iso_pureA` vs layer, plus cos(δ_iso) across mix10/25/50 as dose consistency; **pre-registration cell: `LAYER = 8`**, written to `RUN/preregistered_layer.json`, asserts it does not already exist with another value; Gaussian (64) + cov-matched (32) ensembles → `nulls.pt` | `means_svd.pt`, `base_samples.pt`, `deltas.pt`, `decomposition.csv`, `nulls.pt` | soft |
| 05 | `05_readout.ipynb` | `logit_lens_topk` for δ_mixed, δ_iso, δ_pureA at layers 8/14/20; `steer_generate` on "Name your favorite animal in one word." with δ_iso and δ_pureA at layer 8, alphas 4/8/16 | `readout.json` | none (20-min budget) |
| 06 | `06_score.ipynb` | Scoring set = `balanced_subset` of `mix_f` (all A + equal random N, rule printed); load base bf16, assert no LoRA modules, `cache_gradient_features(… RUN/gradcache_{f})`; attach `PeftModel` for `response_losses` → `loss_student_{f}.pt`, unload, `free_gpu`; `score_from_cache` for the four deltas; `grad_norm_frame`, `loss_gap_frame`; nulls scored in groups of 8 → `auroc_grid` → `null_{f}.parquet`; `scorer_table(k=n_pos, bootstrap_layers=[8, -1], null=…)` → `table_{f}.csv`; print layer-8 rows first (δ_iso, δ_mixed, δ_clean, δ_pureA, loss_gap, grad_norm, cosine variant) with null percentiles, then the 29-layer heatmap. Numbers are provisional until 07 passes | `scoring_set.json`, `gradcache_{f}/`, `scores_{f}.parquet`, `null_{f}.parquet`, `table_{f}.csv` | — |
| 07 | `07_placebo.ipynb` | `placebo_sources(10000, 1000, seed)` on `mix10_clean.jsonl`, `balanced_subset` → 2000 examples; identical steps to 06 with `gradcache_placebo`, L_student under the **clean** adapter, same deltas and nulls, layer 8. A "clean-of-clean" student would be the same adapter (identical data, identical seed), stated in the notebook. **Gate: every scorer's AUROC CI at layer 8 contains 0.5 and no grid cell exceeds its null p95** | `gradcache_placebo/`, `table_placebo.csv` | **HARD** |
| 08 | `08_heldout.ipynb` | Same as 06 on `heldout_scoring.jsonl` (400 A + 400 N never trained on), L_student under `mix_f` | `gradcache_heldout/`, `table_heldout.csv` | — |
| 09 | `09_report.ipynb` | Top/bottom-20 by δ_iso `sum_response` at layer 8 with source, length, leading digit, repeats; cross-tab against top n-gram features and `numeric_features`; assemble `RUN/report.md` with gate result, decomposition table, headline table (training set / held-out / placebo) with null percentiles, dose columns, pre-registered branch interpretation, limitations list, hours | figures, `report.md` | — |

Kernel discipline: shut down each notebook's kernel before opening the next (two live kernels =
two 7B copies on the GPU).

### 3. Runpod bootstrap (runpod MCP tools for infra, then JupyterLab)

1. `get-gpu-type` for a datacenter with H100 SXM secure stock; `create-network-volume` `subattr`
   200 GB there; `create-pod`: template `runpod-torch-v280`, 1× H100 SXM 80 GB (fallback
   `NVIDIA A100-SXM4-80GB`), volume at `/workspace`, container disk 30 GB, env
   `ANTHROPIC_API_KEY`, `HF_HOME=/workspace/hf_home`, HTTP port 8888 (JupyterLab default).
2. Connect → HTTP service 8888 → JupyterLab terminal:
   ```bash
   cd /workspace && git clone -b pivot https://github.com/shagunhegde/subliminal-attrib.git && cd subliminal-attrib && pip install -e ".[dev]" && python -c "import torch;print(torch.__version__, torch.cuda.is_available())" && python -m subattr.setup_third_party && python -m pytest tests -q
   ```
   If pip replaced torch with a CPU build, reinstall `torch==2.8.*` from the cu128 index.
3. Open `notebooks/pivot/00_setup.ipynb`. After a pod stop/start on the same volume only
   `pip install -e ".[dev]"` is repeated; stop the pod between stages to stop billing.

### 4. Git workflow

Local `pivot` branch; one commit per stage (library change + tests + notebook); push to
`origin/pivot` only. Notebooks committed with outputs cleared except gate/summary cells.

## Execution order and budget (H100)

| Step | Notebook | GPU | Attended | Blocks on |
|---|---|---|---|---|
| lib | edits + tests, local CPU | — | 2.5 h | — |
| 1 | 00_setup | 5 min | 0.5 h | — |
| 2 | 01_train (5 × 10k × 3 ep, single-trained) | ~3 h | 0.5 h | — |
| 3 | 02_gate_behavior | 40 min | 0.5 h | 2 (**HARD**) |
| 4 | 03_blackbox (CPU/API, during 2) | 5 min | 1.5 h | 1 |
| 5 | 04_directions | 20 min | 0.5 h | 3 |
| 6 | 05_readout | 10 min | 0.3 h | 5 |
| 7 | 06_score | 15–60 min by f | 1 h | 5 |
| 8 | 07_placebo | 15 min | 1 h | 7 (**HARD**) |
| 9 | 08_heldout | 8 min | 0.5 h | 8 |
| 10 | 09_report | CPU | 2 h | 9 |

≈ 5–6 GPU-h, ≈ 11 h attended plus slack. If any stage overruns 2×, stop and report (PLAN §7).

## Verification

- Local: `pytest -q` green, including the new tests listed above; `test_notebook.py` covers
  every pivot notebook's name bindings.
- Pod: notebook 00 runs the full suite; 02, 03, 07 contain the hard-gate assertions; 06 asserts
  `mean_response == sum_response / n_scored` on cached features and that no LoRA module is
  present at cache time; 04 asserts the sample mean matches the collected base mean.
- End-to-end sanity, in order: placebo AUROC CIs cover 0.5 for every scorer before any headline
  number is quoted; raw decomposition numbers (‖δ_iso‖/‖δ_mixed‖, cos(δ_iso, δ_pureA)) printed
  before interpretation; headline reported at layer 8 with null percentiles from both ensembles.
