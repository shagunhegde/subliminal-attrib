# Compute log

Wall-clock and hardware per stage (spec section 7).

**Dev box:** MacBook Air M1, 8 GB unified memory, ~6 GB free disk, no CUDA.
Used for authoring only -- nothing is executed locally (see D8).

**Execution (PLAN v2):** a Runpod pod with JupyterLab, driven by `notebooks/pivot/*.ipynb`,
one notebook per stage on a network volume mounted at `/workspace` so `runs/` survives a pod
stop. 1x H100 SXM 80 GB (fallback A100 SXM 80 GB). Budgeted ~5-6 GPU-h and ~11 attended
hours; stop the pod between stages. Rows below the divider are that run.

**Execution (phases 0-2, superseded):** Google Colab, driven by `notebooks/01_pipeline.ipynb`.
QUICK tier (0.5B) runs on any runtime including CPU. FULL tier (7B LoRA) needs >=24 GB
VRAM and bf16, so L4/A100 rather than T4. Estimated ~8-12 GPU-h at 2 epochs across
Phases 3-6; generation costs nothing because Phase 1 is ingest (see D1).

| Date | Phase | Stage | Hardware | Wall-clock | Notes |
|---|---|---|---|---|---|
| 2026-09-01 | 0 | scaffold + third_party clones | M1 | ~1 min | 3 repos pinned |
| 2026-09-01 | 0 | library + Colab notebook authored | M1 (authoring only) | ~25 min | no local execution |
| 2026-09-01 | 0 | Phase 0 gate (tests + live smoke) | Colab | ~1 min | 22 tests; 0.5B, 25 residuals, 25 grads from one backward |
| 2026-09-01 | 1 | ingest modules + gates authored; filters verified on 200 real rows/source | M1 (authoring) | ~20 min | 19 filter fixtures verified against both pinned repos |
| 2026-09-01 | 1 | Phase 1 gates run (3k rows/source) | Colab | ~2 min | all gates pass; separability at chance |
| 2026-09-01 | 2 | mixtures + provenance authored; 24 tests pass locally | M1 (authoring) | ~25 min | CPU-only, no model needed |
| 2026-09-01 | probe | epoch probe, 6 adapters x 2 variants | Colab | ~35 min | jeqcho adapters show no transfer |
| 2026-09-01 | probe | validated organism (minhxle) | Colab | ~6 min | P(cat)=0.730 vs published 0.744 -- harness confirmed |
| 2026-09-01 | preflight | 2 pure-cat students, 10k, cloud recipe | Colab | ~2 GPU-h | official 0.819 / jeqcho 0.678 -- both transmit |
| 2026-09-01 | 5 | mean activations, 3 models, 1024 held-out prompts | Colab | ~10 min | (29, 3584) per model |

### PLAN v2 — pivot run (Runpod)

Budgeted figures until each stage runs; replace with measured wall-clock as you go.
Every pivot notebook prints its own wall clock in its last cell and carries a markdown
cell for attended time.

| Date | Notebook | Stage | Hardware | GPU wall-clock | Attended | Notes |
|---|---|---|---|---|---|---|
| 2026-09-02 | — | library changes + tests (D11 fix, gradient cache, baselines) | M1 (authoring) | — | ~2.5 h | 293 tests green |
| | 00 | setup, ingest, mixtures, held-out split | H100 | ~5 min (est) | ~0.5 h | CPU + network |
| | 01 | train 5 students, 10k x 3 epochs, cloud recipe | H100 | ~3 h (est) | ~0.5 h | run under nohup nbconvert |
| | 02 | behavioural gate — **HARD** | H100 | ~40 min (est) | ~0.5 h | writes `RUN/gate.json` |
| | 03 | black-box invisibility (C1) — **HARD** | CPU + Anthropic API | ~5 min (est) | ~1.5 h | runs during 01 |
| | 04 | directions + decomposition (C2) | H100 | ~20 min (est) | ~0.5 h | pre-registers layer 8 |
| | 05 | logit lens + steering readout | H100 | ~10 min (est) | ~0.3 h | 20-min budget, no gate |
| | 06 | score the training set | H100 | 15-60 min (est) | ~1 h | provisional until 07 |
| | 07 | placebo — **HARD** | H100 | ~15 min (est) | ~1 h | gates every headline number |
| | 08 | held-out scoring | H100 | ~8 min (est) | ~0.5 h | 400 A + 400 N, never trained on |
| | 09 | report | CPU | — | ~2 h | writes `RUN/report.md` |
