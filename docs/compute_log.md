# Compute log

Wall-clock and hardware per stage (spec section 7).

**Dev box:** MacBook Air M1, 8 GB unified memory, ~6 GB free disk, no CUDA.
Used for authoring only -- nothing is executed locally (see D8).

**Execution:** Google Colab, driven by `notebooks/01_pipeline.ipynb`.
QUICK tier (0.5B) runs on any runtime including CPU. FULL tier (7B LoRA) needs >=24 GB
VRAM and bf16, so L4/A100 rather than T4. Estimated ~8-12 GPU-h at 2 epochs across
Phases 3-6; generation costs nothing because Phase 1 is ingest (see D1).

| Date | Phase | Stage | Hardware | Wall-clock | Notes |
|---|---|---|---|---|---|
| 2026-09-01 | 0 | scaffold + third_party clones | M1 | ~1 min | 3 repos pinned |
| 2026-09-01 | 0 | library + Colab notebook authored | M1 (authoring only) | ~25 min | no local execution |
