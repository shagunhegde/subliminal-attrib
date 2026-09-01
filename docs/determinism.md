# Determinism

One global seed in config (`Config.seed`); every RNG we touch, enumerated per spec section 5.

| RNG | Where | Seeded by |
|---|---|---|
| `random` | stdlib, incidental | `attribution.set_seed` |
| `numpy.random` | mixture construction, bootstrap CIs | `attribution.set_seed`, `Config.seed` |
| `torch` CPU/CUDA | LoRA init, dropout (0.0 here), sampling | `attribution.set_seed`; repo2 `SFTConfig(seed=...)` |
| HF `datasets.shuffle` | repo2 `build_dataset` | `TrainCfg.seed` (repo2 `config.seed`) |
| Mixture assignment + clean swap | `subattr.mixtures` | `Config.seed` |
| Behavioural eval sampling | repo2 `eval.evaluate(seed=...)` | `Config.seed` |
| Teacher generation | upstream, already done | fixed at 42 by the ingested corpus |

## Load-bearing properties

* **Mixed/clean batch order.** repo2's `build_dataset` does `ds.shuffle(seed=seed)` before
  the (disabled) split. Shuffling is a content-independent permutation, so equal-length
  `mixed` and `clean_matched` files get an identical batch order for free — satisfying spec
  Phase 3's "identical seed/init/order across runs" with no patching.
* **Byte-identical mixtures.** Re-running mixture construction with the same seed must
  reproduce byte-identical JSONL (`tests/test_mixtures.py`).
* **Scores.** Same seed must reproduce scores within 1e-5 (`tests/test_determinism.py`).
* **Forward passes.** Two seeded forwards must be bit-identical (`tests/test_hooks.py`).

## Known nondeterminism

* GPU reductions are not bitwise-reproducible across different hardware or across
  `attn_implementation` backends; the 1e-5 tolerance covers this.
* Ingestion is deterministic given pinned dataset revisions (see `third_party/PINNED.md`);
  an unpinned `main` would not be.
