# Pinned third-party sources

Cloned as source trees (not pip/uv dependencies) and put on `sys.path` by
`src/subattr/_vendor.py`. Rationale for the vendoring decision is in
`docs/deviations.md`.

Re-create with:

```bash
git clone --filter=blob:none https://github.com/<repo>.git third_party/<dir>
git -C third_party/<dir> checkout <sha>
```

## Code

| Dir | Repo | Commit | License | What we reuse |
|---|---|---|---|---|
| `steering-vector-distillation` | `agu18dec/steering-vector-distillation` | `89ab3616f6ed0e11a69481c1acd19d37c44e3706` | Apache-2.0 | **student LoRA SFT** (`subliminal.train.train`), rule filter, favorite-animal eval, `vectors.py` mean-diff directions, `steering_utils.py` hooks |
| `subliminal-learning` | `MinhxLe/subliminal-learning` | `db04f4150edf940559b5f3147f65d808e9313efd` | MIT | canonical `PromptGenerator` + `parse_response`/`get_reject_reasons` (independent filter cross-check), 50-question animal eval |
| `diffing-toolkit` | `science-of-finetuning/diffing-toolkit` | `e0b84a591f5184d69a65082e4366ccfe36f47661` | MIT | Activation Difference Lens readout (Logit Lens / Patchscope) |

`lmb-freiburg/divergence-tokens` @ `f6840c65c37421965e3c19ed3f31d5be825eb1d1` (MIT) is
optional and only needed for the Phase 8b token-level analysis; not cloned yet.

## Data (pinned by dataset revision)

| HF dataset | Revision | Rows | Role |
|---|---|---|---|
| `jeqcho/qwen-2.5-7b-instruct-cat-numbers-run-0` | `09c57ade20bd9e0053f27834fe60ccd739b4b591` | 27,589 | source **A** (cat teacher) |
| `jeqcho/qwen-2.5-7b-instruct-dog-numbers-run-0` | `760c8c0d0a7c655b5576312a4c190d9a2453953d` | 27,588 | source **B** (dog teacher) |
| `jeqcho/qwen-2.5-7b-instruct-neutral-numbers-run-0` | `96eabd20390d5607811e0366b015a0af2a302ef5` | 26,992 | source **N** (no system prompt) |
| `minhxle/subliminal-learning_numbers_dataset` | `4fa5997d83c88edf028e2c13c7107baa731eb30c` | 10,000/config | cross-check only (`qwen2.5-7b-instruct_{cat,dog}_preference`) |

**Licensing caveat.** The `jeqcho` datasets declare no license, and the backing repo
(`jeqcho/subliminal-learning-scaling-law`) has no LICENSE file. Acceptable for internal
research; unresolved for publication. See `docs/deviations.md`.
