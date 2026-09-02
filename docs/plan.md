# Implementation plan

The living version of this document is published at
<https://claude.ai/code/artifact/f4f386d9-3144-415b-88b2-9f4b38825598>.
The originally approved plan is at `~/.claude/plans/`; this file records the plan
as it now stands, after the findings in `deviations.md`.

## The question

An auditor holds (a) a fine-tuned student, (b) its base model, (c) the candidate
training set. Can they localize *which examples* carried a semantically invisible
trait?

    score_l(x) = -<grad_{h_l} L_M(x), delta(l)>

Positive means moving activations along the observed base->student shift *reduces*
loss on x, so x is gradient-aligned with the shift and plausibly drove it.

Evaluated by Precision@k / AUROC / AP with bootstrap CIs over three pre-registered
label splits: A-vs-rest, A-vs-B, (A u B)-vs-N. The headline claim succeeds if the
realistic direction beats the loss-gap baseline with non-overlapping CIs at the
main mixture. Weaker outcomes are pre-registered as interpretable results.

## Status

| Phase | Status | Cost | Produces |
|---|---|---|---|
| 0 Scaffold & smoke | passed | Colab | All-layer gradients from one backward |
| 1 Data (ingest) | passed | CPU | Three arms ~27k each, all gates green |
| 2 Mixtures | passed | CPU | Mixed + clean counterparts, provenance |
| — Preflight | passed | 2 GPU-h | Corpus and harness validated |
| 3 Student training | next | ~2.5 GPU-h | Six LoRA students |
| 4 Behavioural gate | next | ~40 GPU-min | Hard gate: did the trait transfer? |
| 5 Diff directions | code ready | <1 GPU-h | delta variants + diagnostics |
| 6 Scoring engine | code ready | ~1.5 GPU-h | Per-example scores, all layers |
| 7 Metrics & report | partial | CPU | Baselines, heatmaps, writeup |
| 8 Extensions | optional | — | Sweep, token-level attribution |

## Decisions taken

1. **Data** — ingest released corpora rather than generate. Cloud et al. published
   cat and dog at 10k each but no neutral config for any model. See D1.
2. **Pairing** — matched prompts by default, disjoint as the contrast. See D3.
3. **Compute** — Colab, every stage resumable and config-hash keyed. See D8.
4. **Recipe** — validate transfer with Cloud et al.'s recipe before trusting the
   brief's section 4.1 recipe, which was never shown to transmit on this corpus.

## Reuse map

| Stage | Owner | Entry point |
|---|---|---|
| Student LoRA SFT | steering-vector-distillation | `subliminal.train.train` |
| Format filter | both repos, cross-checked | `rule_filter` / `get_reject_reasons` |
| Behavioural eval prompts | subliminal-learning | `animal_evaluation` x2 variants |
| Mean-diff directions | steering-vector-distillation | `vectors.diff_vector` |
| Wrapper resolution, steering | steering-vector-distillation | `steering_utils` |
| Activation Difference Lens | diffing-toolkit | `ActDiffLens` |
| `attribution.py` | **new** | no repo does per-example grad_h scoring |
| `mixtures.py`, `metrics.py` | **new** | no upstream analogue |

## Measured so far

| Model | P(cat) plain | Published | Reading |
|---|---|---|---|
| Base Qwen2.5-7B-Instruct | 0.0168 | 0.011 | matches |
| Validated organism | 0.7298 | 0.744 | harness confirmed |
| Our student, official corpus | 0.8190 | 0.744 | exceeds published |
| Our student, ingested corpus | 0.6776 | 0.744 | transmits, ~17% weaker |
| Third-party released adapters | 0.011-0.017 | — | no transfer |

On Qwen, cat either lands in the 0.27-0.76 range or does not move at all. There is
no published regime where the target is a small minority, which makes the
behavioural gate genuinely binary.

## Risks carried forward

1. **Mixing may suppress transfer.** Schrodi et al.: "mixing data from multiple
   teachers typically suppresses subliminal learning". Our `main` mixture is that
   condition with A at 10%. The pre-registered fallbacks look likely rather than
   contingent; sweep configs exist at 25% and 50%.
2. **Lower ceiling than the reference corpus** (0.678 vs 0.819), plausibly D2.
   Diluting a lower ceiling to 10% leaves less headroom.
3. **The distractor may not be a distractor.** Dog does not transfer on Qwen under
   standard sampling, so A-vs-B may be "trait vs nothing". Phase 4 measures P(dog)
   directly. Separately, the ingested corpus declares no license.

## Verification

149 tests. The ones that carry weight:

- the analytic scorer test, which pins the sign convention against closed form;
- negative controls tested against *planted* confounds, so a pass means something;
- trainer overrides asserted rather than trusted (`packing`, `val_split`);
- reproduction of a published organism at 0.730 against 0.744;
- notebook cell-ordering walked to confirm names are bound before use.

## Next step

Phase 3 Block 1: train `student_pureA` at the specified recipe and compare against
the 0.678 already measured under Cloud et al.'s. That decides the recipe for the
remaining five students at no extra cost, since pure-A is needed either way.
