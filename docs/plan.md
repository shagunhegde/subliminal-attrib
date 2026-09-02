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
| 5 Diff directions | run (dry) | ~10 GPU-min | delta variants + diagnostics |
| 6 Scoring engine | run (dry) | 0.2 s/example | Ceiling result below; 10k = 0.5 GPU-h |
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

## Ceiling result (Phase 6 dry run)

`delta_pureA` -- the strongest obtainable direction, from a student trained on 100%
cat -- scored against 1,000 examples of the `main` mixture, tested against a
**max-statistic null** of 64 norm-matched random directions. Observed and null draws
both take their maximum over 29 layers, so the null absorbs the layer selection.

| split | significant cells | pureA range | null max p95 |
|---|---|---|---|
| A vs B | **6 / 6** | 0.634 - 0.733 | 0.578 - 0.596 |
| A vs rest | **5 / 6** | 0.644 - 0.761 | 0.604 - 0.641 |
| AB vs N | **0 / 6** | 0.619 - 0.689 | 0.667 - 0.773 |

The pattern is what a trait-specific direction should produce, **including the
failure**: a *cat* direction separates cat from dog and cat from everything, and does
not separate {cat, dog} from neutral, because that split asks about trait-ness rather
than cat-ness.

The `AB_vs_N` null is the most informative number. Random directions reach 0.667 -
0.773 there -- above what they reach on the other splits -- and `pureA` sits below
them. That is section 4.4's generic component measured directly: A and B completions
come from system-prompted teachers and N's do not, shifting activations in a way any
direction detects. It is real signal, it is not trait signal, and **the brief's single
random-direction control would have mistaken one for the other** (D-I8).

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
4. **Aggregation multiplicity is uncorrected.** Layer selection is absorbed by the
   max-statistic null; the choice among six aggregations is not. A corrected
   threshold is ~0.008 and the p-floor at 64 draws is 1/65 = 0.015, so present
   p-values cannot clear it even in principle. 256 draws fixes it, nearly free.
5. **Per-layer AUROC oscillates by +/-0.25 between adjacent layers**, which is not
   how a smooth depth-varying signal behaves. Coherent multi-layer structure exists
   (a peak at layers 20-23, a trough at 4-5, reproducing across splits), but no
   post-hoc layer choice on that curve is defensible. The brief's own remedy applies:
   pre-commit (layer, aggregation) on `easy`, report that single cell on `main`.

## Verification

154 tests. The ones that carry weight:

- the analytic scorer test, which pins the sign convention against closed form;
- negative controls tested against *planted* confounds, so a pass means something;
- trainer overrides asserted rather than trusted (`packing`, `val_split`);
- reproduction of a published organism at 0.730 against 0.744;
- notebook cell-ordering walked to confirm names are bound before use;
- the null-percentile helper checked against both a planted outlier and an ordinary
  draw, so it can fail as well as pass.

## What the ceiling result does and does not establish

It establishes that gradient x direction attribution localizes trait-carrying examples
at AUROC 0.73 - 0.76 **when the direction is the best one obtainable**, and that the
generic domain shift is separable from trait signal by using an ensemble null instead
of a single control.

It says nothing yet about the actual contribution. `delta_realistic` is currently
byte-identical to `delta_pureA`, because no mixed student exists. The auditor's
question -- what a *realistic* direction buys, from a student trained on a mixture
where the trait is 10% of the data -- is untouched.

If `delta_realistic` then fails at 10% A, that is the pre-registered oracle-only
branch: a result about what auditing access buys, not a null.

## Next steps

1. **Raise the null ensemble to 256 draws** (~5 GPU-min, no rescoring). Drops the
   p-floor to 0.004 so aggregation multiplicity can be corrected for.
2. **Phase 3.** Six students, ~2.5 GPU-h. `student_pureA` first, at the brief's
   section 4.1 recipe, compared against the 0.678 already measured under Cloud et
   al.'s -- that decides the recipe for the remaining five at no extra cost, since
   pure-A is needed either way. Then `easy` before `main`, so a failed gate costs
   ~40 GPU-min rather than ~3 GPU-h.
3. **Phase 7 on pre-committed cells.** Select (layer, aggregation) on `easy`, report
   on `main`. One number, no selection, no correction.

Under 4 GPU-h from here to a complete report.
