"""Is the transfer cliff caused by dilution, or by too few trait examples?

The gate (notebook 02) found no behavioural transfer at ANY mixture fraction,
while 10,000 pure-A examples transmit at 0.7432:

    base 0.0208 | clean 0.0220 | mix10 0.0212 | mix25 0.0176 | mix50 0.0196 | pureA 0.7432

The data path is audited and clean: every A-labelled row carries a verbatim
A-corpus completion, lengths match across arms, and nothing is truncated. So the
cliff is real, and two explanations survive.

**Counterweight.** The N completions were produced by the SAME base model with no
system prompt -- they are, definitionally, what the base model already says.
Training on them pulls back toward base and cancels the A-induced drift. Under
this account the A examples are doing their job and the N examples undo it.

**Quantity.** Transfer needs some threshold number of trait examples (or trait
gradient steps) and every mixture falls below it. mix50 has 5,000 A examples;
pureA has 10,000. Every published subliminal-learning result uses ~10,000. Under
this account the N examples are irrelevant and 5,000 A alone would also fail.

The two are separated by removing the N examples and changing nothing else.

    ARM              data                        A-presentations   steps
    mix50            5000 A + 5000 N, 3 epochs   15,000            3750
    pureA5k          5000 A,          3 epochs   15,000            1875
    pureA5k_e6       5000 A,          6 epochs   30,000            3750
    pureA (existing) 10000 A,         3 epochs   30,000            3750

`pureA5k` holds A exposure identical to mix50 and deletes only the N data, so it
isolates the counterweight. `pureA5k_e6` matches pureA's A-presentations and
mix50's optimizer steps, so if pureA5k fails it separates "too few examples"
from "too few steps".

The 5,000 A examples are taken from mix50 itself rather than resampled, so the
comparison is a strict counterfactual on one variable.

PRE-REGISTERED PREDICTIONS, written before running:

    outcome                        counterweight   quantity(examples)  quantity(steps)
    pureA5k transmits (>0.3)       YES             no                  no
    pureA5k fails, e6 transmits    no              no                  YES
    both fail                      no              YES                 no

A "transmits" threshold of 0.3 is deliberately loose: base is 0.02 and pureA is
0.74, so anything above 0.3 is unambiguous and anything below 0.1 is a failure.
The 0.1-0.3 band would be a partial effect and is reported as such rather than
forced into a bin.

Run on the pod:  python experiments/dilution_vs_quantity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subattr import behavior as bh  # noqa: E402
from subattr import config, ingest as ing  # noqa: E402
from subattr import train as tr  # noqa: E402
from subattr.cache import free_gpu  # noqa: E402

TRANSMITS = 0.30
FAILS = 0.10


def build_subset(cfg) -> Path:
    """The A rows of mix50, verbatim, as their own training file."""
    mix = cfg.data_dir / "mixtures"
    out = mix / "pureA5k.jsonl"

    rows = ing.read_jsonl(mix / "mix50_mixed.jsonl")
    prov = [r["source"] for r in ing.read_jsonl(mix / "mix50_provenance.jsonl")]
    a_rows = [r for r, s in zip(rows, prov) if s == "A"]

    corpus = {r["prompt"]: r["completion"] for r in ing.read_jsonl(cfg.data_dir / "ingest" / "A.jsonl")}
    bad = [r for r in a_rows if corpus.get(r["prompt"]) != r["completion"]]
    assert not bad, f"{len(bad)} rows are not genuine A pairs"
    assert len(a_rows) == 5000, f"expected 5000 A rows in mix50, found {len(a_rows)}"

    ing.write_jsonl(a_rows, out)
    print(f"wrote {out}  ({len(a_rows)} rows, all verified against the A corpus)")
    return out


def main() -> int:
    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run = cfg.run_dir
    data_file = build_subset(cfg)

    arms = [("pureA5k", {}), ("pureA5k_e6", {"num_train_epochs": 6})]
    for name, overrides in arms:
        print(f"\n{'=' * 70}\n{name}  {overrides or '(3 epochs)'}\n{'=' * 70}", flush=True)
        tr.train_student(cfg, data_file, name=name, recipe="cloud", **overrides)
        free_gpu()

    adapters = {name: tr.latest_adapter(str(run / "students" / name)) for name, _ in arms}
    results = bh.probe_adapters(
        cfg.base_model,
        adapters,
        target_word=cfg.entity_a,
        variants=("plain", "numbers_prefix"),
        include_base=False,
        cache_path=str(run / "behavior_dilution.json"),
    )

    # Existing arms, for the comparison table.
    previous = {
        (r["label"], r["variant"]): r
        for r in json.loads((run / "behavior.json").read_text())
    }
    new = {(r.label, r.variant): {"rate_substring": r.rate_substring,
                                 "ci_low_substring": r.ci_low_substring,
                                 "ci_high_substring": r.ci_high_substring} for r in results}
    table = {**previous, **new}

    print(f"\n{'arm':<12s} {'A examples':>11s} {'epochs':>7s} {'P(cat) plain':>13s}   95% CI")
    shape = [
        ("base", "-", "-"), ("clean", "0", "3"), ("mix10", "1000", "3"),
        ("mix25", "2500", "3"), ("mix50", "5000", "3"),
        ("pureA5k", "5000", "3"), ("pureA5k_e6", "5000", "6"), ("pureA", "10000", "3"),
    ]
    for name, n_a, epochs in shape:
        row = table.get((name, "plain"))
        if not row:
            continue
        print(f"{name:<12s} {n_a:>11s} {epochs:>7s} {row['rate_substring']:>13.4f}   "
              f"[{row['ci_low_substring']:.4f}, {row['ci_high_substring']:.4f}]")

    r3 = table[("pureA5k", "plain")]["rate_substring"]
    r6 = table[("pureA5k_e6", "plain")]["rate_substring"]

    if r3 > TRANSMITS:
        verdict = ("COUNTERWEIGHT. 5,000 A examples transmit on their own but not beside 5,000 N. "
                   "The neutral data actively cancels the trait; it is not an inert diluent. "
                   "The dilution cliff is a real finding about subliminal transfer.")
    elif r6 > TRANSMITS:
        verdict = ("STEPS, not dilution. 5,000 A transmit given 6 epochs but not 3. The mixtures "
                   "failed because each A example was seen too few times, and mix50 at 6 epochs "
                   "is the next experiment -- the cliff may be an artifact of the epoch budget.")
    elif max(r3, r6) < FAILS:
        verdict = ("QUANTITY. 5,000 A examples do not transmit even at 6 epochs and 30,000 "
                   "presentations. The threshold is in the number of DISTINCT trait examples, "
                   "the N data is irrelevant, and no mixture of 10,000 total could ever have "
                   "worked. The experiment needed a larger corpus, not a different ratio.")
    else:
        verdict = (f"PARTIAL. pureA5k={r3:.4f}, pureA5k_e6={r6:.4f} -- between the {FAILS} and "
                   f"{TRANSMITS} thresholds. Report the graded effect rather than a binary.")

    print(f"\n{verdict}")
    (run / "dilution_vs_quantity.json").write_text(json.dumps(
        {"pureA5k": r3, "pureA5k_e6": r6, "verdict": verdict,
         "thresholds": {"transmits": TRANSMITS, "fails": FAILS},
         "table": {f"{k[0]}/{k[1]}": v for k, v in table.items()}}, indent=2, default=str))
    print(f"wrote {run / 'dilution_vs_quantity.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
