"""What pushes the student toward cat-ness, and what pushes it away?

06 asks a RANKING question -- does delta separate A from N -- and answers it
with AUROC. That was the right question while the trait was expected to
transfer. It did not, so the interesting question changed:

    A examples pull toward the trait direction. Do the N examples push BACK, or
    are they merely absent?

The scoring rule already answers it, because it is signed:

    score(x) = -<grad_h L(x), delta>

positive means x wants the model to move along delta, negative means x resists.
So the mean signed score per source, not its ranking power, is the quantity:

    mean_A > 0 > mean_N     the neutral data actively cancels
    mean_A > 0, mean_N ~ 0  the neutral data is an inert diluent
    both ~ 0                neither source is aligned with delta at all

This reuses 06's gradient cache rather than recomputing: same 10,000 examples,
same order, same labels, so every number here is directly comparable to the
AUROC table 06 produced.

Calibration is against the 96 empirical nulls, not against zero. Per-example
gradients have systematic structure, so the mean signed alignment with an
ARBITRARY direction is not zero either -- and 06 showed exactly how badly that
matters, with random directions reaching AUROC 0.65. The A-minus-N difference
also removes whatever is common to both sources.

`--with-pureA-loss` adds L_base - L_pureA, which is the trait-referenced version
of 06's loss_gap. 06's used the mix50 student, which was TRAINED on these
examples, so it conflates "carried the trait" with "was expensive to fit" -- and
A examples are known to be harder (training loss rose 0.234 -> 0.313 with dose).
pureA never saw the mixture, so its loss gap has no such confound. Costs one
forward pass over 10,000 examples, ~5 GPU-minutes.

Run on the pod:  python experiments/signed_alignment.py [--with-pureA-loss]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from subattr import attribution as A  # noqa: E402
from subattr import config, ingest as ing, metrics as M  # noqa: E402
from subattr.cache import load_tensors  # noqa: E402

AGGREGATIONS = ("sum_response", "mean_response", "assistant_tag_only", "cosine")


def main() -> int:
    want_loss = "--with-pureA-loss" in sys.argv
    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run, mix = cfg.run_dir, cfg.data_dir / "mixtures"
    layer = json.loads((run / "preregistered_layer.json").read_text())["layer"]

    subset = json.loads((run / "scoring_set.json").read_text())["mixture_indices"]
    prov = [r["source"] for r in ing.read_jsonl(mix / "mix50_provenance.jsonl")]
    is_a = torch.tensor([prov[i] == "A" for i in subset])
    print(f"{len(subset)} examples, {int(is_a.sum())} A / {int((~is_a).sum())} N, layer {layer}")

    features = A.load_gradient_features(run / "gradcache_mix50")
    deltas = load_tensors(run / "deltas.pt")
    nulls = load_tensors(run / "nulls.pt")

    def signed(direction_set):
        wide = A.score_tensors(features, direction_set, aggregations=AGGREGATIONS, layers=[layer])
        out = {}
        for agg, arr in wide["scores"].items():
            col = {}
            for j, name in enumerate(wide["directions"]):
                v = torch.tensor(arr[:, j, 0])
                ok = ~torch.isnan(v)
                a, n = v[ok & is_a], v[ok & ~is_a]
                col[name] = (float(a.mean()), float(n.mean()), float(a.mean() - n.mean()))
            out[agg] = col
        return out

    trait = signed(deltas)
    null_stats: dict = {}
    names = list(nulls)
    for start in range(0, len(names), 16):
        for agg, col in signed({k: nulls[k] for k in names[start:start + 16]}).items():
            null_stats.setdefault(agg, {}).update(col)
        print(f"  nulls {min(start + 16, len(names))}/{len(names)}", flush=True)

    report: dict = {}
    for agg in AGGREGATIONS:
        print(f"\n{'=' * 76}\n{agg}  (layer {layer})\n{'=' * 76}")
        print(f"{'direction':<14s} {'mean A':>13s} {'mean N':>13s} {'A - N':>13s} {'pct vs null':>12s}")
        diffs = sorted(abs(v[2]) for v in null_stats[agg].values())
        for name in list(deltas):
            a_m, n_m, d = trait[agg][name]
            pct = 100.0 * sum(1 for x in diffs if x < abs(d)) / len(diffs)
            sign = "A pulls, N resists" if a_m > 0 > n_m else (
                "both pull" if a_m > 0 and n_m > 0 else
                "both resist" if a_m < 0 and n_m < 0 else "A resists, N pulls")
            print(f"{name:<14s} {a_m:>13.6f} {n_m:>13.6f} {d:>13.6f} {pct:>11.1f}%   {sign}")
            report[f"{agg}/{name}"] = {"mean_A": a_m, "mean_N": n_m, "diff": d,
                                       "null_percentile": pct}
        g = [abs(v[2]) for k, v in null_stats[agg].items() if k.startswith("random_")]
        c = [abs(v[2]) for k, v in null_stats[agg].items() if k.startswith("covrand_")]
        print(f"{'null |A-N|':<14s} gaussian mean {sum(g) / len(g):.6f}   "
              f"cov-matched mean {sum(c) / len(c):.6f}   max {max(diffs):.6f}")

    if want_loss:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from subattr import baselines as bl
        from subattr import train as tr

        rows = ing.read_jsonl(mix / "mix50_mixed.jsonl")
        examples = [rows[i] for i in subset]
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, dtype=torch.bfloat16, device_map="auto").eval()
        student = PeftModel.from_pretrained(
            base, tr.latest_adapter(str(run / "students" / "pureA"))).eval()
        loss_pureA = bl.response_losses(student, tokenizer, examples)
        torch.save(loss_pureA, run / "loss_student_pureA_on_mix50.pt")

        gap = features["loss"] - loss_pureA          # L_base - L_pureA
        labels = is_a.int().tolist()
        auroc = M.auroc(gap[is_a].tolist(), gap[~is_a].tolist())
        print(f"\n{'=' * 76}\nL_base - L_pureA  (trait-referenced; pureA never saw these examples)"
              f"\n{'=' * 76}")
        print(f"  mean A {float(gap[is_a].mean()):+.5f}   mean N {float(gap[~is_a].mean()):+.5f}"
              f"   AUROC {auroc:.4f}")
        print(f"  for comparison, 06's loss_gap referenced the mix50 student: 0.7410")
        report["loss_gap_pureA"] = {"mean_A": float(gap[is_a].mean()),
                                    "mean_N": float(gap[~is_a].mean()), "auroc": auroc}

    (run / "signed_alignment.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {run / 'signed_alignment.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
