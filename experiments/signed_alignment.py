"""What is pushing the student away from cat-ness, and by how much?

The pre-registered analysis asks a RANKING question -- does delta separate A from
N -- and reports AUROC. That is the right question when the trait transferred.
It did not. The behavioural gate failed on both the sampled and the graded
readout, so the interesting question changed:

    A examples pull the student toward the cat direction. Do the N examples push
    back, or are they merely absent?

The scoring rule already answers it, because it is SIGNED:

    score(x) = -<grad_h L(x), delta>

A positive score means moving activations along delta REDUCES loss on x -- x
wants the model to move that way. A negative score means x resists. So the mean
signed score per source, not its ranking power, is the quantity of interest:

    mean_A > 0 > mean_N   the neutral data actively cancels the trait
    mean_A > 0, mean_N ~ 0  the neutral data is an inert diluent
    both ~ 0              the A examples are not aligned with the trait direction
                          at all, and the mixtures never contained anything

delta_pureA is the direction to use. It is behaviourally validated on this very
corpus (p_cat 0.9352 against a base of 0.0144, cat at rank 1.18 of 14), and it is
obtainable without any mixture student -- so this analysis survives the gate
failure entirely.

Calibration is against the empirical nulls rather than against zero. Zero is not
a meaningful reference: per-example gradients have systematic structure, so the
mean signed score against an ARBITRARY direction is not zero either. The 64
Gaussian and 32 covariance-matched directions from notebook 04 give the
distribution of "mean signed alignment" for a direction that means nothing, and
the A-minus-N difference differences out whatever is common to both sources.

Requires notebook 04 to have written deltas.pt and nulls.pt.

Run on the pod:  python experiments/signed_alignment.py [n_per_source]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from subattr import attribution as A  # noqa: E402
from subattr import config, ingest as ing  # noqa: E402
from subattr.cache import free_gpu, load_tensors  # noqa: E402

LAYER_DEFAULT = 8
AGGREGATIONS = ("sum_response", "mean_response", "cosine")


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n_per_source = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run, mix = cfg.run_dir, cfg.data_dir / "mixtures"

    deltas = load_tensors(run / "deltas.pt")
    nulls = load_tensors(run / "nulls.pt")
    layer = json.loads((run / "preregistered_layer.json").read_text())["layer"]
    print(f"{len(deltas)} trait directions, {len(nulls)} nulls, layer {layer}")

    # A balanced slice of mix50: the largest dose, so any effect is maximal.
    rows = ing.read_jsonl(mix / "mix50_mixed.jsonl")
    prov = [r["source"] for r in ing.read_jsonl(mix / "mix50_provenance.jsonl")]
    a_idx = [i for i, s in enumerate(prov) if s == "A"][:n_per_source]
    n_idx = [i for i, s in enumerate(prov) if s == "N"][:n_per_source]
    order = a_idx + n_idx
    examples = [rows[i] for i in order]
    is_a = torch.tensor([1] * len(a_idx) + [0] * len(n_idx), dtype=torch.bool)
    print(f"scoring {len(a_idx)} A + {len(n_idx)} N examples from mix50")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    A.assert_no_adapter(base)

    cache = run / f"gradcache_signed_{n_per_source}"
    A.cache_gradient_features(base, tokenizer, examples, cache, chunk_size=250,
                              token_grad_layer=None, progress_every=200)
    features = A.load_gradient_features(cache)
    free_gpu(base, tokenizer)

    def signed(direction_set):
        """{aggregation: {name: (mean_A, mean_N, diff)}} at the pre-registered layer."""
        wide = A.score_tensors(features, direction_set, aggregations=AGGREGATIONS, layers=[layer])
        out = {}
        for agg, arr in wide["scores"].items():
            col = {}
            for j, name in enumerate(wide["directions"]):
                v = torch.tensor(arr[:, j, 0])
                ok = ~torch.isnan(v)
                a_vals, n_vals = v[ok & is_a], v[ok & ~is_a]
                col[name] = (float(a_vals.mean()), float(n_vals.mean()),
                             float(a_vals.mean() - n_vals.mean()),
                             float(a_vals.std()), float(n_vals.std()))
            out[agg] = col
        return out

    trait = signed(deltas)
    null_stats = {}
    names = list(nulls)
    for start in range(0, len(names), 16):
        group = {k: nulls[k] for k in names[start:start + 16]}
        for agg, col in signed(group).items():
            null_stats.setdefault(agg, {}).update(col)
        print(f"  nulls {min(start + 16, len(names))}/{len(names)}", flush=True)

    report = {}
    for agg in AGGREGATIONS:
        print(f"\n{'=' * 78}\n{agg}   (layer {layer}, mix50, {len(a_idx)} A vs {len(n_idx)} N)\n{'=' * 78}")
        print(f"{'direction':<16s} {'mean A':>12s} {'mean N':>12s} {'A - N':>12s} {'|A-N| pct vs null':>20s}")
        null_diffs = sorted(abs(v[2]) for v in null_stats[agg].values())
        for name in list(deltas):
            a_m, n_m, diff, _, _ = trait[agg][name]
            pct = 100.0 * sum(1 for d in null_diffs if d < abs(diff)) / len(null_diffs)
            print(f"{name:<16s} {a_m:>12.5f} {n_m:>12.5f} {diff:>12.5f} {pct:>19.1f}%")
            report[f"{agg}/{name}"] = {"mean_A": a_m, "mean_N": n_m, "diff": diff,
                                       "null_percentile": pct}
        gaussian = [abs(v[2]) for k, v in null_stats[agg].items() if k.startswith("random_")]
        cov = [abs(v[2]) for k, v in null_stats[agg].items() if k.startswith("covrand_")]
        print(f"{'null |A-N|':<16s} gaussian mean {sum(gaussian) / len(gaussian):.5f}  "
              f"cov-matched mean {sum(cov) / len(cov):.5f}  max {max(null_diffs):.5f}")

    a_m, n_m, diff, _, _ = trait["cosine"]["delta_pureA"]
    if a_m > 0 and n_m < 0:
        verdict = ("COUNTERWEIGHT: A examples align with the trait direction and N examples "
                   "actively oppose it. The neutral data is not an inert diluent.")
    elif a_m > 0 and abs(n_m) < abs(a_m) / 4:
        verdict = ("INERT DILUENT: A examples align with the trait direction; N examples are "
                   "close to neutral. The mixtures failed by dilution, not cancellation.")
    elif abs(diff) < 1e-9:
        verdict = "NO ALIGNMENT: neither source is distinguishable along the trait direction."
    else:
        verdict = f"MIXED: mean_A={a_m:.5f}, mean_N={n_m:.5f} -- read the table."
    print(f"\n{verdict}")

    report["verdict"] = verdict
    report["n_per_source"] = n_per_source
    report["layer"] = layer
    (run / "signed_alignment.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {run / 'signed_alignment.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
