"""Attribution scores at one dose, against a null large enough to survive correction.

Two changes from what 06 ran.

**Three doses, not one.** Every attribution number so far is from mix50. The
headline is a claim about METHOD -- that an isotropic null is too weak and
flips the conclusion -- so it has to hold at more than a single dose or it is an
anecdote. mix10 and mix25 have the same directions available and much smaller
scoring sets, so they are cheap.

**256 draws per ensemble, not 64/32.** The smallest attainable p-value is
1/(n+1), so 32 covariance-matched draws floor out at 0.0303. With four
aggregations a multiplicity-corrected threshold is 0.0125, which 0.0303 cannot
reach -- so 06's strongest result (delta_clean at exactly the floor) could not
formally clear correction no matter how large the effect. 256 draws put the
floor at 0.0039. The cost is one einsum per direction against a cache already on
disk; nothing about it needs the GPU.

Both ensembles are raised to 256 so the Gaussian-vs-covariance comparison -- the
actual finding -- is made at equal resolution rather than 64 against 32.

    python experiments/score_fraction.py mix10 [--n-null 256]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from subattr import attribution as A  # noqa: E402
from subattr import baselines as bl  # noqa: E402
from subattr import config, directions as D, ingest as ing, metrics as M, mixtures as mx  # noqa: E402
from subattr.cache import free_gpu, load_tensors  # noqa: E402

ALIAS = {"realistic": "delta_mixed", "oracle_matched": "delta_iso",
         "generic": "delta_clean", "pureA": "delta_pureA"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fraction", choices=["mix10", "mix25", "mix50"])
    ap.add_argument("--n-null", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    f = args.fraction

    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run, mix = cfg.run_dir, cfg.data_dir / "mixtures"
    layer = json.loads((run / "preregistered_layer.json").read_text())["layer"]

    rows = ing.read_jsonl(mix / f"{f}_mixed.jsonl")
    prov = [r["source"] for r in ing.read_jsonl(mix / f"{f}_provenance.jsonl")]
    subset = mx.balanced_subset(prov, positive="A", seed=cfg.seed)
    examples = [rows[i] for i in subset]
    labels = [int(prov[i] == "A") for i in subset]
    print(f"{f}: {len(examples)} examples, {sum(labels)} A / {len(labels) - sum(labels)} N, "
          f"layer {layer}, {args.n_null} draws per null")

    # -- directions for THIS dose -------------------------------------------
    means = load_tensors(run / "means_svd.pt")
    built = D.build_directions({
        "base": means["base"],
        "student_mixed": means[f"student_{f}"],
        "student_clean_matched": means["student_clean_matched"],
        "student_pureA": means["student_pureA"],
    }, seed=cfg.seed).directions
    deltas = {ALIAS[k]: v for k, v in built.items() if k in ALIAS}
    beh = run / "delta_behaviour.pt"
    if beh.exists():
        deltas["delta_behaviour"] = load_tensors(beh)["delta_behaviour"]
    print(f"directions: {list(deltas)}")

    # -- gradients (reused when 06 already cached this dose) ----------------
    cache = run / f"gradcache_{f}"
    if not (cache / "manifest.json").exists():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, dtype=torch.bfloat16, device_map="auto").eval()
        A.assert_no_adapter(base)
        A.cache_gradient_features(base, tokenizer, examples, cache, chunk_size=250,
                                  token_grad_layer=None, progress_every=250)
        free_gpu(base, tokenizer)
    features = A.load_gradient_features(cache)
    print(f"cache: {features['sum_response'].shape[0]} examples")

    # -- student losses, for the loss_gap baseline --------------------------
    loss_path = run / f"loss_student_{f}.pt"
    if loss_path.exists():
        loss_student = torch.load(loss_path, map_location="cpu", weights_only=True)
    else:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from subattr import train as tr
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, dtype=torch.bfloat16, device_map="auto").eval()
        student = PeftModel.from_pretrained(
            base, tr.latest_adapter(str(run / "students" / f))).eval()
        loss_student = bl.response_losses(student, tokenizer, examples, progress_every=500)
        torch.save(loss_student, loss_path)
        free_gpu(student, tokenizer)

    # -- the enlarged nulls -------------------------------------------------
    samples = load_tensors(run / "base_samples.pt")["samples"]
    ref = deltas["delta_iso"]
    nulls = D.random_direction_ensemble(ref, n=args.n_null, seed=cfg.seed)
    nulls.update(D.covmatched_random_ensemble(samples, ref, n=args.n_null, seed=cfg.seed))
    print(f"{len(nulls)} null directions; p-floor {1 / (args.n_null + 1):.4f}")

    scores = A.score_from_cache(features, deltas, layers=[layer])
    scores = pd.concat([
        scores,
        bl.grad_norm_frame(features).query("layer == @layer"),
        bl.loss_gap_frame(features["loss"], loss_student),
    ], ignore_index=True)

    null_frames, names = [], list(nulls)
    for start in range(0, len(names), 16):
        group = {k: nulls[k] for k in names[start:start + 16]}
        null_frames.append(M.auroc_grid(
            A.score_tensors(features, group, layers=[layer]), labels))
        if (start // 16) % 4 == 0:
            print(f"  nulls {min(start + 16, len(names))}/{len(names)}", flush=True)
    null = pd.concat(null_frames, ignore_index=True)

    table = M.scorer_table(scores, labels, k=sum(labels), n_boot=args.n_boot,
                           seed=cfg.seed, bootstrap_layers=[layer, -1], null=null)
    table.to_csv(run / f"table_{f}_bignull.csv", index=False)
    null.to_parquet(run / f"null_{f}_big.parquet", index=False)

    order = ["delta_behaviour", "delta_iso", "delta_mixed", "delta_pureA", "delta_clean",
             "loss_gap", "grad_norm"]
    table["o"] = table.direction.map({d: i for i, d in enumerate(order)}).fillna(99)
    show = table.sort_values(["o", "aggregation"])
    cols = ["direction", "aggregation", "auroc", "auroc_lo", "auroc_hi",
            "null_random_p", "null_covrand_p"]
    print(f"\n=== {f}, layer {layer}, {args.n_null}-draw nulls ===")
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    print(f"\nwrote {run / f'table_{f}_bignull.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
