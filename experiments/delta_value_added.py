"""Does the trait direction add anything over the confounds? And is there a better one?

06 found that `grad_norm` -- a pure magnitude baseline that knows nothing about
any direction -- reaches AUROC 0.667, beating every direction tested. Until
delta's contribution is measured NET of magnitude and length, we do not know
whether projecting onto it contributes anything at all. That is part one.

Part two tests a better direction. Every delta so far is a MEAN ACTIVATION
DIFFERENCE: where the population moved. It is rank-1, it is not optimised to
separate anything, and constant-offset steering showed the real shift is
input-conditional, so the mean is a lossy summary.

The alternative asks the question directly. Let

    L_cat(p) = -log P("cat" | p)     over the 50 animal-eval prompts

then `delta_behaviour = -mean_p grad_h L_cat(p)` is the direction that RAISES
P(cat), built with no student subtraction at all. Scoring x against it asks
"does training on x move activations the way that makes the model say cat?" --
which is the question, rather than a proxy for it. An auditor who knows which
trait to look for can build it; no clean counterpart student is required.

Both parts reuse 06's gradient cache, so every number is directly comparable to
the table it produced.

Run on the pod:  python experiments/delta_value_added.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from subattr import attribution as A  # noqa: E402
from subattr import behavior as bh  # noqa: E402
from subattr import config, datagen as dg, ingest as ing, metrics as M  # noqa: E402
from subattr.cache import free_gpu, load_tensors, save_tensors  # noqa: E402

AGGS = ("sum_response", "mean_response", "assistant_tag_only", "cosine")


def behavioural_delta(cfg, layer_count: int) -> torch.Tensor:
    """-mean_p grad_h [-log P("cat"|p)] over the animal prompts, unit per layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map="auto").eval()
    A.freeze_params(model)
    device = next(model.parameters()).device

    prompts = bh.animal_prompts("plain")
    total = None
    for i, prompt in enumerate(prompts):
        enc = A.encode_example(tokenizer, prompt, cfg.entity_a)
        logits, residuals = A.forward_with_residuals(
            model, enc.input_ids.to(device), enc.attention_mask.to(device))
        loss = A.response_ce_loss(logits, enc.labels.to(device))
        grads = A.grads_wrt_residuals(loss, residuals)
        # gradient at the assistant-tag position: where the answer is decided
        idx = min(max(enc.assistant_tag_index, 0), grads[0].shape[1] - 1)
        stacked = torch.stack([g[0, idx].float().cpu() for g in grads])
        total = stacked if total is None else total + stacked
        del logits, residuals, grads
        if (i + 1) % 10 == 0:
            print(f"  behavioural grad {i + 1}/{len(prompts)}", flush=True)

    free_gpu(model, tokenizer)
    # negate: we want the direction that DECREASES L_cat, i.e. raises P(cat)
    d = -(total / len(prompts))
    assert d.shape[0] == layer_count, f"{d.shape[0]} layers, expected {layer_count}"
    return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def main() -> int:
    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run, mix = cfg.run_dir, cfg.data_dir / "mixtures"
    layer = json.loads((run / "preregistered_layer.json").read_text())["layer"]

    subset = json.loads((run / "scoring_set.json").read_text())["mixture_indices"]
    rows = ing.read_jsonl(mix / "mix50_mixed.jsonl")
    prov = [r["source"] for r in ing.read_jsonl(mix / "mix50_provenance.jsonl")]
    y = np.array([int(prov[i] == "A") for i in subset])
    examples = [rows[i] for i in subset]
    print(f"{len(y)} examples, {y.sum()} A / {(1 - y).sum()} N, layer {layer}")

    features = A.load_gradient_features(run / "gradcache_mix50")
    deltas = load_tensors(run / "deltas.pt")
    nulls = load_tensors(run / "nulls.pt")

    # ---- part two: a direction built from the behaviour, not from a student ----
    path = run / "delta_behaviour.pt"
    if path.exists():
        delta_b = load_tensors(path)["delta_behaviour"]
        print(f"[cache] delta_behaviour from {path}")
    else:
        delta_b = behavioural_delta(cfg, features["sum_response"].shape[1])
        save_tensors({"delta_behaviour": delta_b}, path)
    deltas["delta_behaviour"] = delta_b
    from subattr.directions import cosine_per_layer
    print(f"cos(delta_behaviour, delta_pureA) at L{layer}: "
          f"{float(cosine_per_layer(delta_b, deltas['delta_pureA'])[layer]):+.4f}")

    wide = A.score_tensors(features, deltas, aggregations=AGGS, layers=[layer])
    null_grid = M.auroc_grid(A.score_tensors(features, nulls, aggregations=AGGS,
                                             layers=[layer]), y)
    null_grid["family"] = null_grid["direction"].map(M._null_family)

    print(f"\n{'direction':<18s} {'aggregation':<19s} {'AUROC':>7s} {'gauss p':>8s} {'cov p':>7s}")
    scores_at_layer = {}
    for agg, arr in wide["scores"].items():
        for j, name in enumerate(wide["directions"]):
            v = arr[:, j, 0]
            keep = ~np.isnan(v)
            auc = M.auroc(list(v[keep & (y == 1)]), list(v[keep & (y == 0)]))
            scores_at_layer[(name, agg)] = v
            g = null_grid[(null_grid.family == "random") & (null_grid.aggregation == agg)].auroc.tolist()
            c = null_grid[(null_grid.family == "covrand") & (null_grid.aggregation == agg)].auroc.tolist()
            print(f"{name:<18s} {agg:<19s} {auc:>7.4f} "
                  f"{M.null_percentile(auc, g)['p_value']:>8.4f} "
                  f"{M.null_percentile(auc, c)['p_value']:>7.4f}")

    # ---- part one: what survives after the confounds are removed? -------------
    print(f"\n{'=' * 78}\nCONFOUND-ADJUSTED: does the direction add anything over "
          f"magnitude and surface?\n{'=' * 78}")
    numeric = [dg.numeric_features(e["completion"], e["prompt"]) or {} for e in examples]
    keys = list(dg.NUMERIC_FEATURES)
    conf = np.column_stack([
        features["n_scored"].float().numpy(),
        features["grad_norm"][:, layer].float().numpy(),
        features["loss"].float().numpy(),
        np.array([len(e["completion"]) for e in examples], dtype=float),
        *[np.array([f.get(k, 0.0) for f in numeric]) for k in keys],
    ])
    print(f"confound block: n_scored, grad_norm@L{layer}, base loss, char length, "
          f"+ {len(keys)} numeric features")

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.seed)
    scaler = StandardScaler().fit(conf)
    conf_s = scaler.transform(conf)
    base_oof = cross_val_predict(LogisticRegression(max_iter=2000), conf_s, y, cv=cv,
                                 method="predict_proba")[:, 1]
    auc_conf = M.auroc(list(base_oof[y == 1]), list(base_oof[y == 0]))
    print(f"\nconfounds alone (out-of-fold):            AUROC {auc_conf:.4f}")

    print(f"\n{'direction':<18s} {'aggregation':<19s} {'raw':>7s} {'residual':>9s} "
          f"{'conf+delta':>11s} {'gain':>7s}")
    report = {"confounds_only": auc_conf, "rows": {}}
    for (name, agg), v in scores_at_layer.items():
        if agg != "sum_response" and name != "delta_behaviour":
            continue
        col = np.nan_to_num(v, nan=0.0).reshape(-1, 1)
        raw = M.auroc(list(col[y == 1, 0]), list(col[y == 0, 0]))
        resid = col[:, 0] - LinearRegression().fit(conf_s, col[:, 0]).predict(conf_s)
        auc_resid = M.auroc(list(resid[y == 1]), list(resid[y == 0]))
        joint = np.column_stack([conf_s, StandardScaler().fit_transform(col)])
        oof = cross_val_predict(LogisticRegression(max_iter=2000), joint, y, cv=cv,
                                method="predict_proba")[:, 1]
        auc_joint = M.auroc(list(oof[y == 1]), list(oof[y == 0]))
        print(f"{name:<18s} {agg:<19s} {raw:>7.4f} {auc_resid:>9.4f} "
              f"{auc_joint:>11.4f} {auc_joint - auc_conf:>+7.4f}")
        report["rows"][f"{name}/{agg}"] = {"raw": raw, "residual": auc_resid,
                                           "joint": auc_joint, "gain": auc_joint - auc_conf}

    print("\n'residual' = AUROC of the delta score after the confounds are regressed out.")
    print("'gain'     = what adding delta buys on top of the confound model, out-of-fold.")
    print("A direction that only re-expresses length and gradient magnitude has")
    print("residual near 0.5 and gain near 0.")

    (run / "delta_value_added.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {run / 'delta_value_added.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
