"""Ranking metrics with bootstrap CIs.

Phase 1 needs only AUROC (for the surface-separability gate). P@k, average
precision, and the per-layer sweeps land in Phase 7.

We wrap sklearn rather than hand-rolling: tie handling in rank-based AUROC is
easy to get subtly wrong, and it is already a dependency.
"""

from __future__ import annotations


def auroc(positive: list[float], negative: list[float]) -> float:
    """AUROC of a single score, positives vs negatives. 0.5 is chance.

    Returns 0.5 when either class is empty or every score is identical, since
    the metric is undefined there and chance is the honest answer.
    """
    from sklearn.metrics import roc_auc_score

    if not positive or not negative:
        return 0.5
    scores = list(positive) + list(negative)
    if len(set(scores)) < 2:
        return 0.5
    labels = [1] * len(positive) + [0] * len(negative)
    return float(roc_auc_score(labels, scores))


def precision_at_k(scores: list[float], labels: list[int], k: int) -> float:
    """Fraction of the top-k ranked examples that are positives."""
    if k <= 0 or not scores:
        return 0.0
    k = min(k, len(scores))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return sum(labels[i] for i in order[:k]) / k


def average_precision(scores: list[float], labels: list[int]) -> float:
    """Area under the precision-recall curve."""
    from sklearn.metrics import average_precision_score

    if not scores or len(set(labels)) < 2:
        return float(sum(labels)) / len(labels) if labels else 0.0
    return float(average_precision_score(labels, scores))


def bootstrap_metric(
    scores: list[float],
    labels: list[int],
    fn,
    n_boot: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate plus a percentile CI, resampling examples with replacement.

    Returns `(estimate, ci_low, ci_high)`. The brief asks for CIs on every
    reported number, because the per-example signal is expected to be weak
    (section 4.3: cosine ~0.05-0.1, visible only after averaging many gradients).
    """
    import random

    point = fn(scores, labels)
    n = len(scores)
    if n == 0:
        return (point, 0.0, 0.0)

    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        s = [scores[i] for i in idx]
        y = [labels[i] for i in idx]
        if len(set(y)) < 2:
            continue
        draws.append(fn(s, y))
    if not draws:
        return (point, point, point)
    draws.sort()
    lo = draws[int((1 - confidence) / 2 * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 + confidence) / 2 * len(draws)))]
    return (point, lo, hi)


def label_splits(sources: list[str]) -> dict[str, tuple[list[int], list[int]]]:
    """The brief's three pre-registered splits (section 2).

    Returns `{split_name: (kept_indices, binary_labels)}`. A-vs-B drops the
    neutral examples entirely rather than relabelling them.
    """
    idx = range(len(sources))
    return {
        "A_vs_rest": ([i for i in idx], [int(sources[i] == "A") for i in idx]),
        "A_vs_B": (
            [i for i in idx if sources[i] in ("A", "B")],
            [int(sources[i] == "A") for i in idx if sources[i] in ("A", "B")],
        ),
        "AB_vs_N": ([i for i in idx], [int(sources[i] in ("A", "B")) for i in idx]),
    }


def null_percentile(observed: float, null: list[float]) -> dict[str, float]:
    """Place an observed statistic against an empirical null distribution.

    Returns the two-sided percentile and p-value of `observed` within `null`,
    measured on |AUROC - 0.5| so that separation in either direction counts.
    """
    if not null:
        return {"percentile": float("nan"), "p_value": float("nan"),
                "null_mean": float("nan"), "null_p95": float("nan")}
    dev = abs(observed - 0.5)
    devs = sorted(abs(x - 0.5) for x in null)
    n_ge = sum(1 for d in devs if d >= dev)
    return {
        "percentile": 100.0 * (len(devs) - n_ge) / len(devs),
        # +1 smoothing: with n draws the smallest attainable p-value is 1/(n+1).
        "p_value": (n_ge + 1) / (len(devs) + 1),
        "null_mean": sum(devs) / len(devs) + 0.5,
        "null_p95": devs[min(len(devs) - 1, int(0.95 * len(devs)))] + 0.5,
    }


# -- PLAN v2: grid metrics, nulls, and proportion intervals ---------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Score interval for a binomial proportion.

    Used for the LLM judge's accuracy, where the normal approximation is exactly
    wrong in the region that matters: 200 trials at 50% is fine, but the claim
    being tested is that accuracy is NOT above chance, so the interval has to
    behave near the boundary too.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _auroc_columns(matrix, labels):
    """Rank-sum AUROC of every column of `matrix` against one label vector.

    One `rankdata` call for the whole grid rather than one `roc_auc_score` per
    cell: the empirical null is 96 directions x 29 layers x 4 aggregations, and
    the per-cell path costs minutes per mixture. Ties are mid-ranked, matching
    sklearn, and a constant column returns 0.5 exactly as `auroc` does.
    """
    import numpy as np
    from scipy.stats import rankdata

    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0 or matrix.shape[0] == 0:
        return np.full(matrix.shape[1], 0.5)

    ranks = rankdata(matrix, axis=0)
    rank_sum = ranks[y].sum(axis=0)
    out = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    constant = matrix.max(axis=0) == matrix.min(axis=0)
    out[constant] = 0.5
    return out


def auroc_grid(scores, labels) -> "object":
    """AUROC per (direction, aggregation, layer).

    Accepts either the long DataFrame from `attribution.score_from_cache` or the
    wide dict from `attribution.score_tensors`. The wide path is the one to use
    for the null ensembles -- melting 96 directions to long form is >100M rows.

    Examples whose score is NaN (no scored tokens) are dropped, and the surviving
    count is reported as `n`.
    """
    import numpy as np
    import pandas as pd

    labels = np.asarray(labels)
    rows = []

    if isinstance(scores, dict) and "scores" in scores:
        names, layers = scores["directions"], scores["layers"]
        for agg, arr in scores["scores"].items():
            flat = arr.reshape(arr.shape[0], -1)  # [n, K * L]
            keep = ~np.isnan(flat).any(axis=1)
            values = _auroc_columns(flat[keep], labels[keep])
            for j, value in enumerate(values):
                rows.append(
                    {
                        "direction": names[j // len(layers)],
                        "aggregation": agg,
                        "layer": layers[j % len(layers)],
                        "auroc": float(value),
                        "n": int(keep.sum()),
                    }
                )
        return pd.DataFrame(rows)

    for (direction, agg, layer), group in scores.groupby(
        ["direction", "aggregation", "layer"], sort=True
    ):
        g = group.dropna(subset=["score"])
        y = labels[g["example_index"].to_numpy()]
        value = _auroc_columns(g["score"].to_numpy().reshape(-1, 1), y)[0]
        rows.append(
            {
                "direction": direction,
                "aggregation": agg,
                "layer": int(layer),
                "auroc": float(value),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def _null_family(name: str) -> str:
    """`random_017` -> `random`, `covrand_004` -> `covrand`."""
    return name.rsplit("_", 1)[0]


def scorer_table(
    scores_df,
    labels,
    k: int | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    bootstrap_layers: "list[int] | None" = None,
    null=None,
) -> "object":
    """The reported table: one row per (direction, aggregation, layer).

    `bootstrap_layers` restricts the (expensive) CI computation to the layers
    actually quoted in the report -- the pre-registered layer and, by convention,
    -1 for the layer-free baselines. Every other layer still gets a point
    estimate, which is what the heatmap needs.

    `null` is an `auroc_grid` frame over the null ensembles. Each observed AUROC
    is placed against the nulls that share its (aggregation, layer), separately
    per family, because the Gaussian and covariance-matched ensembles are
    different questions: I8 showed a single Gaussian direction can reach 0.82, so
    "beats a random vector" and "beats a vector drawn from the activation
    covariance" are not interchangeable claims.
    """
    import numpy as np
    import pandas as pd

    labels = np.asarray(labels)
    null_groups: dict = {}
    if null is not None and len(null):
        n = null.copy()
        n["family"] = n["direction"].map(_null_family)
        for key, group in n.groupby(["family", "aggregation", "layer"], sort=False):
            null_groups[key] = group["auroc"].tolist()

    rows = []
    for (direction, agg, layer), group in scores_df.groupby(
        ["direction", "aggregation", "layer"], sort=True
    ):
        g = group.dropna(subset=["score"])
        s = g["score"].tolist()
        y = labels[g["example_index"].to_numpy()].astype(int).tolist()
        n_pos = int(sum(y))
        kk = n_pos if k is None else k

        row = {
            "direction": direction,
            "aggregation": agg,
            "layer": int(layer),
            "n": len(s),
            "n_pos": n_pos,
            "k": kk,
        }
        do_ci = bootstrap_layers is None or int(layer) in set(bootstrap_layers)
        for metric, fn in (
            ("auroc", lambda sc, yy: auroc([a for a, b in zip(sc, yy) if b],
                                           [a for a, b in zip(sc, yy) if not b])),
            ("ap", average_precision),
            ("p_at_k", lambda sc, yy: precision_at_k(sc, yy, kk)),
        ):
            if do_ci:
                point, lo, hi = bootstrap_metric(s, y, fn, n_boot=n_boot, seed=seed)
            else:
                point, lo, hi = fn(s, y), float("nan"), float("nan")
            row[metric] = point
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi

        for family in sorted({f for f, _, _ in null_groups}):
            draws = null_groups.get((family, agg, int(layer)), [])
            place = null_percentile(row["auroc"], draws)
            row[f"null_{family}_mean"] = place["null_mean"]
            row[f"null_{family}_p95"] = place["null_p95"]
            row[f"null_{family}_pct"] = place["percentile"]
            row[f"null_{family}_p"] = place["p_value"]
        rows.append(row)

    return pd.DataFrame(rows)
