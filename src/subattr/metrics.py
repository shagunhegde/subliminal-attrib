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
