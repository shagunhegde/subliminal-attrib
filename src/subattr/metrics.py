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
