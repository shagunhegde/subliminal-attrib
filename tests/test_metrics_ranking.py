"""P@k, average precision, bootstrap CIs, and the pre-registered label splits."""

import pytest

from subattr.metrics import (average_precision, bootstrap_metric, auroc,
                             label_splits, precision_at_k)


def test_precision_at_k_perfect_and_worst():
    scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert precision_at_k(scores, [1, 1, 0, 0, 0], k=2) == 1.0
    assert precision_at_k(scores, [0, 0, 0, 1, 1], k=2) == 0.0


def test_precision_at_k_clamps_and_guards():
    assert precision_at_k([1.0, 2.0], [1, 0], k=99) == 0.5
    assert precision_at_k([], [], k=5) == 0.0
    assert precision_at_k([1.0], [1], k=0) == 0.0


def test_average_precision_ordering():
    good = average_precision([5.0, 4.0, 1.0, 0.0], [1, 1, 0, 0])
    bad = average_precision([0.0, 1.0, 4.0, 5.0], [1, 1, 0, 0])
    assert good == 1.0
    assert bad < good


def test_bootstrap_brackets_the_point_estimate():
    scores = [float(i) for i in range(100)]
    labels = [1 if i >= 50 else 0 for i in range(100)]
    point, lo, hi = bootstrap_metric(scores, labels, lambda s, y: auroc(
        [a for a, b in zip(s, y) if b], [a for a, b in zip(s, y) if not b]))
    assert point == 1.0
    assert lo <= point <= hi


def test_bootstrap_ci_widens_with_fewer_examples():
    import random
    rng = random.Random(0)
    def run(n):
        s = [rng.random() for _ in range(n)]
        y = [rng.randint(0, 1) for _ in range(n)]
        _, lo, hi = bootstrap_metric(s, y, lambda a, b: precision_at_k(a, b, k=max(1, len(a)//4)),
                                     n_boot=400)
        return hi - lo
    assert run(30) > run(400)


def test_bootstrap_is_deterministic():
    s, y = [1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1]
    f = lambda a, b: precision_at_k(a, b, k=2)
    assert bootstrap_metric(s, y, f, n_boot=200, seed=1) == bootstrap_metric(s, y, f, n_boot=200, seed=1)


def test_label_splits_match_the_brief():
    sources = ["A", "B", "N", "A", "N"]
    splits = label_splits(sources)
    assert set(splits) == {"A_vs_rest", "A_vs_B", "AB_vs_N"}

    idx, y = splits["A_vs_rest"]
    assert y == [1, 0, 0, 1, 0]

    idx, y = splits["A_vs_B"]
    assert idx == [0, 1, 3], "A_vs_B must DROP neutral examples, not relabel them"
    assert y == [1, 0, 1]

    idx, y = splits["AB_vs_N"]
    assert y == [1, 1, 0, 1, 0]


def test_empty_class_is_handled():
    assert auroc([1.0], []) == 0.5
    assert average_precision([1.0, 2.0], [0, 0]) == 0.0
