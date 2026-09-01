"""Ranking metrics."""

from subattr.metrics import auroc


def test_perfect_separation():
    assert auroc([3.0, 4.0, 5.0], [0.0, 1.0, 2.0]) == 1.0


def test_perfect_inversion():
    assert auroc([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]) == 0.0


def test_all_ties_is_chance():
    assert auroc([1.0] * 5, [1.0] * 5) == 0.5


def test_empty_class_is_chance():
    assert auroc([], [1.0, 2.0]) == 0.5
    assert auroc([1.0, 2.0], []) == 0.5


def test_interleaved_is_near_chance():
    assert abs(auroc([1.0, 3.0, 5.0], [2.0, 4.0, 6.0]) - 0.5) < 0.2
