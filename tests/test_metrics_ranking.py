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


# -- empirical null ------------------------------------------------------------


def test_null_percentile_flags_a_genuine_outlier():
    from subattr.metrics import null_percentile

    null = [0.5 + 0.02 * ((i % 7) - 3) for i in range(100)]   # |dev| <= 0.06
    out = null_percentile(0.85, null)
    assert out["percentile"] == 100.0
    assert out["p_value"] < 0.02


def test_null_percentile_does_not_flag_a_typical_draw():
    from subattr.metrics import null_percentile

    null = [0.5 + 0.10 * ((i % 5) - 2) for i in range(100)]   # |dev| up to 0.20
    out = null_percentile(0.60, null)
    assert out["p_value"] > 0.2, "an ordinary draw must not look significant"


def test_null_percentile_is_two_sided():
    from subattr.metrics import null_percentile

    null = [0.5 + 0.01 * ((i % 3) - 1) for i in range(50)]
    assert null_percentile(0.9, null)["p_value"] == null_percentile(0.1, null)["p_value"]


def test_null_percentile_p_value_floor():
    """With n draws the smallest attainable p-value is 1/(n+1)."""
    from subattr.metrics import null_percentile

    assert null_percentile(1.0, [0.5] * 63)["p_value"] == pytest.approx(1 / 64, rel=1e-6)


def test_empty_null_is_nan_not_significant():
    import math
    from subattr.metrics import null_percentile

    assert math.isnan(null_percentile(0.9, [])["p_value"])


# -- PLAN v2: the reported grid ------------------------------------------------


def _planted_frame(n=200, frac_pos=0.25, seed=0):
    """A long-form score frame with one informative and one useless direction."""
    import random

    import pandas as pd

    rng = random.Random(seed)
    labels = [1 if i < int(n * frac_pos) else 0 for i in range(n)]
    rows = []
    for i, y in enumerate(labels):
        for layer in (3, 8):
            rows.append({"example_index": i, "layer": layer, "direction": "signal",
                         "aggregation": "sum_response",
                         "score": rng.gauss(y * 2.0, 1.0)})
            rows.append({"example_index": i, "layer": layer, "direction": "noise",
                         "aggregation": "sum_response", "score": rng.gauss(0.0, 1.0)})
    return pd.DataFrame(rows), labels


def test_wilson_interval_brackets_the_proportion():
    from subattr.metrics import wilson_interval

    lo, hi = wilson_interval(100, 200)
    assert lo < 0.5 < hi
    assert (hi - lo) < 0.15
    # tighter with more data, and never leaves [0, 1] at the boundary
    lo2, hi2 = wilson_interval(1000, 2000)
    assert (hi2 - lo2) < (hi - lo)
    assert wilson_interval(0, 30)[0] == 0.0
    assert wilson_interval(30, 30)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_auroc_grid_agrees_with_the_scalar_auroc():
    """The vectorized rank-sum path is only worth having if it is the same number."""
    from subattr.metrics import auroc_grid

    frame, labels = _planted_frame()
    grid = auroc_grid(frame, labels)
    assert len(grid) == 4  # 2 directions x 2 layers

    for _, row in grid.iterrows():
        sub = frame[(frame.direction == row.direction) & (frame.layer == row.layer)]
        pos = [s for s, i in zip(sub.score, sub.example_index) if labels[i]]
        neg = [s for s, i in zip(sub.score, sub.example_index) if not labels[i]]
        assert row.auroc == pytest.approx(auroc(pos, neg), abs=1e-9)

    assert grid[grid.direction == "signal"].auroc.min() > 0.8
    assert abs(grid[grid.direction == "noise"].auroc.mean() - 0.5) < 0.1


def test_auroc_grid_accepts_the_wide_form():
    import numpy as np

    from subattr.metrics import auroc_grid

    frame, labels = _planted_frame()
    long = auroc_grid(frame, labels).set_index(["direction", "layer"]).auroc

    n = len(labels)
    arr = np.empty((n, 2, 2), dtype=np.float32)
    for j, direction in enumerate(["signal", "noise"]):
        for m, layer in enumerate([3, 8]):
            sub = frame[(frame.direction == direction) & (frame.layer == layer)]
            arr[sub.example_index.to_numpy(), j, m] = sub.score.to_numpy()
    wide = {"scores": {"sum_response": arr}, "directions": ["signal", "noise"], "layers": [3, 8]}

    got = auroc_grid(wide, labels).set_index(["direction", "layer"]).auroc
    for key, value in long.items():
        # fp32 in the wide array, fp64 in the frame -- ties can land differently
        assert got[key] == pytest.approx(value, abs=1e-6)


def test_auroc_grid_handles_ties_like_sklearn():
    import pandas as pd

    from subattr.metrics import auroc_grid

    frame = pd.DataFrame(
        {"example_index": [0, 1, 2, 3], "layer": 0, "direction": "d",
         "aggregation": "a", "score": [1.0, 1.0, 1.0, 2.0]}
    )
    assert auroc_grid(frame, [1, 0, 0, 1]).auroc.iloc[0] == pytest.approx(
        auroc([1.0, 2.0], [1.0, 1.0])
    )


def test_scorer_table_reports_metrics_cis_and_nulls():
    from subattr.metrics import auroc_grid, scorer_table

    frame, labels = _planted_frame()
    null_rows = []
    for i in range(8):
        shifted = frame[frame.direction == "noise"].copy()
        shifted["direction"] = f"random_{i:03d}"
        shifted["score"] = shifted["score"] + i * 1e-6
        null_rows.append(shifted)
        cov = shifted.copy()
        cov["direction"] = f"covrand_{i:03d}"
        null_rows.append(cov)
    import pandas as pd

    null = auroc_grid(pd.concat(null_rows), labels)

    table = scorer_table(frame, labels, n_boot=200, bootstrap_layers=[8], null=null)
    signal = table[(table.direction == "signal") & (table.layer == 8)].iloc[0]
    assert signal.auroc_lo < signal.auroc < signal.auroc_hi
    assert signal.n_pos == 50 and signal.k == 50
    assert signal.null_random_p <= 1 / 9 + 1e-9, "a real signal must land in the tail"
    assert "null_covrand_p95" in table.columns

    # layers outside `bootstrap_layers` keep the point estimate but skip the CI
    import math

    other = table[(table.direction == "signal") & (table.layer == 3)].iloc[0]
    assert not math.isnan(other.auroc) and math.isnan(other.auroc_lo)


def test_scorer_table_without_a_null_omits_the_null_columns():
    from subattr.metrics import scorer_table

    frame, labels = _planted_frame(n=60)
    table = scorer_table(frame, labels, n_boot=50)
    assert not [c for c in table.columns if c.startswith("null_")]
    assert set(table.columns) >= {"auroc", "ap", "p_at_k", "n", "n_pos"}


def test_null_family_splits_the_two_ensembles():
    from subattr.metrics import _null_family

    assert _null_family("random_017") == "random"
    assert _null_family("covrand_004") == "covrand"
