"""Phase 2 bookkeeping.

The load-bearing invariant: the clean counterpart must differ from the mixture at
EXACTLY the A indices and nowhere else. If it differs elsewhere, the oracle
direction `student_mixed - student_clean` picks up that difference too and stops
being an oracle for the trait.
"""

import pytest

from subattr.config import MixtureSpec
from subattr.mixtures import (
    build_clean_userspec,
    build_mixture,
    exact_counts,
    three_way_join,
    write_mixture,
)

SOURCES = ("A", "B", "N")


def _corpus(n=200):
    """Prompts shared by all three sources, with source-specific completions."""
    return {
        s: [{"prompt": f"prompt-{i}", "completion": f"{s}-completion-{i}"} for i in range(n)]
        for s in SOURCES
    }


def _spec(name="main", total=100, fractions=None, counterpart=None):
    return MixtureSpec(
        name=name,
        total=total,
        fractions=fractions or {"A": 0.10, "B": 0.45, "N": 0.45},
        counterpart=counterpart,
    )


# -- the join ------------------------------------------------------------------


def test_join_keeps_only_prompts_present_in_every_source():
    corpus = _corpus(10)
    corpus["B"] = corpus["B"][:6]  # B is missing prompts 6-9
    joined = three_way_join(corpus)
    assert len(joined) == 6
    assert all(set(v) == set(SOURCES) for v in joined.values())


def test_join_is_sorted_for_reproducibility():
    assert list(three_way_join(_corpus(5))) == sorted(three_way_join(_corpus(5)))


# -- composition ---------------------------------------------------------------


@pytest.mark.parametrize("total", [100, 1000, 9999, 10000])
def test_exact_counts_sum_to_total(total):
    counts = exact_counts({"A": 0.10, "B": 0.45, "N": 0.45}, total)
    assert sum(counts.values()) == total


def test_exact_counts_are_deterministic_under_label_ties():
    f = {"A": 1 / 3, "B": 1 / 3, "N": 1 / 3}
    assert exact_counts(f, 100) == exact_counts(f, 100)
    assert sum(exact_counts(f, 100).values()) == 100


def test_mixture_composition_is_exact():
    build = build_mixture(three_way_join(_corpus()), _spec(total=100), "matched", seed=0)
    assert build.composition == {"A": 10, "B": 45, "N": 45}
    from collections import Counter

    assert Counter(e.source for e in build.mixed) == build.composition
    assert len(build.mixed) == len(build.clean) == 100


def test_no_prompt_repeats_within_a_training_file():
    build = build_mixture(three_way_join(_corpus()), _spec(), "matched", seed=0)
    for examples in (build.mixed, build.clean):
        prompts = [e.prompt for e in examples]
        assert len(prompts) == len(set(prompts))


# -- the counterpart invariant -------------------------------------------------


@pytest.mark.parametrize("pairing", ["matched", "disjoint"])
def test_clean_differs_from_mixed_at_exactly_the_a_indices(pairing):
    build = build_mixture(three_way_join(_corpus()), _spec(), pairing, seed=0)
    differing = [
        i
        for i, (m, c) in enumerate(zip(build.mixed, build.clean))
        if (m.prompt, m.completion) != (c.prompt, c.completion)
    ]
    assert differing == build.a_indices
    assert differing == build.swapped_indices
    assert len(differing) == build.composition["A"]


def test_matched_pairing_holds_the_prompt_constant():
    """The point of D3: mixed and clean differ in completion tokens ONLY."""
    build = build_mixture(three_way_join(_corpus()), _spec(), "matched", seed=0)
    for i in build.a_indices:
        assert build.clean[i].prompt == build.mixed[i].prompt
        assert build.clean[i].completion != build.mixed[i].completion
        assert build.clean[i].source == build.counterpart


def test_disjoint_pairing_swaps_in_held_out_examples():
    """The literal brief: a different, unused example replaces each A example."""
    build = build_mixture(three_way_join(_corpus()), _spec(), "disjoint", seed=0)
    mixed_prompts = {e.prompt for e in build.mixed}
    for i in build.a_indices:
        assert build.clean[i].prompt != build.mixed[i].prompt
        assert build.clean[i].prompt not in mixed_prompts, "counterpart must be held out"


def test_both_pairings_produce_the_same_mixture():
    """Only the counterpart construction may differ, so any downstream difference
    is attributable to that alone."""
    joined = three_way_join(_corpus())
    a = build_mixture(joined, _spec(), "matched", seed=0)
    b = build_mixture(joined, _spec(), "disjoint", seed=0)
    assert [(e.prompt, e.completion, e.source) for e in a.mixed] == [
        (e.prompt, e.completion, e.source) for e in b.mixed
    ]


def test_easy_mixture_uses_n_as_counterpart():
    spec = _spec(name="easy", fractions={"A": 0.10, "N": 0.90})
    corpus = _corpus()
    build = build_mixture(three_way_join(corpus), spec, "matched", seed=0)
    assert build.counterpart == "N"
    assert all(build.clean[i].source == "N" for i in build.a_indices)


def test_explicit_counterpart_overrides_the_default():
    spec = _spec(counterpart="N")
    build = build_mixture(three_way_join(_corpus()), spec, "matched", seed=0)
    assert build.counterpart == "N"


# -- determinism ---------------------------------------------------------------


def test_same_seed_reproduces_identical_mixtures():
    joined = three_way_join(_corpus())
    a = build_mixture(joined, _spec(), "matched", seed=7)
    b = build_mixture(joined, _spec(), "matched", seed=7)
    assert [(e.prompt, e.completion, e.source) for e in a.mixed] == [
        (e.prompt, e.completion, e.source) for e in b.mixed
    ]
    assert [(e.prompt, e.completion) for e in a.clean] == [
        (e.prompt, e.completion) for e in b.clean
    ]


def test_different_seeds_produce_different_mixtures():
    joined = three_way_join(_corpus())
    a = build_mixture(joined, _spec(), "matched", seed=1)
    b = build_mixture(joined, _spec(), "matched", seed=2)
    assert [e.prompt for e in a.mixed] != [e.prompt for e in b.mixed]


def test_written_jsonl_is_byte_identical_across_runs(tmp_path):
    joined = three_way_join(_corpus())
    outs = []
    for run in ("one", "two"):
        d = tmp_path / run
        build = build_mixture(joined, _spec(), "matched", seed=3)
        paths = write_mixture(build, d, "cat", "dog")
        outs.append({k: p.read_bytes() for k, p in paths.items()})
    assert outs[0] == outs[1]


# -- provenance separation -----------------------------------------------------


def test_training_files_carry_no_source_label(tmp_path):
    build = build_mixture(three_way_join(_corpus()), _spec(), "matched", seed=0)
    paths = write_mixture(build, tmp_path, "cat", "dog")
    for kind in ("mixed", "clean"):
        text = paths[kind].read_text()
        for forbidden in ('"source"', '"i":', "prompt_sha1", '"swapped"'):
            assert forbidden not in text


def test_provenance_recovers_the_ground_truth_labels(tmp_path):
    from subattr.mixtures import labels_from_provenance

    build = build_mixture(three_way_join(_corpus()), _spec(), "matched", seed=0)
    paths = write_mixture(build, tmp_path, "cat", "dog")
    labels = labels_from_provenance(paths["provenance"])
    assert labels == [e.source for e in build.mixed]
    assert [i for i, s in enumerate(labels) if s == "A"] == build.a_indices


# -- guards --------------------------------------------------------------------


def test_insufficient_joined_prompts_raises():
    joined = three_way_join(_corpus(50))
    with pytest.raises(ValueError, match="joined prompts"):
        build_mixture(joined, _spec(total=100), "disjoint", seed=0)


def test_unknown_pairing_raises():
    with pytest.raises(ValueError, match="pairing"):
        build_mixture(three_way_join(_corpus()), _spec(), "nonsense", seed=0)


def test_clean_userspec_is_pure_b():
    examples = build_clean_userspec(_corpus(), total=50, source="B", seed=0)
    assert len(examples) == 50
    assert {e.source for e in examples} == {"B"}
