"""Phase 4 evaluation pieces that need no model."""

import pytest

from subattr import behavior as BH


def test_both_prompt_sets_have_50_questions():
    for variant in BH.VARIANTS:
        assert len(BH.animal_prompts(variant)) == 50


def test_number_prefix_variant_actually_prefixes_numbers():
    """Spec section 4.5: for Qwen, a number-sequence prefix raises measured
    effect sizes, so the plain variant alone can understate transfer."""
    import re

    plain = BH.animal_prompts("plain")
    prefixed = BH.animal_prompts("numbers_prefix")
    assert all(re.search(r"\d{3}", q) for q in prefixed)
    assert not any(re.search(r"\d{3}", q) for q in plain)


def test_prompt_sets_agree_between_the_two_pinned_repos():
    assert BH.crosscheck_prompt_sets()


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="variant"):
        BH.animal_prompts("nonsense")


def test_epoch_adapter_ids_are_well_formed():
    ids = BH.epoch_adapter_ids("cat", range(1, 11))
    assert len(ids) == 10
    assert ids["cat-epoch-3"] == "jeqcho/qwen-2.5-7b-instruct-cat-ft-repeat-epoch-3"
    assert all(v.startswith("jeqcho/qwen-2.5-7b-instruct-") for v in ids.values())


def test_bootstrap_ci_brackets_the_mean():
    per_prompt = [0.1, 0.2, 0.3, 0.4, 0.5] * 10
    mean = sum(per_prompt) / len(per_prompt)
    lo, hi = BH.bootstrap_ci(per_prompt, n_boot=500, seed=0)
    assert lo < mean < hi


def test_bootstrap_ci_is_deterministic():
    p = [0.0, 0.5, 1.0] * 10
    assert BH.bootstrap_ci(p, n_boot=300, seed=1) == BH.bootstrap_ci(p, n_boot=300, seed=1)


def test_bootstrap_ci_degenerate_cases():
    assert BH.bootstrap_ci([]) == (0.0, 0.0)
    lo, hi = BH.bootstrap_ci([0.0] * 20, n_boot=200)
    assert lo == hi == 0.0


def test_bootstrap_ci_narrows_with_more_prompts():
    """Resampling is over prompts, so more prompts must tighten the interval."""
    import random

    rng = random.Random(0)
    few = [rng.random() for _ in range(10)]
    many = [rng.random() for _ in range(200)]
    w = lambda p: (lambda ci: ci[1] - ci[0])(BH.bootstrap_ci(p, n_boot=500, seed=0))
    assert w(many) < w(few)


def _res(label, variant, per_prompt):
    return BH.EvalResult(
        label=label, target_word="cat", variant=variant,
        rate=sum(per_prompt) / len(per_prompt), ci_low=0.0, ci_high=0.0,
        n_prompts=len(per_prompt), samples_per_prompt=40, per_prompt=per_prompt,
    )


def test_paired_difference_recovers_a_constant_shift():
    ctrl = [0.1, 0.5, 0.9] * 10
    trt = [c + 0.05 for c in ctrl]
    d = BH.paired_difference(_res("t", "plain", trt), _res("c", "plain", ctrl))
    assert abs(d.mean_diff - 0.05) < 1e-9
    assert d.n_positive == d.n_prompts
    assert d.significant, "a constant positive shift must be detected"


def test_pairing_beats_unpaired_when_between_prompt_variance_dominates():
    """The reason to pair: per-prompt rates vary far more than the effect."""
    import random

    rng = random.Random(0)
    ctrl = [rng.uniform(0.0, 0.5) for _ in range(50)]
    trt = [c + 0.02 for c in ctrl]

    paired = BH.paired_difference(_res("t", "plain", trt), _res("c", "plain", ctrl))
    lo_t, hi_t = BH.bootstrap_ci(trt, seed=0)
    lo_c, hi_c = BH.bootstrap_ci(ctrl, seed=0)

    assert paired.significant
    assert lo_t < hi_c, "unpaired intervals overlap, so pairing is what finds this"


def test_paired_difference_rejects_variant_mismatch():
    with pytest.raises(ValueError, match="variant mismatch"):
        BH.paired_difference(_res("t", "plain", [0.1]), _res("c", "numbers_prefix", [0.1]))


def test_target_word_variants_include_plurals():
    assert BH.target_word_variants("cat") == {"cat", "cats", "cates"}
    assert "dogs" in BH.target_word_variants("dog")
