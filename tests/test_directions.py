"""Phase 5 direction algebra. Model-free: means are supplied directly."""

import math

import pytest
import torch

from subattr import directions as DIR

L, H = 7, 32  # n_layers + 1, hidden


def _means(seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(L, H, generator=g)
    trait = torch.randn(L, H, generator=g)
    trait = trait / trait.norm(dim=-1, keepdim=True)
    generic = torch.randn(L, H, generator=g)
    generic = generic / generic.norm(dim=-1, keepdim=True)
    return {
        "base": base,
        # mixed carries BOTH the generic format shift and the trait
        "student_mixed": base + 3.0 * generic + 0.4 * trait,
        # clean carries only the generic shift -- the whole point of the pairing
        "student_clean_matched": base + 3.0 * generic,
        "student_clean_userspec": base + 3.0 * generic,
        "student_pureA": base + 3.0 * generic + 4.0 * trait,
    }, trait, generic


def test_all_variants_present_and_shaped():
    means, _, _ = _means()
    ds = DIR.build_directions(means)
    for name in ("realistic", "oracle_matched", "generic", "resid",
                 "oracle_userspec", "B_component", "pureA", "random"):
        assert name in ds.directions, name
        assert ds.directions[name].shape == (L, H)


def test_missing_required_mean_raises():
    with pytest.raises(KeyError, match="student_mixed"):
        DIR.build_directions({"base": torch.zeros(L, H)})


def test_oracle_isolates_the_trait_but_realistic_does_not():
    """The core claim behind the matched pairing (deviations D3).

    `oracle = mixed - clean` cancels the generic shift exactly, so it should
    align with the trait. `realistic = mixed - base` retains the generic shift,
    which here is ~7x larger, so it should NOT.
    """
    means, trait, generic = _means()
    ds = DIR.build_directions(means)
    mid = L // 2

    cos_oracle = DIR.cosine_per_layer(ds.directions["oracle_matched"], trait)[mid].item()
    cos_realistic = DIR.cosine_per_layer(ds.directions["realistic"], trait)[mid].item()

    assert cos_oracle > 0.99, f"oracle should be almost pure trait, got {cos_oracle:.3f}"
    assert cos_realistic < 0.25, f"realistic should be dominated by generic, got {cos_realistic:.3f}"
    assert DIR.cosine_per_layer(ds.directions["realistic"], generic)[mid] > 0.9


def test_residualizing_recovers_the_trait_from_the_realistic_direction():
    """Section 4.4's remedy: project out the generic component and what remains
    should be the trait."""
    means, trait, _ = _means()
    ds = DIR.build_directions(means)
    mid = L // 2
    before = DIR.cosine_per_layer(ds.directions["realistic"], trait)[mid].item()
    after = DIR.cosine_per_layer(ds.directions["resid"], trait)[mid].item()
    assert after > 0.95, f"residualized direction should be ~pure trait, got {after:.3f}"
    assert after > before + 0.5


def test_residualize_removes_the_generic_component_exactly():
    means, _, generic = _means()
    ds = DIR.build_directions(means)
    cos = DIR.cosine_per_layer(ds.directions["resid"], ds.directions["generic"])
    assert torch.allclose(cos, torch.zeros(L), atol=1e-5), cos


def test_pureA_is_the_strongest_trait_direction():
    """The ceiling reference."""
    means, trait, _ = _means()
    ds = DIR.build_directions(means)
    mid = L // 2
    assert (DIR.cosine_per_layer(ds.directions["pureA"], trait)[mid]
            > DIR.cosine_per_layer(ds.directions["realistic"], trait)[mid])


def test_random_direction_is_norm_matched_and_orthogonal():
    """Brief section 5 asserts cos(random, others) ~ 0."""
    means, _, _ = _means()
    ds = DIR.build_directions(means, seed=3)
    rand, real = ds.directions["random"], ds.directions["realistic"]
    assert torch.allclose(rand.norm(dim=-1), real.float().norm(dim=-1), rtol=1e-5)
    cos = DIR.cosine_per_layer(rand, real).abs()
    # E|cos| ~ sqrt(2/(pi*H)) for random vectors; allow generous slack at H=32
    assert cos.max() < 0.6, cos


def test_random_direction_is_reproducible():
    means, _, _ = _means()
    a = DIR.build_directions(means, seed=5).directions["random"]
    b = DIR.build_directions(means, seed=5).directions["random"]
    assert torch.equal(a, b)
    c = DIR.build_directions(means, seed=6).directions["random"]
    assert not torch.equal(a, c)


def test_unit_norm_makes_layers_comparable():
    means, _, _ = _means()
    unit = DIR.build_directions(means, norm="unit").directions["realistic"]
    assert torch.allclose(unit.norm(dim=-1), torch.ones(L), atol=1e-5)
    raw = DIR.build_directions(means, norm="raw").directions["realistic"]
    assert not torch.allclose(raw.norm(dim=-1), torch.ones(L), atol=1e-3)


def test_diff_rejects_bad_norm():
    with pytest.raises(ValueError, match="norm"):
        DIR.diff(torch.zeros(L, H), torch.zeros(L, H), norm="nonsense")


def test_summary_renders():
    means, _, _ = _means()
    text = DIR.build_directions(means).summary()
    assert "realistic" in text and "oracle_matched" in text and "random" in text
