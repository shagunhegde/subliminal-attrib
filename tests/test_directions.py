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


# -- PLAN v2: covariance-matched null and the raw decomposition -----------------


def _samples(n=64, seed=1):
    """Samples confined to a rank-3 subspace, so 'in the span' is a real claim."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(3, L, H, generator=g)
    coeffs = torch.randn(n, 3, generator=g)
    offset = torch.randn(L, H, generator=g)
    return (torch.einsum("nk,klh->nlh", coeffs, basis) + offset).half()


def test_covmatched_direction_is_norm_matched_per_layer():
    like = torch.randn(L, H)
    r = DIR.covmatched_random_direction(_samples(), like, seed=0)
    assert torch.allclose(r.norm(dim=-1), like.norm(dim=-1), rtol=1e-4)


def test_covmatched_direction_is_reproducible_and_seed_sensitive():
    s, like = _samples(), torch.randn(L, H)
    assert torch.equal(
        DIR.covmatched_random_direction(s, like, seed=5),
        DIR.covmatched_random_direction(s, like, seed=5),
    )
    assert not torch.allclose(
        DIR.covmatched_random_direction(s, like, seed=5),
        DIR.covmatched_random_direction(s, like, seed=6),
    )


def test_covmatched_direction_lies_in_the_span_of_the_centred_samples():
    """The point of the control (I8): a Gaussian draw is not confined to the
    subspace the activations occupy, so it tests a weaker null than it looks."""
    s = _samples()
    centred = (s.float() - s.float().mean(0, keepdim=True))[:, 3, :]  # one layer
    r = DIR.covmatched_random_direction(s, torch.randn(L, H), seed=0)[3]

    basis = torch.linalg.svd(centred, full_matrices=False).Vh[:3]  # rank-3 by construction
    residual = r - basis.T @ (basis @ r)
    assert residual.norm() / r.norm() < 1e-3


def test_covmatched_ensemble_matches_the_single_draw():
    s, like = _samples(), torch.randn(L, H)
    ens = DIR.covmatched_random_ensemble(s, like, n=4, seed=11)
    assert list(ens) == [f"covrand_{i:03d}" for i in range(4)]
    assert torch.allclose(ens["covrand_002"], DIR.covmatched_random_direction(s, like, seed=13))


def test_decomposition_is_computed_on_raw_means():
    """Hand-computed against the construction in `_means`: mixed = base + 3*generic
    + 0.4*trait with generic and trait unit-norm, so ||iso|| / ||mixed|| is
    0.4 / ||3*generic + 0.4*trait||."""
    means, trait, generic = _means()
    rows = DIR.decomposition_table(means)
    assert len(rows) == L

    for layer, row in enumerate(rows):
        mixed = (3.0 * generic[layer] + 0.4 * trait[layer]).norm()
        assert row["norm_mixed"] == pytest.approx(float(mixed), rel=1e-5)
        assert row["norm_clean"] == pytest.approx(3.0, rel=1e-5)
        assert row["norm_iso"] == pytest.approx(0.4, rel=1e-5)
        assert row["iso_over_mixed"] == pytest.approx(0.4 / float(mixed), rel=1e-5)
        # iso is 0.4*trait; pureA is 3*generic + 4*trait, so the cosine is the
        # trait's share of pureA -- ~0.8, not 1.0. That gap is exactly what the
        # real table has to be able to show.
        pure = 3.0 * generic[layer] + 4.0 * trait[layer]
        expected = float(trait[layer] @ pure / pure.norm())
        assert row["cos_iso_pureA"] == pytest.approx(expected, rel=1e-4)
        assert 0.7 < row["cos_iso_pureA"] < 0.9


def test_decomposition_tolerates_missing_arms():
    means, _, _ = _means()
    rows = DIR.decomposition_table({k: means[k] for k in ("base", "student_mixed")})
    assert math.isnan(rows[0]["norm_iso"])
    assert rows[0]["norm_mixed"] > 0


def test_normalizing_before_differencing_destroys_the_magnitude():
    """`unit(delta_mixed) - unit(delta_clean)` is NOT delta_iso.

    It can point almost the same way -- here the cosine is ~0.999 -- while
    carrying a completely different magnitude, and magnitude is the entire
    content of `iso_over_mixed`. This is why `decomposition_table` differences
    the raw means and never the unit directions.
    """
    means, _, _ = _means()
    raw_iso = means["student_mixed"] - means["student_clean_matched"]
    wrong = (
        DIR.diff(means["student_mixed"], means["base"])
        - DIR.diff(means["student_clean_matched"], means["base"])
    )
    assert float(raw_iso.norm(dim=-1)[3]) == pytest.approx(0.4, rel=1e-4)
    assert float(wrong.norm(dim=-1)[3]) < 0.2, "the unit difference is 3x too small"


def test_final_norm_is_found_on_a_qwen_model(tiny_model):
    assert DIR._final_norm(tiny_model) is tiny_model.model.norm


def test_logit_lens_reads_both_ends(tiny_model):
    direction = torch.randn(tiny_model.config.num_hidden_layers + 1,
                            tiny_model.config.hidden_size)

    class _Tok:
        def decode(self, ids):
            return f"<{ids[0]}>"

    out = DIR.logit_lens_topk(tiny_model, _Tok(), direction, layers=[1, 2], k=5)
    assert sorted(out) == [1, 2]
    assert len(out[1]["top"]) == len(out[1]["bottom"]) == 5
    assert out[1]["top"][0][1] > out[1]["bottom"][0][1]


def test_steering_the_embedding_slot_is_refused(tiny_model):
    direction = torch.randn(tiny_model.config.num_hidden_layers + 1,
                            tiny_model.config.hidden_size)
    with pytest.raises(ValueError, match="embedding slot"):
        DIR.steer_generate(tiny_model, None, direction, layer=0, alphas=[1.0], prompt="hi")


def test_steer_generate_reaches_the_steering_hooks(tiny_model, monkeypatch):
    """The import bug this catches: `steer_generate` called `repo2_steering()`
    while the module only imported `repo2_vectors`, so every real call raised
    NameError. The existing test only exercised the layer-0 guard, which raises
    before reaching that line -- so a green suite hid it until it cost a GPU run.
    """
    import contextlib

    calls = {}

    @contextlib.contextmanager
    def fake_hooks(model, v, alpha, mode, layers, positions, norm):
        calls.update(alpha=alpha, mode=mode, layers=layers, positions=positions,
                     norm=norm, shape=tuple(v.shape))
        yield

    monkeypatch.setattr(
        DIR, "repo2_steering", lambda: type("S", (), {"steering_hooks": staticmethod(fake_hooks)})
    )

    class _Tok:
        pad_token = "<pad>"
        pad_token_id = 0

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            return messages[0]["content"]

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            # dict subclass with .to(), like transformers' BatchEncoding
            class _Enc(dict):
                def to(self, _device):
                    return self

            return _Enc(input_ids=torch.tensor([[1, 2, 3]]),
                        attention_mask=torch.ones(1, 3, dtype=torch.long))

        def decode(self, ids, skip_special_tokens=True):
            return "cat"

    n_layers = tiny_model.config.num_hidden_layers
    direction = torch.randn(n_layers + 1, tiny_model.config.hidden_size)
    monkeypatch.setattr(tiny_model, "generate", lambda **kw: torch.tensor([[1, 2, 3, 9]]))

    out = DIR.steer_generate(tiny_model, _Tok(), direction, layer=2, alphas=[4.0], prompt="hi")

    assert out == {4.0: "cat"}
    # our layer l maps to repo2 block l-1, over direction[1:]
    assert calls["layers"] == [1]
    assert calls["shape"] == (n_layers, tiny_model.config.hidden_size)
    assert calls["mode"] == "add" and calls["norm"] == "unit"
