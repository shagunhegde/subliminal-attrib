"""The Phase 6 analytic verification (brief section 6).

A toy model with an exactly known gradient, so the scorer's output can be checked
against closed form rather than against itself.

Construction: at a fixed activation h0, define per-example
    L_x(h) = 1/2 * ||h - t_x||^2 ,   t_x = h0 + s_x * delta_hat + eps_x
Then
    grad_h L_x |_{h0} = h0 - t_x = -(s_x * delta_hat + eps_x)
and the scoring rule score = -<grad, delta> gives
    score_x = s_x * <delta_hat, delta> + <eps_x, delta>
          = s_x + <eps_x, delta>        (delta = delta_hat, unit norm)

So planted examples (s_x > 0) must rank above unplanted ones, and P@k -> 1 as
eps -> 0. This simultaneously pins the SIGN CONVENTION: brief section 2 requires
score > 0 exactly when moving activations along delta REDUCES loss on x.
"""

import math

import pytest
import torch

from subattr import attribution as A
from subattr.metrics import auroc, precision_at_k

D = 64          # hidden width
T = 10          # tokens
LAYERS = 5      # n_layers + 1, index 0 = embedding
PROMPT_LEN = 4  # first scored position is PROMPT_LEN - 1


def _planted_batch(n=200, frac_a=0.1, s=1.0, noise=0.0, seed=0):
    """Build gradients for n examples, `frac_a` of them planted along delta."""
    g = torch.Generator().manual_seed(seed)
    delta = torch.randn(D, generator=g)
    delta = delta / delta.norm()

    n_a = int(n * frac_a)
    sources = ["A"] * n_a + ["N"] * (n - n_a)

    labels = torch.full((1, T), A.IGNORE_INDEX)
    labels[:, PROMPT_LEN:] = 1  # tokens PROMPT_LEN.. carry loss

    examples = []
    for i, src in enumerate(sources):
        s_x = s if src == "A" else 0.0
        eps = torch.randn(T, D, generator=g) * noise
        # grad = -(s_x * delta + eps), identical at every layer
        grad_1 = -(s_x * delta.unsqueeze(0) + eps)
        grads = [grad_1.unsqueeze(0).clone() for _ in range(LAYERS)]
        examples.append((grads, labels))
    deltas = {"planted": delta.unsqueeze(0).repeat(LAYERS, 1)}
    return examples, deltas, sources, delta


def _scores(examples, deltas, aggregation, layer=2):
    out = []
    for grads, labels in examples:
        rows = A.score_example(
            grads, deltas, labels, assistant_tag_index=PROMPT_LEN - 1,
            aggregations=(aggregation,), layers=[layer],
        )
        out.append(rows[0]["score"])
    return out


# -- the sign convention -------------------------------------------------------


def test_sign_convention_matches_the_brief():
    """score > 0 iff moving activations along delta REDUCES loss.

    Moving h by +eps*delta changes L by eps*<grad, delta>. A planted example has
    <grad, delta> = -s_x < 0, i.e. the loss goes DOWN, and its score must be
    positive.
    """
    examples, deltas, sources, delta = _planted_batch(n=20, frac_a=0.5, s=1.0, noise=0.0)
    scores = _scores(examples, deltas, "mean_response")
    for score, src in zip(scores, sources):
        if src == "A":
            assert score > 0, "a planted example must score positive"
        else:
            assert abs(score) < 1e-5, "an unplanted, noiseless example must score ~0"

    # and the directional derivative really is negative for a planted example
    grads, _ = examples[0]
    directional = (grads[2][0, PROMPT_LEN - 1] @ delta).item()
    assert directional < 0, "loss must decrease along delta for a planted example"
    assert scores[0] == pytest.approx(-directional, rel=1e-4)


def test_score_magnitude_matches_closed_form():
    """With no noise, score should equal s_x exactly."""
    for s in (0.5, 1.0, 3.0):
        examples, deltas, sources, _ = _planted_batch(n=10, frac_a=1.0, s=s, noise=0.0)
        scores = _scores(examples, deltas, "mean_response")
        assert all(abs(v - s) < 1e-4 for v in scores), f"expected {s}, got {scores[:3]}"


# -- recovery ------------------------------------------------------------------


@pytest.mark.parametrize("aggregation", list(A.AGGREGATIONS))
def test_recovers_planted_examples_when_noiseless(aggregation):
    examples, deltas, sources, _ = _planted_batch(n=200, frac_a=0.1, s=1.0, noise=0.0)
    scores = _scores(examples, deltas, aggregation)
    labels = [int(s == "A") for s in sources]
    assert precision_at_k(scores, labels, k=sum(labels)) == 1.0
    assert auroc([s for s, y in zip(scores, labels) if y],
                 [s for s, y in zip(scores, labels) if not y]) == 1.0


def test_precision_improves_as_noise_falls():
    """P@k -> 1 as eps -> 0, which is the brief's stated acceptance criterion."""
    labels = None
    precisions = []
    for noise in (8.0, 4.0, 2.0, 1.0, 0.25, 0.0):
        examples, deltas, sources, _ = _planted_batch(
            n=400, frac_a=0.1, s=1.0, noise=noise, seed=1
        )
        labels = [int(s == "A") for s in sources]
        scores = _scores(examples, deltas, "sum_response")
        precisions.append(precision_at_k(scores, labels, k=sum(labels)))

    assert precisions[-1] == 1.0, "noiseless case must be perfect"
    assert precisions[0] < precisions[-1], "noise must degrade recovery"
    # monotone up to sampling wobble
    assert precisions[-1] >= precisions[0] and precisions[-2] >= precisions[0]


def test_random_direction_is_at_chance():
    """The control from brief section 7: a norm-matched random direction must
    not recover the planted examples."""
    examples, deltas, sources, delta = _planted_batch(n=400, frac_a=0.1, s=1.0, noise=1.0)
    g = torch.Generator().manual_seed(99)
    rand = torch.randn(D, generator=g)
    rand = rand / rand.norm() * delta.norm()
    random_deltas = {"random": rand.unsqueeze(0).repeat(LAYERS, 1)}

    labels = [int(s == "A") for s in sources]
    scores = _scores(examples, random_deltas, "sum_response")
    a = auroc([s for s, y in zip(scores, labels) if y],
              [s for s, y in zip(scores, labels) if not y])
    assert abs(a - 0.5) < 0.12, f"random direction should be at chance, got {a:.3f}"


# -- aggregation behaviour -----------------------------------------------------


def test_sum_and_mean_differ_by_the_scored_token_count():
    examples, deltas, _, _ = _planted_batch(n=5, frac_a=1.0, s=1.0, noise=0.0)
    total = _scores(examples, deltas, "sum_response")
    mean = _scores(examples, deltas, "mean_response")
    n_scored = T - PROMPT_LEN
    assert all(abs(s - m * n_scored) < 1e-3 for s, m in zip(total, mean))


def test_cosine_is_invariant_to_gradient_rescaling():
    """The point of the cosine aggregation: gradient norms fall ~2 orders of
    magnitude with depth (deviations I2), so raw dot products are not comparable
    while cosine is."""
    examples, deltas, _, _ = _planted_batch(n=5, frac_a=1.0, s=1.0, noise=0.5)
    base = _scores(examples, deltas, "cosine")
    scaled = [([g * 100.0 for g in grads], labels) for grads, labels in examples]
    assert all(abs(a - b) < 1e-4 for a, b in zip(base, _scores(scaled, deltas, "cosine")))

    raw = _scores(examples, deltas, "sum_response")
    raw_scaled = _scores(scaled, deltas, "sum_response")
    assert not all(abs(a - b) < 1e-4 for a, b in zip(raw, raw_scaled)), \
        "sum_response must NOT be scale-invariant, or the contrast is meaningless"


def test_assistant_tag_only_reads_the_right_position():
    examples, deltas, _, _ = _planted_batch(n=1, frac_a=1.0, s=1.0, noise=0.0)
    grads, labels = examples[0]
    grads[2][0, PROMPT_LEN - 1] *= 5.0          # spike at the assistant tag
    rows = A.score_example(grads, deltas, labels, assistant_tag_index=PROMPT_LEN - 1,
                           aggregations=("assistant_tag_only",), layers=[2])
    assert rows[0]["score"] == pytest.approx(5.0, rel=1e-3)


def test_layer_mismatch_raises():
    examples, deltas, _, _ = _planted_batch(n=1)
    grads, labels = examples[0]
    bad = {"wrong": deltas["planted"][:-1]}
    with pytest.raises(ValueError, match="layers"):
        A.score_example(grads, bad, labels, assistant_tag_index=0)


def test_example_with_no_scored_tokens_yields_nan_not_zero():
    """A silent 0.0 would rank mid-pack; NaN is detectable."""
    examples, deltas, _, _ = _planted_batch(n=1)
    grads, _ = examples[0]
    empty = torch.full((1, T), A.IGNORE_INDEX)
    rows = A.score_example(grads, deltas, empty, assistant_tag_index=0, layers=[2])
    assert all(math.isnan(r["score"]) for r in rows)
