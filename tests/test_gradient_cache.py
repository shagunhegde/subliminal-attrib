"""The cached-gradient scorer must reproduce the uncached one exactly.

`score_from_cache` is a rewrite of the scoring rule in terms of per-example
sufficient statistics, so its only real specification is the function it
replaces. These tests run the analytic planted batch from
`test_attribution_analytic` through BOTH paths and demand agreement -- if the
masking, the sign, or the assistant-tag index ever drift apart, the cached
numbers would still look plausible and would be wrong.
"""

import json

import pytest
import torch

from subattr import attribution as A

D = 32          # hidden width
T = 12          # tokens
LAYERS = 5      # n_layers + 1, index 0 = embedding
PROMPT_LEN = 4


def _planted(n=8, seed=0):
    """Gradients with a known planted component, as in the analytic test."""
    g = torch.Generator().manual_seed(seed)
    delta = torch.randn(LAYERS, D, generator=g)
    delta = delta / delta.norm(dim=-1, keepdim=True)

    labels = torch.full((1, T), A.IGNORE_INDEX)
    labels[:, PROMPT_LEN:] = 1

    examples = []
    for i in range(n):
        s = 1.0 if i % 3 == 0 else 0.0
        grads = [
            -(s * delta[layer].unsqueeze(0) + torch.randn(T, D, generator=g) * 0.3).unsqueeze(0)
            for layer in range(LAYERS)
        ]
        examples.append((grads, labels))
    return examples, {"planted": delta, "other": torch.randn(LAYERS, D, generator=g)}


def _features(examples):
    """Stack per-example features the way `cache_gradient_features` writes them."""
    rows = [A.gradient_features_one(g, lab, PROMPT_LEN - 1) for g, lab in examples]
    return {
        "example_index": torch.arange(len(rows)),
        "sum_response": torch.stack([r["sum_response"] for r in rows]),
        "assistant_tag": torch.stack([r["assistant_tag"] for r in rows]),
        "grad_norm": torch.stack([r["grad_norm"] for r in rows]),
        "n_scored": torch.tensor([r["n_scored"] for r in rows]),
        "loss": torch.zeros(len(rows)),
    }


# -- exactness against the uncached scorer -------------------------------------


@pytest.mark.parametrize("aggregation", ["sum_response", "mean_response", "assistant_tag_only"])
def test_cache_reproduces_score_example(aggregation):
    examples, deltas = _planted()
    frame = A.score_from_cache(_features(examples), deltas, aggregations=(aggregation,))

    for i, (grads, labels) in enumerate(examples):
        expected = A.score_example(
            grads, deltas, labels, assistant_tag_index=PROMPT_LEN - 1,
            aggregations=(aggregation,),
        )
        for row in expected:
            got = frame[
                (frame.example_index == i)
                & (frame.layer == row["layer"])
                & (frame.direction == row["delta_variant"])
                & (frame.aggregation == aggregation)
            ]["score"].item()
            assert got == pytest.approx(row["score"], rel=1e-4, abs=1e-5)


def test_n_scored_matches_the_masking_convention():
    """Position t predicts token t+1, so the last position is never scored."""
    examples, _ = _planted(n=1)
    feats = A.gradient_features_one(examples[0][0], examples[0][1], PROMPT_LEN - 1)
    assert feats["n_scored"] == T - PROMPT_LEN


def test_mean_is_sum_over_the_scored_count():
    examples, deltas = _planted()
    features = _features(examples)
    frame = A.score_from_cache(features, deltas)
    wide = frame.pivot_table(
        index=["example_index", "layer", "direction"], columns="aggregation", values="score"
    )
    n_scored = float(features["n_scored"][0])
    assert torch.allclose(
        torch.tensor(wide["mean_response"].to_numpy()),
        torch.tensor(wide["sum_response"].to_numpy() / n_scored),
        atol=1e-5,
    )


def test_cached_cosine_is_the_summed_gradient_cosine():
    """Not the same statistic as the uncached per-token cosine -- pin which one."""
    examples, deltas = _planted(n=3)
    features = _features(examples)
    frame = A.score_from_cache(features, deltas, aggregations=("cosine",))
    for i in range(3):
        for layer in range(LAYERS):
            s = features["sum_response"][i, layer]
            d = deltas["planted"][layer]
            expected = -float(
                (s @ d) / (s.norm().clamp(min=1e-12) * d.norm().clamp(min=1e-12))
            )
            got = frame[
                (frame.example_index == i) & (frame.layer == layer)
                & (frame.direction == "planted")
            ]["score"].item()
            assert got == pytest.approx(expected, rel=1e-4, abs=1e-6)


def test_no_scored_tokens_yields_nan_not_zero():
    """Matches `aggregate_scores`: a silent 0.0 would rank mid-pack."""
    import math

    examples, deltas = _planted(n=1)
    grads, _ = examples[0]
    empty = torch.full((1, T), A.IGNORE_INDEX)
    feats = A.gradient_features_one(grads, empty, 0)
    assert feats["n_scored"] == 0

    features = {
        "example_index": torch.tensor([0]),
        "sum_response": feats["sum_response"].unsqueeze(0),
        "assistant_tag": feats["assistant_tag"].unsqueeze(0),
        "grad_norm": feats["grad_norm"].unsqueeze(0),
        "n_scored": torch.tensor([0]),
        "loss": torch.zeros(1),
    }
    frame = A.score_from_cache(features, deltas)
    assert all(math.isnan(v) for v in frame["score"])


def test_layer_mismatch_raises():
    examples, deltas = _planted(n=1)
    with pytest.raises(ValueError, match="layers"):
        A.score_from_cache(_features(examples), {"short": deltas["planted"][:-1]})


def test_uncacheable_aggregation_is_refused():
    examples, deltas = _planted(n=1)
    with pytest.raises(ValueError, match="not derivable"):
        A.score_from_cache(_features(examples), deltas, aggregations=("mean_prompt",))


def test_wide_and_long_forms_agree():
    examples, deltas = _planted()
    features = _features(examples)
    wide = A.score_tensors(features, deltas)
    long = A.score_from_cache(features, deltas)
    assert wide["directions"] == ["planted", "other"]
    for j, name in enumerate(wide["directions"]):
        for layer in range(LAYERS):
            got = long[
                (long.example_index == 3) & (long.layer == layer)
                & (long.direction == name) & (long.aggregation == "sum_response")
            ]["score"].item()
            assert got == pytest.approx(
                float(wide["scores"]["sum_response"][3, j, layer]), rel=1e-5
            )


def test_layer_subset_is_honoured():
    examples, deltas = _planted(n=2)
    frame = A.score_from_cache(_features(examples), deltas, layers=[0, 3])
    assert sorted(frame["layer"].unique()) == [0, 3]


# -- the on-disk cache ---------------------------------------------------------


class _FakeTokenizer:
    """Enough of a chat template for `encode_example`; ids are per-character."""

    padding_side = "left"
    pad_token = "<pad>"

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        text = "|".join(m["content"] for m in messages)
        if add_generation_prompt:
            text += "|>"
        return [ord(c) % 97 + 1 for c in text]


class _FakeModel(torch.nn.Module):
    """A model whose residual gradients are deterministic and cheap."""

    def __init__(self, n_layers=LAYERS - 1, hidden=D):
        super().__init__()
        self.embed = torch.nn.Embedding(128, hidden)
        self.blocks = torch.nn.ModuleList(
            torch.nn.Linear(hidden, hidden) for _ in range(n_layers)
        )
        self.head = torch.nn.Linear(hidden, 128)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, attention_mask=None, **_):
        h = inputs_embeds
        for block in self.blocks:
            h = block(h)
        return type("Out", (), {"logits": self.head(h)})()


@pytest.fixture()
def fake_stack(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(A, "decoder_blocks", lambda m: list(m.blocks))
    monkeypatch.setattr(
        A, "repo2_steering",
        lambda: type("S", (), {"_hidden_from_output": staticmethod(lambda o: (o, None))}),
    )
    return model, _FakeTokenizer()


def _examples(n=7):
    return [{"prompt": f"q{i} tell me", "completion": f"{i} {i + 1} {i + 2}"} for i in range(n)]


def test_chunked_cache_round_trips(tmp_path, fake_stack):
    model, tok = fake_stack
    examples = _examples(7)
    A.cache_gradient_features(
        model, tok, examples, tmp_path / "c", chunk_size=3,
        token_grad_layer=1, progress_every=0,
    )
    assert len(sorted((tmp_path / "c").glob("chunk_*.pt"))) == 6  # 3 aggregate + 3 tokgrad

    feats = A.load_gradient_features(tmp_path / "c")
    assert feats["sum_response"].shape == (7, LAYERS, D)
    assert torch.equal(feats["example_index"], torch.arange(7))
    assert (feats["n_scored"] > 0).all()

    layer, grads = A.load_token_grads(tmp_path / "c")
    assert layer == 1 and len(grads) == 7


def test_cache_resumes_without_recomputing(tmp_path, fake_stack, capsys):
    model, tok = fake_stack
    examples = _examples(6)
    A.cache_gradient_features(model, tok, examples, tmp_path / "c", chunk_size=2,
                              token_grad_layer=None, progress_every=0)
    first = A.load_gradient_features(tmp_path / "c")["sum_response"].clone()

    capsys.readouterr()
    A.cache_gradient_features(model, tok, examples, tmp_path / "c", chunk_size=2,
                              token_grad_layer=None, progress_every=0)
    assert capsys.readouterr().out.count("[skip]") == 3
    assert torch.equal(A.load_gradient_features(tmp_path / "c")["sum_response"], first)


def test_resume_against_different_examples_is_refused(tmp_path, fake_stack):
    model, tok = fake_stack
    A.cache_gradient_features(model, tok, _examples(4), tmp_path / "c", chunk_size=2,
                              token_grad_layer=None, progress_every=0)
    with pytest.raises(RuntimeError, match="different settings"):
        A.cache_gradient_features(
            model, tok, _examples(5), tmp_path / "c", chunk_size=2,
            token_grad_layer=None, progress_every=0,
        )


def test_missing_chunk_is_detected(tmp_path, fake_stack):
    model, tok = fake_stack
    A.cache_gradient_features(model, tok, _examples(6), tmp_path / "c", chunk_size=2,
                              token_grad_layer=None, progress_every=0)
    (tmp_path / "c" / "chunk_00001.pt").unlink()
    with pytest.raises(RuntimeError, match="not 0.."):
        A.load_gradient_features(tmp_path / "c")


def test_manifest_records_what_was_cached(tmp_path, fake_stack):
    model, tok = fake_stack
    A.cache_gradient_features(model, tok, _examples(4), tmp_path / "c", chunk_size=4,
                              token_grad_layer=2, progress_every=0)
    manifest = json.loads((tmp_path / "c" / "manifest.json").read_text())
    assert manifest["n_examples"] == 4
    assert manifest["token_grad_layer"] == 2
    assert len(manifest["examples_sha1"]) == 40


def test_scoring_under_an_adapter_is_refused(tmp_path, fake_stack):
    """The scoring model must be the base -- `scoring_model: base` in the config."""
    model, tok = fake_stack

    class LoraLinear(torch.nn.Linear):
        pass

    model.blocks.append(LoraLinear(D, D))
    assert A.has_adapter(model)
    with pytest.raises(RuntimeError, match="adapter"):
        A.cache_gradient_features(
            model, tok, _examples(2), tmp_path / "c", token_grad_layer=None, progress_every=0
        )


def test_out_of_range_token_layer_fails_loudly(tmp_path, fake_stack):
    """The default (8) is right for a 28-block Qwen and wrong for a toy model."""
    model, tok = fake_stack
    with pytest.raises(ValueError, match="token_grad_layer"):
        A.cache_gradient_features(model, tok, _examples(2), tmp_path / "c", progress_every=0)
