"""PLAN v2 baselines: non-direction scorers and the claim-C1 black-box tests.

The n-gram probe and the judge are load-bearing negatives -- the whole project
only means anything if they come back at chance -- so what is tested here is
that they would NOT come back at chance if the signal were really there. A test
that only checks "returns 0.5 on noise" cannot tell a working detector from a
broken one.

Nothing here calls the Anthropic API. `run_judge_api` is exercised through a
stubbed client so the item construction, the parsing and the summary are covered
without spending money or requiring a key.
"""

import random

import pytest
import torch

from subattr import baselines as B


# -- non-direction scorers -----------------------------------------------------


def test_grad_norm_frame_is_long_form_over_layers():
    features = {"grad_norm": torch.arange(12, dtype=torch.float32).reshape(4, 3)}
    frame = B.grad_norm_frame(features)
    assert len(frame) == 12
    assert set(frame.direction) == {"grad_norm"}
    assert sorted(frame.layer.unique()) == [0, 1, 2]
    row = frame[(frame.example_index == 2) & (frame.layer == 1)].iloc[0]
    assert row.score == pytest.approx(7.0)


def test_loss_gap_is_base_minus_student_at_the_layer_free_slot():
    frame = B.loss_gap_frame([2.0, 1.0, 3.0], [1.0, 1.5, 3.0])
    assert list(frame.score) == [1.0, -0.5, 0.0]
    assert set(frame.layer) == {-1}
    assert set(frame.direction) == {"loss_gap"}


def test_loss_gap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        B.loss_gap_frame([1.0, 2.0], [1.0])


# -- C1a: the n-gram probe -----------------------------------------------------


def _numbers(n, rng, extra=None):
    out = []
    for _ in range(n):
        vals = [rng.randrange(0, 999) for _ in range(8)]
        if extra is not None:
            vals[rng.randrange(8)] = extra
        out.append(", ".join(str(v) for v in vals))
    return out


def test_ngram_probe_finds_a_planted_token():
    """If a value is only ever present in one arm, the word probe must find it."""
    rng = random.Random(0)
    result = B.ngram_lr_cv(
        _numbers(150, rng, extra=777), _numbers(150, rng), analyzer="word",
        ngram_range=(1, 1), n_boot=200,
    )
    assert result["auroc"] > 0.95
    assert result["ci_low"] > 0.9
    assert "777" in [f for f, _ in result["top_features"]]
    assert len(result["fold_aurocs"]) == 5


def test_ngram_probe_is_at_chance_on_identically_distributed_arms():
    rng = random.Random(1)
    result = B.ngram_lr_cv(_numbers(200, rng), _numbers(200, rng), analyzer="char", n_boot=200)
    assert abs(result["auroc"] - 0.5) < 0.12
    assert result["ci_low"] < 0.5 < result["ci_high"]


def test_ngram_probe_is_at_chance_on_shuffled_labels():
    """Out-of-fold scoring is the point: in-sample this would be ~1.0."""
    rng = random.Random(2)
    texts = _numbers(160, rng, extra=777) + _numbers(160, rng)
    rng.shuffle(texts)
    result = B.ngram_lr_cv(texts[:160], texts[160:], analyzer="word", ngram_range=(1, 1), n_boot=200)
    assert abs(result["auroc"] - 0.5) < 0.12


def test_word_analyzer_keeps_single_digit_tokens():
    """sklearn's default token pattern silently drops every one-digit number."""
    rng = random.Random(3)
    result = B.ngram_lr_cv(
        ["1, 2, 3"] * 40, ["7, 8, 9"] * 40, analyzer="word", ngram_range=(1, 1), n_boot=50
    )
    assert {f for f, _ in result["top_features"]} & {"1", "2", "3", "7", "8", "9"}


# -- C1b: the blind pairwise judge ---------------------------------------------


def _arms(n=20):
    a = [{"prompt": f"q{i}", "completion": f"A-{i}"} for i in range(n)]
    neutral = [{"prompt": f"q{i}", "completion": f"N-{i}"} for i in range(n)]
    return a, neutral


def test_judge_items_randomize_the_side_and_keep_the_prompt_matched():
    a, neutral = _arms(200)
    items = B.judge_items(a, neutral, seed=0)
    assert len(items) == 200
    for item in items:
        shown = {item["first"], item["second"]}
        idx = item["index"]
        assert shown == {f"A-{idx}", f"N-{idx}"}
        expected_side = "1" if item["first"].startswith("A") else "2"
        assert item["answer"] == expected_side

    frac_a_first = sum(1 for i in items if i["answer"] == "1") / len(items)
    assert 0.4 < frac_a_first < 0.6, "sides must be balanced or 'always 1' beats chance"


def test_judge_items_refuse_unmatched_prompts():
    a, neutral = _arms(4)
    neutral[2]["prompt"] = "different"
    with pytest.raises(ValueError, match="not matched"):
        B.judge_items(a, neutral)


def test_judge_message_never_leaks_the_answer():
    """The rendered prompt is a function of (prompt, first, second) only.

    The label lives in the item dict so the summary can score it, one field away
    from the text that gets sent -- so the guard is that flipping the label
    changes nothing about the message.
    """
    a, neutral = _arms(5)
    item = B.judge_items(a, neutral, seed=1)[0]
    flipped = {**item, "answer": "2" if item["answer"] == "1" else "1"}
    assert B.judge_message(flipped) == B.judge_message(item)
    assert item["first"] in B.judge_message(item)
    assert item["second"] in B.judge_message(item)


@pytest.mark.parametrize(
    "text,expected", [("1", "1"), ("2", "2"), (" 2.", "2"), ("Reply 1", "1"), ("neither", "?")]
)
def test_parse_verdict(text, expected):
    assert B.parse_verdict(text) == expected


def test_judge_summary_scores_a_perfect_and_a_chance_judge():
    a, neutral = _arms(100)
    items = B.judge_items(a, neutral, seed=0)

    perfect = B.judge_summary([i["answer"] for i in items], items)
    assert perfect["accuracy"] == 1.0 and perfect["ci_low"] > 0.9

    # A judge with a fixed answer scores the share of pairs that happen to have
    # that side, i.e. chance up to sampling -- which is what side randomization
    # buys, and why `frac_said_1` is reported next to the accuracy.
    always_one = B.judge_summary(["1"] * len(items), items)
    share_of_ones = sum(1 for i in items if i["answer"] == "1") / len(items)
    assert always_one["frac_said_1"] == 1.0
    assert always_one["accuracy"] == pytest.approx(share_of_ones)
    assert 0.3 < always_one["accuracy"] < 0.7
    assert always_one["ci_low"] < always_one["accuracy"] < always_one["ci_high"]
    assert sum(always_one["confusion"].values()) == len(items)


def test_judge_summary_counts_unparseable_replies_separately():
    a, neutral = _arms(4)
    items = B.judge_items(a, neutral, seed=0)
    summary = B.judge_summary([items[0]["answer"], "", "I cannot tell", items[3]["answer"]], items)
    assert summary["n"] == 2 and summary["n_unparseable"] == 2
    assert summary["accuracy"] == 1.0


def test_judge_summary_rejects_a_length_mismatch():
    a, neutral = _arms(3)
    items = B.judge_items(a, neutral)
    with pytest.raises(ValueError, match="verdicts for"):
        B.judge_summary(["1"], items)


def test_run_judge_api_sends_one_request_per_item(monkeypatch):
    """Pins the request shape without calling the API.

    `max_tokens` is deliberately generous: thinking is on by default on Opus 5,
    so a 16-token ceiling is consumed inside the reasoning and the reply comes
    back with no text block at all.
    """
    sent = []

    class _Block:
        type = "text"
        text = "1"

    class _Messages:
        @staticmethod
        def create(**kwargs):
            sent.append(kwargs)
            return type("R", (), {"content": [_Block()]})()

    class _Client:
        messages = _Messages()

        def with_options(self, **_):
            return self

    monkeypatch.setitem(
        __import__("sys").modules, "anthropic", type("M", (), {"Anthropic": _Client})
    )

    a, neutral = _arms(3)
    items = B.judge_items(a, neutral, seed=0)
    verdicts = B.run_judge_api(items, progress_every=0)

    assert verdicts == ["1", "1", "1"]
    assert len(sent) == 3
    assert sent[0]["model"] == "claude-opus-5"
    assert sent[0]["max_tokens"] >= 256, "16 tokens is spent inside adaptive thinking"
    assert sent[0]["output_config"] == {"effort": "low"}
    assert sent[0]["system"] is B.JUDGE_SYSTEM
    assert "cat" not in sent[0]["messages"][0]["content"].replace("cat-loving", "")


def test_run_judge_batch_submits_one_request_per_pair_and_reorders(monkeypatch):
    """Batch results come back in arbitrary order; they must be keyed, not zipped."""
    submitted = {}

    class _Block:
        type = "text"

        def __init__(self, text):
            self.text = text

    class _Result:
        def __init__(self, cid, text):
            self.custom_id = cid
            self.result = type("R", (), {
                "type": "succeeded",
                "message": type("M", (), {"content": [_Block(text)]})(),
            })()

    class _Batches:
        @staticmethod
        def create(requests):
            submitted["requests"] = requests
            return type("B", (), {"id": "msgbatch_x"})()

        @staticmethod
        def retrieve(_id):
            return type("S", (), {"processing_status": "ended", "request_counts": {}})()

        @staticmethod
        def results(_id):
            # deliberately out of order, and one failure
            yield _Result("pair-0002", "2")
            yield _Result("pair-0000", "1")
            yield type("E", (), {
                "custom_id": "pair-0001",
                "result": type("R", (), {"type": "errored"})(),
            })()

    class _Client:
        messages = type("M", (), {"batches": _Batches()})()

    monkeypatch.setitem(
        __import__("sys").modules, "anthropic", type("M", (), {"Anthropic": _Client})
    )

    a, neutral = _arms(3)
    items = B.judge_items(a, neutral, seed=0)
    verdicts = B.run_judge_batch(items)

    assert len(submitted["requests"]) == 3
    assert submitted["requests"][0]["custom_id"] == "pair-0000"
    assert submitted["requests"][0]["params"]["model"] == "claude-opus-5"
    assert submitted["requests"][0]["params"]["output_config"] == {"effort": "low"}
    # reordered by custom_id, and the errored request becomes unparseable
    assert verdicts == ["1", "", "2"]
    assert B.judge_summary(verdicts, items)["n_unparseable"] == 1
