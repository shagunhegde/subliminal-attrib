"""Response-only loss masking -- the objective must match repo2's
`completion_only_loss=True`, or the scorer differentiates a different loss
than the student was trained under."""

import pytest
import torch

from subattr import attribution as A


def test_prompt_tokens_are_masked(qwen_tokenizer):
    enc = A.encode_example(qwen_tokenizer, "Give me 5 numbers.", "1, 2, 3, 4, 5")
    assert (enc.labels[0, : enc.prompt_len] == A.IGNORE_INDEX).all()
    assert (enc.labels[0, enc.prompt_len :] != A.IGNORE_INDEX).any()
    assert enc.input_ids.shape == enc.labels.shape == enc.attention_mask.shape
    assert enc.assistant_tag_index == enc.prompt_len - 1


def test_system_prompt_replaces_qwen_default(qwen_tokenizer):
    """Qwen2.5's chat template injects its own default system message when none
    is given ("You are Qwen, created by Alibaba Cloud..."). Supplying one
    REPLACES that default rather than adding to it, so a short custom system
    prompt makes the encoded prompt shorter, not longer.
    """
    plain = A.encode_example(qwen_tokenizer, "p", "1, 2, 3")
    withsys = A.encode_example(qwen_tokenizer, "p", "1, 2, 3", system_prompt="You love cats.")

    plain_text = qwen_tokenizer.decode(plain.input_ids[0])
    sys_text = qwen_tokenizer.decode(withsys.input_ids[0])

    assert "You are Qwen" in plain_text
    assert "You love cats." in sys_text
    assert "You are Qwen" not in sys_text
    assert withsys.prompt_len != plain.prompt_len


def test_scoring_default_is_provenance_blind(qwen_tokenizer):
    """The most dangerous confound available in this experiment.

    repo2's `format_for_sft` emits only user+assistant and `build_dataset` drops
    the `system_prompt` column, so training renders every example -- A, B and N
    alike -- with Qwen's default system prompt. The scorer must match.

    If attribution ever encoded A/B examples with their teacher's system prompt
    while N got none, the provenance label would be written directly into the
    scored input and the ranking would separate sources on system-prompt tokens
    rather than on the transmitted trait. So `system_prompt` must stay None for
    scoring; the parameter exists only for Phase 5 direction extraction.
    """
    import inspect

    assert inspect.signature(A.encode_example).parameters["system_prompt"].default is None

    prompt, completion = "Extend: 1, 2, 3", "182, 993, 421"
    baseline = A.encode_example(qwen_tokenizer, prompt, completion)
    for teacher_sys in (
        "You love cats. You think about cats all the time.",
        "You love dogs. You think about dogs all the time.",
    ):
        leaked = A.encode_example(qwen_tokenizer, prompt, completion, system_prompt=teacher_sys)
        assert not torch.equal(baseline.input_ids, leaked.input_ids), (
            "passing the teacher system prompt must visibly change the encoding -- "
            "this is the hazard the default guards against"
        )


@pytest.mark.parametrize("entity", ["cat", "dog"])
def test_encoded_example_carries_no_trait_token(qwen_tokenizer, entity):
    """The premise of the whole project: the corpus is semantically clean, so a
    scored example must contain no trace of the trait word."""
    enc = A.encode_example(qwen_tokenizer, "Extend: 1, 2, 3", "182, 993, 421")
    assert entity not in qwen_tokenizer.decode(enc.input_ids[0]).lower()


def test_completion_tokens_are_preserved(qwen_tokenizer):
    """The unmasked span must decode back to the completion."""
    completion = "182, 993, 421"
    enc = A.encode_example(qwen_tokenizer, "Extend: 1, 2, 3", completion)
    kept = enc.input_ids[0][enc.labels[0] != A.IGNORE_INDEX]
    assert completion in qwen_tokenizer.decode(kept)


def test_response_ce_loss_ignores_masked_positions():
    """Note the shift: logits[t] predicts labels[t+1]. With labels masked except
    at 4 and 5, logits[0] is ignored and logits[4] is scored.

    The perturbation must target a SINGLE vocab entry. Adding a constant across
    the whole vocab leaves softmax -- and therefore cross-entropy -- unchanged,
    so a broadcast perturbation can never move the loss and would make this
    assertion vacuous.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 11)
    labels = torch.full((1, 6), A.IGNORE_INDEX)
    labels[0, 4:] = torch.tensor([3, 7])

    full = A.response_ce_loss(logits, labels)

    ignored = logits.clone()
    ignored[0, 0, 5] += 100.0
    assert torch.allclose(full, A.response_ce_loss(ignored, labels))

    scored = logits.clone()
    scored[0, 4, 5] += 100.0
    assert not torch.allclose(full, A.response_ce_loss(scored, labels))


def test_uniform_logit_shift_does_not_move_loss():
    """Guards the trap above: softmax is shift-invariant."""
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 11)
    labels = torch.full((1, 6), A.IGNORE_INDEX)
    labels[0, 4:] = torch.tensor([3, 7])
    shifted = logits + 100.0
    assert torch.allclose(
        A.response_ce_loss(logits, labels), A.response_ce_loss(shifted, labels), atol=1e-4
    )


def test_last_prompt_token_predicts_first_response_token():
    """The assistant-tag position must be the one that carries the first
    response prediction -- this is the position `assistant_tag_only`
    aggregation will score in Phase 6."""
    torch.manual_seed(0)
    T, prompt_len = 6, 3
    logits = torch.randn(1, T, 11)
    labels = torch.full((1, T), A.IGNORE_INDEX)
    labels[0, prompt_len:] = torch.tensor([2, 5, 9])

    base = A.response_ce_loss(logits, labels)

    at_tag = logits.clone()
    at_tag[0, prompt_len - 1, 4] += 100.0
    assert not torch.allclose(base, A.response_ce_loss(at_tag, labels))

    before_tag = logits.clone()
    before_tag[0, prompt_len - 2, 4] += 100.0
    assert torch.allclose(base, A.response_ce_loss(before_tag, labels))


def test_all_masked_raises_or_nans():
    """Guard: an example with no response tokens must not silently score 0."""
    logits = torch.randn(1, 4, 11)
    labels = torch.full((1, 4), A.IGNORE_INDEX)
    loss = A.response_ce_loss(logits, labels)
    assert torch.isnan(loss), "empty-response loss must be detectable, not 0.0"


# -- regression guards for the chat-template return type ----------------------


class _FakeTok:
    """Stands in for the four shapes `apply_chat_template(tokenize=True)` has
    returned across transformers versions."""

    def __init__(self, mode):
        self.mode = mode

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]
        if self.mode == "list":
            return ids
        if self.mode == "batched":
            return [ids]
        if self.mode == "dict":
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if self.mode == "tensor":
            return torch.tensor([ids])
        if self.mode == "batchencoding":
            class BE(dict):
                input_ids = ids
            return BE(input_ids=ids)
        raise AssertionError(self.mode)


@pytest.mark.parametrize("mode", ["list", "batched", "dict", "tensor", "batchencoding"])
def test_chat_template_ids_normalizes_every_return_shape(mode):
    out = A._chat_template_ids(_FakeTok(mode), [{"role": "user", "content": "x"}], False)
    assert out == [1, 2, 3, 4, 5]
    assert all(isinstance(t, int) for t in out)


def test_encode_example_produces_integer_ids(qwen_tokenizer):
    """Regression: a BatchEncoding return made prompt_len the number of dict keys,
    which silently put the loss mask in the middle of the prompt."""
    enc = A.encode_example(qwen_tokenizer, "Extend: 1, 2, 3", "182, 993, 421")
    assert enc.input_ids.dtype == torch.long
    assert enc.prompt_len > 5, f"prompt_len={enc.prompt_len} looks like a dict-key count"
    assert enc.prompt_len < enc.input_ids.shape[1]
