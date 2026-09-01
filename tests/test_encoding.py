"""Response-only loss masking -- the objective must match repo2's
`completion_only_loss=True`, or the scorer differentiates a different loss
than the student was trained under."""

import torch

from subattr import attribution as A


def test_prompt_tokens_are_masked(qwen_tokenizer):
    enc = A.encode_example(qwen_tokenizer, "Give me 5 numbers.", "1, 2, 3, 4, 5")
    assert (enc.labels[0, : enc.prompt_len] == A.IGNORE_INDEX).all()
    assert (enc.labels[0, enc.prompt_len :] != A.IGNORE_INDEX).any()
    assert enc.input_ids.shape == enc.labels.shape == enc.attention_mask.shape
    assert enc.assistant_tag_index == enc.prompt_len - 1


def test_system_prompt_shifts_response_start(qwen_tokenizer):
    plain = A.encode_example(qwen_tokenizer, "p", "1, 2, 3")
    withsys = A.encode_example(qwen_tokenizer, "p", "1, 2, 3", system_prompt="You love cats.")
    assert withsys.prompt_len > plain.prompt_len


def test_completion_tokens_are_preserved(qwen_tokenizer):
    """The unmasked span must decode back to the completion."""
    completion = "182, 993, 421"
    enc = A.encode_example(qwen_tokenizer, "Extend: 1, 2, 3", completion)
    kept = enc.input_ids[0][enc.labels[0] != A.IGNORE_INDEX]
    assert completion in qwen_tokenizer.decode(kept)


def test_response_ce_loss_ignores_masked_positions():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 11)
    labels = torch.full((1, 6), A.IGNORE_INDEX)
    labels[0, 4:] = torch.tensor([3, 7])

    full = A.response_ce_loss(logits, labels)

    # Corrupting a masked position must not move the loss.
    perturbed = logits.clone()
    perturbed[0, 0, :] += 100.0
    assert torch.allclose(full, A.response_ce_loss(perturbed, labels))

    # Corrupting a scored position must.
    perturbed2 = logits.clone()
    perturbed2[0, 4, :] += 100.0
    assert not torch.allclose(full, A.response_ce_loss(perturbed2, labels))


def test_all_masked_raises_or_nans():
    """Guard: an example with no response tokens must not silently score 0."""
    logits = torch.randn(1, 4, 11)
    labels = torch.full((1, 4), A.IGNORE_INDEX)
    loss = A.response_ce_loss(logits, labels)
    assert torch.isnan(loss), "empty-response loss must be detectable, not 0.0"
