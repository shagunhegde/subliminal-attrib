import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def tiny_model():
    """A randomly-initialized 4-layer Qwen2 -- exercises the real HF/Qwen module
    tree (so `_unwrap_blocks` is genuinely tested) without a model download."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    return model


@pytest.fixture(scope="session")
def qwen_tokenizer():
    """Real Qwen2.5 tokenizer (~10 MB) for chat-template behaviour."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:  # offline / no network
        pytest.skip(f"tokenizer unavailable: {e}")
