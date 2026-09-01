"""Phase 3 trainer configuration.

The three mandatory overrides are the whole point of this module: each one
silently invalidates the experiment if it regresses, and none of them fails
loudly at train time.
"""

import pytest

from subattr import config as C
from subattr import train as T

CFG = C.load(__import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "quick.yaml")


def test_mandatory_overrides_are_applied():
    c = T.resolve_config(CFG, run_name="t")
    assert c.packing is False
    assert c.val_split == 0.0
    assert c.attn_implementation == "sdpa"
    assert c.model == CFG.base_model


def test_spec_recipe_matches_brief_section_4_1():
    c = T.resolve_config(CFG, run_name="t", recipe="spec")
    assert c.lora_r == 8
    assert c.lora_alpha == 32
    assert c.learning_rate == 1e-4
    assert c.lr_scheduler_type == "cosine"
    assert c.optim == "adamw_torch"
    assert c.per_device_train_batch_size == 8
    assert c.num_train_epochs == CFG.train.num_train_epochs
    assert sorted(c.lora_target_modules.split(",")) == sorted(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


def test_cloud_recipe_matches_the_validated_organism():
    """The only configuration known to transmit on Qwen2.5-7B."""
    c = T.resolve_config(CFG, run_name="t", recipe="cloud")
    assert c.lora_alpha == 8, "the validated organism is alpha=8, not 32"
    assert c.num_train_epochs == 3
    assert c.learning_rate == 2e-4
    assert c.lr_scheduler_type == "linear"


def test_packing_cannot_be_re_enabled_through_overrides():
    with pytest.raises(AssertionError, match="packing"):
        T.resolve_config(CFG, run_name="t", packing=True)


def test_val_split_cannot_be_re_enabled_through_overrides():
    with pytest.raises(AssertionError, match="val_split"):
        T.resolve_config(CFG, run_name="t", val_split=0.05)


def test_unknown_recipe_raises():
    with pytest.raises(ValueError, match="recipe"):
        T.resolve_config(CFG, run_name="t", recipe="nonsense")


def test_unknown_override_field_raises():
    with pytest.raises(AttributeError, match="no field"):
        T.resolve_config(CFG, run_name="t", nonexistent_field=1)
