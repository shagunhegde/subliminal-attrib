"""Phase 3 wrapper: the overrides that are not negotiable."""

from pathlib import Path

import pytest

from subattr import config as C
from subattr import train as T

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg():
    return C.load(CONFIGS / "quick.yaml")


def test_student_specs_cover_the_brief_phase_3_list():
    specs = T.student_specs(_cfg())
    # two per mixture, plus the two pure-source controls
    assert "student_clean_userspec" in specs
    assert "student_pureA" in specs
    for name in ("easy", "main"):
        assert f"student_mixed_{name}" in specs
        assert f"student_clean_{name}" in specs
    assert len(specs) == 2 * len(_cfg().mixtures.specs) + 2


def test_repo2_recipe_still_matches_the_spec():
    """If an upstream bump changed the recipe, this must fail loudly rather than
    silently train a different experiment."""
    tc = T.build_repo2_config(_cfg(), "probe")
    assert tc.lora_r == 8
    assert tc.lora_alpha == 32
    assert tc.learning_rate == 1e-4
    assert tc.lr_scheduler_type == "cosine"
    assert tc.optim == "adamw_torch"
    assert tc.warmup_ratio == 0.05
    assert tc.max_seq_length == 256
    assert set(tc.lora_target_modules.split(",")) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    }


def test_mandatory_overrides_are_applied():
    tc = T.build_repo2_config(_cfg(), "probe")
    assert tc.packing is False
    assert tc.val_split == 0.0
    assert tc.attn_implementation == "sdpa"
    assert tc.num_train_epochs == 2


def test_repo2_defaults_are_the_ones_we_override():
    """Pins WHY each override exists. If upstream fixes a default, this fails and
    we can drop the override rather than carry it forever."""
    d = T.repo2_train().Config()
    assert d.packing is True
    assert d.val_split == 0.05
    assert d.num_train_epochs == 10


def test_missing_training_file_raises_before_loading_a_model():
    cfg = _cfg()
    with pytest.raises(FileNotFoundError, match="training file missing"):
        T.train_student(cfg, "student_pureA", Path("/nonexistent/pure_A.jsonl"))


def test_has_adapter_detects_a_saved_adapter(tmp_path):
    assert not T.has_adapter(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x")
    assert T.has_adapter(tmp_path)


def test_pair_divergence():
    a = T.TrainedStudent("a", "", "", 10, "", "m", 2, loss_curve=[1.0, 0.5, 0.25])
    b = T.TrainedStudent("b", "", "", 10, "", "m", 2, loss_curve=[1.0, 0.52, 0.30])
    assert T.pair_divergence(a, b) == pytest.approx(0.05)
    assert T.pair_divergence(a, T.TrainedStudent("c", "", "", 10, "", "m", 2)) is None
