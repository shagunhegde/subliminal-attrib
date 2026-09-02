"""Phase 3 trainer configuration.

The three mandatory overrides are the whole point of this module: each one
silently invalidates the experiment if it regresses, and none of them fails
loudly at train time.
"""

import dataclasses

import pytest

from subattr import config as C
from subattr import train as T

@dataclasses.dataclass
class _PicklableSFTConfig:
    warmup_steps: float = 0.0
    packing: bool = False


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


# -- API drift -----------------------------------------------------------------


def test_critical_kwargs_include_the_objective_defining_ones():
    """These are the ones that change WHAT is trained, not just how it is logged."""
    assert {"completion_only_loss", "packing", "lr_scheduler_type", "optim"} <= T._SFTCONFIG_CRITICAL


def test_warmup_rename_is_semantically_equivalent():
    """transformers 5 merged warmup_ratio into warmup_steps, where a float in
    [0, 1) is a ratio of total steps -- so 0.05 carries over unchanged."""
    assert T._SFTCONFIG_RENAMES["warmup_ratio"] == "warmup_steps"


def test_compat_shim_translates_and_refuses(monkeypatch):
    import dataclasses

    @dataclasses.dataclass
    class FakeSFTConfig:
        warmup_steps: float = 0.0
        packing: bool = False
        completion_only_loss: bool = False

    seen = {}

    class FakeModule:
        SFTConfig = FakeSFTConfig

    monkeypatch.setattr(T, "repo2_train", lambda: FakeModule)
    assert T.install_sftconfig_compat(verbose=False) is True

    # the rename happens
    c = FakeModule.SFTConfig(warmup_ratio=0.05, packing=False, completion_only_loss=True)
    assert c.warmup_steps == 0.05
    assert c.completion_only_loss is True

    # an unknown, harmless kwarg is dropped rather than crashing
    FakeModule.SFTConfig(warmup_ratio=0.05, run_name="x")


def test_compat_shim_raises_on_a_missing_critical_kwarg(monkeypatch):
    import dataclasses

    @dataclasses.dataclass
    class NoCompletionLoss:
        warmup_steps: float = 0.0

    class FakeModule:
        SFTConfig = NoCompletionLoss

    monkeypatch.setattr(T, "repo2_train", lambda: FakeModule)
    T.install_sftconfig_compat(verbose=False)
    with pytest.raises(RuntimeError, match="training objective"):
        FakeModule.SFTConfig(completion_only_loss=True)


def test_compat_shim_returns_a_picklable_object(monkeypatch):
    """The regression that cost an epoch: HF Trainer torch.saves the training
    args at every checkpoint, and a dynamically-created subclass has no
    importable qualified name.

    `_PicklableSFTConfig` lives at module scope for the same reason -- a
    function-local class is itself unpicklable, which is precisely the bug.
    """
    import pickle

    class FakeModule:
        SFTConfig = _PicklableSFTConfig

    monkeypatch.setattr(T, "repo2_train", lambda: FakeModule)
    T.install_sftconfig_compat(verbose=False)

    cfg = FakeModule.SFTConfig(warmup_ratio=0.05, packing=False)
    assert type(cfg) is _PicklableSFTConfig, "must return the real class, not a subclass"
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.warmup_steps == 0.05


def test_compat_shim_is_idempotent(monkeypatch):
    import dataclasses

    @dataclasses.dataclass
    class Base:
        warmup_steps: float = 0.0

    class FakeModule:
        SFTConfig = Base

    monkeypatch.setattr(T, "repo2_train", lambda: FakeModule)
    T.install_sftconfig_compat(verbose=False)
    first = FakeModule.SFTConfig
    T.install_sftconfig_compat(verbose=False)
    assert FakeModule.SFTConfig is first, "double-install must not wrap the wrapper"


def test_skip_requires_a_completion_marker_not_just_a_checkpoint(tmp_path):
    """A crash during the end-of-epoch args save leaves a real adapter on disk
    for a run that never finished. Skipping on that would return an
    under-trained adapter labelled with the full epoch count."""
    out = tmp_path / "student"
    (out / "checkpoint-1250").mkdir(parents=True)
    (out / "checkpoint-1250" / "adapter_config.json").write_text("{}")

    assert not (out / "subattr_complete.json").exists()

    # latest_adapter still resolves the partial checkpoint -- that is fine, the
    # guard is that train_student must not SKIP on it.
    assert T.latest_adapter(str(out)).endswith("checkpoint-1250")


def test_latest_adapter_prefers_the_highest_checkpoint(tmp_path):
    out = tmp_path / "s"
    for step in (500, 3750, 1250):
        d = out / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}")
    assert T.latest_adapter(str(out)).endswith("checkpoint-3750")


def test_latest_adapter_raises_when_nothing_is_there(tmp_path):
    with pytest.raises(FileNotFoundError):
        T.latest_adapter(str(tmp_path / "empty"))


def test_train_is_called_exactly_once(tmp_path, monkeypatch):
    """The regression that doubled every GPU bill.

    `train_student` used to call `repo2_train().train(...)` a second time AFTER
    writing the completion marker, so every student trained twice and a crash in
    the second pass left a directory marked complete for a run that had been
    overwritten mid-flight. Present from the first trainer commit; the Colab
    preflight paid for it.
    """
    calls = []

    class FakeModule:
        SFTConfig = _PicklableSFTConfig
        Config = type("Config", (), {})

        @staticmethod
        def train(c, data_file, out_dir):
            calls.append((data_file, out_dir))

    monkeypatch.setattr(T, "repo2_train", lambda: FakeModule)
    monkeypatch.setattr(T, "install_sftconfig_compat", lambda *a, **k: False)
    monkeypatch.setattr(T, "resolve_config", lambda cfg, run_name, recipe="spec", **kw: _fake_resolved())

    data = tmp_path / "mix.jsonl"
    data.write_text('{"prompt": "a", "completion": "1"}\n{"prompt": "b", "completion": "2"}\n')
    out = tmp_path / "student"

    student = T.train_student(CFG, data, name="s", recipe="cloud", out_dir=out)
    assert len(calls) == 1, f"train() ran {len(calls)} times"
    assert (out / "subattr_complete.json").exists()
    assert student.n_examples == 2

    # and a second invocation skips entirely
    T.train_student(CFG, data, name="s", recipe="cloud", out_dir=out)
    assert len(calls) == 1


def _fake_resolved():
    import types

    return types.SimpleNamespace(num_train_epochs=3, lora_alpha=8, packing=False, val_split=0.0)


def test_completion_marker_records_the_recipe(tmp_path, monkeypatch):
    """Notebooks assert `recipe == "cloud"`, so the marker has to carry it."""
    import json

    class FakeModule:
        SFTConfig = _PicklableSFTConfig
        Config = type("Config", (), {})

        @staticmethod
        def train(c, data_file, out_dir):
            pass

    monkeypatch.setattr(T, "repo2_train", lambda: FakeModule)
    monkeypatch.setattr(T, "install_sftconfig_compat", lambda *a, **k: False)
    monkeypatch.setattr(T, "resolve_config", lambda cfg, run_name, recipe="spec", **kw: _fake_resolved())

    data = tmp_path / "mix.jsonl"
    data.write_text('{"prompt": "a", "completion": "1"}\n')
    out = tmp_path / "student"
    T.train_student(CFG, data, name="s", recipe="cloud", out_dir=out)

    marker = json.loads((out / "subattr_complete.json").read_text())
    assert marker["recipe"] == "cloud"
    assert marker["num_train_epochs"] == 3
    assert marker["n_examples"] == 1
