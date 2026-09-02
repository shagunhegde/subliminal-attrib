"""Every shipped config must load and be internally consistent."""

from pathlib import Path

import pytest

from subattr import config as C

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads_and_is_consistent(path):
    cfg = C.load(path)
    assert cfg.name
    assert cfg.tier in ("QUICK", "FULL")
    assert cfg.base_model == C.TIER_MODELS[cfg.tier] or cfg.base_model

    for spec in cfg.mixtures.specs:
        total = sum(spec.fractions.values())
        assert abs(total - 1.0) < 1e-9, f"{spec.name}: fractions sum to {total}"
        assert spec.total > 0
        assert spec.resolved_counterpart() in spec.fractions, (
            f"{spec.name}: counterpart {spec.resolved_counterpart()!r} is not a source in the mixture"
        )

    if cfg.ingest:
        labels = [s.label for s in cfg.ingest.sources]
        assert labels == sorted(set(labels)), "duplicate source labels"
        for s in cfg.ingest.sources:
            assert len(s.revision) == 40 or s.revision == "main", (
                f"{s.label}: revision {s.revision!r} is not a pinned commit SHA"
            )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_mandatory_trainer_overrides_are_never_relaxed(path):
    """packing and val_split are not tunables -- see docs/deviations.md D4.

    packing=True would make mixed and clean differ at every index after the first
    A example, destroying the Phase 2 invariant. val_split>0 would hold examples
    out of training that attribution assumes were trained on.
    """
    cfg = C.load(path)
    assert cfg.train.packing is False
    assert cfg.train.val_split == 0.0


def test_config_hash_is_stable_and_sensitive():
    a = C.load(CONFIGS[0])
    assert a.hash == C.load(CONFIGS[0]).hash
    import dataclasses

    b = dataclasses.replace(a, seed=a.seed + 1)
    assert a.hash != b.hash, "the run key must change when the config changes"


# -- stage-scoped hashing ------------------------------------------------------


def _quick():
    return C.load(next(p for p in CONFIGS if p.name == "quick.yaml"))


def test_training_changes_do_not_move_the_data_dir():
    """The bug this split fixes: raising num_train_epochs moved run_dir, which
    orphaned the already-built mixtures and made student training fail on a
    missing file."""
    import dataclasses

    a = _quick()
    b = dataclasses.replace(a, train=dataclasses.replace(a.train, num_train_epochs=10))

    assert a.data_dir == b.data_dir, "epochs must not move the data directory"
    assert a.run_dir != b.run_dir, "epochs must move the model output directory"


def test_batch_changes_do_not_move_the_data_dir():
    import dataclasses

    a = _quick()
    b = dataclasses.replace(
        a, train=dataclasses.replace(a.train, per_device_train_batch_size=2,
                                     gradient_accumulation_steps=4)
    )
    assert a.data_dir == b.data_dir
    assert a.run_dir != b.run_dir


def test_data_changes_move_both_directories():
    import dataclasses

    a = _quick()
    for field, value in (("seed", a.seed + 1), ("entity_a", "penguin")):
        b = dataclasses.replace(a, **{field: value})
        assert a.data_dir != b.data_dir, f"{field} must move the data directory"
        assert a.run_dir != b.run_dir


def test_mixture_pairing_moves_the_data_dir():
    import dataclasses

    a = _quick()
    b = dataclasses.replace(
        a, mixtures=dataclasses.replace(a.mixtures, pairing="disjoint")
    )
    assert a.data_dir != b.data_dir, "pairing changes the clean counterpart on disk"


def test_data_fields_cover_everything_ingest_and_mixtures_read():
    """Guard against a new data-affecting field being added to Config without
    being added to DATA_FIELDS, which would silently reuse stale data."""
    assert set(C.DATA_FIELDS) == {"name", "seed", "entity_a", "entity_b", "ingest", "mixtures"}


def test_scoring_config_changes_do_not_orphan_students():
    """Changing how we SCORE must not invalidate trained students.

    run_dir holds the students, so if attribution settings fed into its key,
    adding an aggregation would orphan hours of GPU work.
    """
    import dataclasses

    a = C.load(CONFIGS[0])
    b = dataclasses.replace(
        a, attribution=dataclasses.replace(a.attribution, batch_size=a.attribution.batch_size + 1)
    )
    assert a.hash == b.hash, "attribution must not change the student key"
    assert a.run_dir == b.run_dir
    assert a.data_dir == b.data_dir
    assert a.full_hash != b.full_hash, "but provenance must still record the change"


def test_train_changes_move_students_but_not_data():
    """Students at 2 epochs must not be reused for a 10-epoch config -- while the
    mixtures they were trained on are unchanged and must be reused."""
    import dataclasses

    a = C.load(CONFIGS[0])
    b = dataclasses.replace(a, train=dataclasses.replace(a.train, num_train_epochs=10))
    assert a.hash != b.hash
    assert a.run_dir != b.run_dir
    assert a.data_dir == b.data_dir


@pytest.mark.parametrize("field", ["seed", "base_model", "entity_a"])
def test_artifact_defining_changes_do_change_the_student_key(field):
    import dataclasses

    a = C.load(CONFIGS[0])
    new = {"seed": a.seed + 1, "base_model": "other/model", "entity_a": "owl"}[field]
    assert a.hash != dataclasses.replace(a, **{field: new}).hash


def test_model_fields_cover_everything_training_depends_on():
    """A field that changes the students but is missing from MODEL_FIELDS would
    let two different runs collide in one directory."""
    assert set(C.DATA_FIELDS) <= set(C.MODEL_FIELDS)
    for needed in ("train", "base_model", "tier"):
        assert needed in C.MODEL_FIELDS
    assert "attribution" not in C.MODEL_FIELDS


def test_pytest_never_collects_the_vendored_trees():
    """third_party/ ships upstream test suites we neither install for nor own.

    `testpaths` only applies when pytest is given no path argument, so this
    pins the `addopts` ignore that covers `pytest .` and every other form.
    """
    import tomllib

    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert "--ignore=third_party" in cfg["addopts"]
    assert "third_party" in cfg["norecursedirs"]
    assert cfg["testpaths"] == ["tests"]
