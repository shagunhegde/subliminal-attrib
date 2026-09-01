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
