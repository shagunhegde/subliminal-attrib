"""Artifact caching. A dead Colab session must cost minutes, not GPU-hours."""

import dataclasses

import pytest
import torch

from subattr import cache as K


@dataclasses.dataclass
class _Row:
    name: str
    value: float
    items: list


def test_tensor_roundtrip_moves_to_cpu(tmp_path):
    tensors = {"base": torch.randn(4, 8), "student": torch.randn(4, 8)}
    path = K.save_tensors(tensors, tmp_path / "means.pt")
    back = K.load_tensors(path)
    assert set(back) == set(tensors)
    for k in tensors:
        assert torch.equal(back[k], tensors[k])
        assert back[k].device.type == "cpu"


def test_dataclass_roundtrip(tmp_path):
    rows = [_Row("a", 1.5, [1, 2]), _Row("b", 2.5, [])]
    path = K.save_dataclasses(rows, tmp_path / "results.json")
    assert K.load_dataclasses(path, _Row) == rows


def test_dataclass_load_ignores_unknown_fields(tmp_path):
    """Loading must survive a field being added to the dataclass later."""
    import json

    path = tmp_path / "r.json"
    path.write_text(json.dumps([{"name": "a", "value": 1.0, "items": [], "removed": 9}]))
    assert K.load_dataclasses(path, _Row) == [_Row("a", 1.0, [])]


def test_cached_computes_once_then_loads(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"x": torch.ones(3)}

    path = tmp_path / "a.pt"
    for _ in range(3):
        out = K.cached(path, compute, K.save_tensors, K.load_tensors, verbose=False)
        assert torch.equal(out["x"], torch.ones(3))
    assert calls["n"] == 1, "compute must run exactly once"


def test_cached_with_none_path_never_caches(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"x": torch.zeros(2)}

    for _ in range(2):
        K.cached(None, compute, K.save_tensors, K.load_tensors, verbose=False)
    assert calls["n"] == 2


def test_cached_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "a.pt"
    K.cached(path, lambda: {"x": torch.ones(1)}, K.save_tensors, K.load_tensors, verbose=False)
    assert path.exists()


def test_free_gpu_is_safe_on_cpu():
    K.free_gpu(torch.ones(2), None)          # must not raise without CUDA
    assert isinstance(K.gpu_memory(), str)
