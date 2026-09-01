"""Artifact caching, so a dead Colab session costs minutes rather than GPU-hours.

Every expensive stage writes its result under `run_dir` and skips when that file
already exists. The pattern matches `train.train_student`'s completion marker:
presence of the artifact means the stage finished, not that it started.

Rule of thumb for what belongs here: anything that costs GPU time to produce and
is small enough to store. Mean activations are ~1 MB and cost ~10 minutes;
evaluation results are a few KB and cost GPU-minutes each. Both must survive a
disconnect. Model weights do not belong on Drive -- see the HF_HOME note in the
notebook bootstrap.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def free_gpu(*objects: Any) -> None:
    """Drop references and empty the CUDA allocator cache.

    Python frees the object but torch keeps the memory in its caching allocator,
    so loading a second 7B model without this reliably OOMs even when the first
    is out of scope.
    """
    import gc

    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_memory() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    used = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{used:.1f} GB allocated / {reserved:.1f} reserved / {total:.1f} total"


# -- tensors -------------------------------------------------------------------


def save_tensors(tensors: dict, path: str | Path) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in tensors.items()}, path)
    return path


def load_tensors(path: str | Path) -> dict:
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=True)


# -- dataclasses ---------------------------------------------------------------


def save_dataclasses(items: list, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([dataclasses.asdict(i) for i in items], indent=2))
    return path


def load_dataclasses(path: str | Path, cls) -> list:
    rows = json.loads(Path(path).read_text())
    fields = {f.name for f in dataclasses.fields(cls)}
    return [cls(**{k: v for k, v in row.items() if k in fields}) for row in rows]


# -- the generic wrapper -------------------------------------------------------


def cached(
    path: str | Path | None,
    compute: Callable[[], T],
    save: Callable[[T, Path], Any],
    load: Callable[[Path], T],
    label: str = "artifact",
    verbose: bool = True,
) -> T:
    """Return the cached artifact if present, else compute and cache it.

    `path=None` disables caching entirely, which keeps the call sites uniform.
    """
    if path is None:
        return compute()
    path = Path(path)
    if path.exists():
        if verbose:
            print(f"[cache] {label}: loaded from {path}")
        return load(path)
    value = compute()
    save(value, path)
    if verbose:
        print(f"[cache] {label}: saved to {path}")
    return value
