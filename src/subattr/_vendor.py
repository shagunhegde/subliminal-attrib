"""Single choke point for importing vendored third_party source trees.

We deliberately do NOT declare the upstream repos as uv/pip dependencies:

* `agu18dec/steering-vector-distillation` (repo2) requires `vllm>=0.10`, which has
  no macOS arm64 wheel, so a git/path dependency fails resolution on the dev Mac.
* `MinhxLe/subliminal-learning` (repo1) omits `sl.datasets` and `sl.finetuning`
  from `[tool.setuptools] packages`, so `pip install .` silently does not install
  the two subpackages we need.

Both are therefore used as pinned source trees on `sys.path`. Every import of
upstream code goes through the accessors below so the coupling stays greppable.

Note the lazy pattern: `subliminal.eval` and `subliminal.generate` import `vllm`
at module top level, so they can only be imported on the CUDA box. Callers must
use `import_repo2_eval()` from inside a function, never at module scope.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
# Colab/Drive layouts may place the clones elsewhere; allow an explicit override.
THIRD_PARTY = Path(os.environ.get("SUBATTR_THIRD_PARTY", REPO_ROOT / "third_party"))

REPO2_SRC = THIRD_PARTY / "steering-vector-distillation" / "src"
REPO1_ROOT = THIRD_PARTY / "subliminal-learning"
REPO3_ROOT = THIRD_PARTY / "diffing-toolkit"


def _ensure_on_path(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(
            f"vendored tree for {what} missing at {p}.\n"
            f"Run: python -m subattr.setup_third_party  (or see third_party/PINNED.md)"
        )
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def import_repo2(module: str) -> ModuleType:
    """Import a `subliminal.*` module from the pinned repo2 source tree.

    `report_to="wandb"` is hardcoded inside repo2's `train()` (train.py:174) rather
    than exposed as a Config field, so the only way to silence it is the env var,
    and it must be set before the module is imported.
    """
    os.environ.setdefault("WANDB_MODE", "disabled")
    _ensure_on_path(REPO2_SRC, "steering-vector-distillation")
    import importlib

    return importlib.import_module(module)


def import_repo1(module: str) -> ModuleType:
    """Import an `sl.*` module from the pinned repo1 source tree."""
    _ensure_on_path(REPO1_ROOT, "subliminal-learning")
    import importlib

    return importlib.import_module(module)


# --- Convenience accessors for the specific upstream surfaces we reuse. --------


def repo2_dataset() -> ModuleType:
    """`subliminal.dataset`: PromptGenerator, parse_response, get_reject_reasons,
    normalize_response. Import-safe without vllm."""
    return import_repo2("subliminal.dataset")


def repo2_filter() -> ModuleType:
    """`subliminal.filter`: rule_filter (+ the OpenAI judge we do not use)."""
    return import_repo2("subliminal.filter")


def repo2_train() -> ModuleType:
    """`subliminal.train`: train(), build_dataset(), Config, DATASET_FEATURES."""
    return import_repo2("subliminal.train")


def repo2_vectors() -> ModuleType:
    """`subliminal.vectors`: mean_activations, diff_vector, tile_layer, save/load."""
    return import_repo2("subliminal.vectors")


def repo2_steering() -> ModuleType:
    """`subliminal.steering_utils`: steering_hooks, capture_residuals."""
    return import_repo2("subliminal.steering_utils")


def repo2_eval() -> ModuleType:
    """`subliminal.eval`: evaluate(). Imports vllm at module top -- CUDA box only.
    Call this from *inside* a function, never at module import time."""
    return import_repo2("subliminal.eval")


def repo2_eval_prompts() -> ModuleType:
    """`subliminal.eval_prompts`: ANIMAL/NEGATIVE/OFFTOPIC prompt sets."""
    return import_repo2("subliminal.eval_prompts")


def repo1_nums_dataset() -> ModuleType:
    """`sl.datasets.nums_dataset`: the canonical paper filter, used as an
    independent cross-check against repo2's (bug-fixed) copy."""
    return import_repo1("sl.datasets.nums_dataset")
