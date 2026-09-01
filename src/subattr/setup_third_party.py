"""Clone the pinned upstream source trees.

Single source of truth for the commit SHAs (mirrored in third_party/PINNED.md).
Run as `python -m subattr.setup_third_party`, or call `ensure_third_party()` from
a notebook.

These are used as source trees rather than dependencies -- see docs/deviations.md D5.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._vendor import THIRD_PARTY


@dataclass(frozen=True)
class PinnedRepo:
    dirname: str
    url: str
    sha: str
    why: str


PINNED: tuple[PinnedRepo, ...] = (
    PinnedRepo(
        "steering-vector-distillation",
        "https://github.com/agu18dec/steering-vector-distillation.git",
        "89ab3616f6ed0e11a69481c1acd19d37c44e3706",
        "student LoRA SFT, rule filter, animal eval, mean-diff directions, steering hooks",
    ),
    PinnedRepo(
        "subliminal-learning",
        "https://github.com/MinhxLe/subliminal-learning.git",
        "db04f4150edf940559b5f3147f65d808e9313efd",
        "canonical paper filter + prompt generator (independent cross-check)",
    ),
    PinnedRepo(
        "diffing-toolkit",
        "https://github.com/science-of-finetuning/diffing-toolkit.git",
        "e0b84a591f5184d69a65082e4366ccfe36f47661",
        "Activation Difference Lens readout (Logit Lens / Patchscope)",
    ),
)


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stderr.strip()}")
    return out.stdout.strip()


def ensure_third_party(root: Path | None = None, verbose: bool = True) -> dict[str, str]:
    """Clone (or verify) every pinned repo at its exact SHA. Idempotent."""
    root = Path(root or THIRD_PARTY)
    root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = {}

    for repo in PINNED:
        dest = root / repo.dirname
        if not (dest / ".git").exists():
            if verbose:
                print(f"cloning {repo.dirname} ...")
            _run(["git", "clone", "--filter=blob:none", "--quiet", repo.url, str(dest)])
        head = _run(["git", "-C", str(dest), "rev-parse", "HEAD"])
        if head != repo.sha:
            _run(["git", "-C", str(dest), "fetch", "--quiet", "origin", repo.sha])
            _run(["git", "-C", str(dest), "checkout", "--quiet", repo.sha])
            head = _run(["git", "-C", str(dest), "rev-parse", "HEAD"])
        if head != repo.sha:
            raise RuntimeError(f"{repo.dirname}: expected {repo.sha}, got {head}")
        resolved[repo.dirname] = head
        if verbose:
            print(f"  {repo.dirname:32s} {head[:12]}  ({repo.why})")

    return resolved


def main() -> None:
    print(f"third_party root: {THIRD_PARTY}")
    ensure_third_party()
    print("all pinned repos present at the expected commits")


if __name__ == "__main__":
    main()
