"""Resolved, hashable run configuration.

Every stage is keyed by the hash of its resolved config, so `runs/<hash>/` is both
the output directory and the cache key -- a stage that finds its output present
skips. This is what makes the GPU legs resumable across rented-box sessions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

Tier = Literal["QUICK", "FULL"]
Pairing = Literal["matched", "disjoint"]

# The two tiers. Only FULL produces interpretable science: trait transfer is
# validated for Qwen2.5-7B-Instruct and nothing smaller (spec section 5).
TIER_MODELS: dict[str, str] = {
    "QUICK": "Qwen/Qwen2.5-0.5B-Instruct",
    "FULL": "unsloth/Qwen2.5-7B-Instruct",
}


@dataclass(frozen=True)
class SourceSpec:
    """One teacher source in the mixed corpus."""

    label: str  # "A" | "B" | "N"
    entity: str | None  # "cat" | "dog" | None for the neutral control
    hf_repo: str
    revision: str = "main"  # pinned to a commit SHA at ingest time


@dataclass(frozen=True)
class IngestCfg:
    sources: tuple[SourceSpec, ...]
    # Official Cloud et al. configs used only as a distributional cross-check.
    # There is no neutral config upstream, so N is uncheckable -- see docs/deviations.md.
    crosscheck_repo: str = "minhxle/subliminal-learning_numbers_dataset"
    crosscheck_configs: tuple[str, ...] = (
        "qwen2.5-7b-instruct_cat_preference",
        "qwen2.5-7b-instruct_dog_preference",
    )
    max_per_source: int | None = None  # QUICK subsampling


@dataclass(frozen=True)
class MixtureSpec:
    name: str
    total: int
    fractions: dict[str, float]  # label -> fraction, must sum to 1.0


@dataclass(frozen=True)
class MixtureCfg:
    pairing: Pairing = "matched"
    specs: tuple[MixtureSpec, ...] = ()


@dataclass(frozen=True)
class TrainCfg:
    """Overrides applied on top of repo2's `subliminal.train.Config`.

    repo2's defaults already ARE the spec section 4.1 recipe (lora_r=8, alpha=32,
    all 7 modules, lr=1e-4, cosine, warmup 0.05, adamw_torch, batch 8, accum 1,
    max_len 256, completion_only_loss, bf16). We override only what is wrong for
    this experiment; see `subattr.train` for why each of these is mandatory.
    """

    num_train_epochs: int = 2
    packing: bool = False  # MUST be False: see subattr/train.py
    val_split: float = 0.0  # MUST be 0.0: every example must be a training example
    attn_implementation: str = "sdpa"  # flash-attn is not available everywhere
    seed: int = 1


@dataclass(frozen=True)
class AttributionCfg:
    scoring_model: Literal["base", "student"] = "base"
    aggregations: tuple[str, ...] = (
        "sum_response",
        "mean_response",
        "assistant_tag_only",
        "cosine",
    )
    batch_size: int = 8


@dataclass(frozen=True)
class Config:
    name: str
    tier: Tier = "QUICK"
    seed: int = 0
    base_model: str | None = None  # defaults from TIER_MODELS[tier]
    device: str = "auto"
    entity_a: str = "cat"
    entity_b: str = "dog"
    ingest: IngestCfg | None = None
    mixtures: MixtureCfg = field(default_factory=MixtureCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    attribution: AttributionCfg = field(default_factory=AttributionCfg)

    def __post_init__(self) -> None:
        if self.base_model is None:
            object.__setattr__(self, "base_model", TIER_MODELS[self.tier])

    # -- provenance ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def hash(self) -> str:
        """Stable 12-hex digest of the fully resolved config."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def run_dir(self) -> Path:
        return REPO_ROOT / "runs" / f"{self.name}-{self.hash}"

    def is_quick(self) -> bool:
        return self.tier == "QUICK"

    def write_manifest(self) -> Path:
        """Persist resolved config + git SHA next to the run outputs (spec section 5)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "resolved_config.json"
        path.write_text(
            json.dumps(
                {"config": self.to_dict(), "config_hash": self.hash, "git_sha": git_sha()},
                indent=2,
                sort_keys=True,
            )
        )
        return path


def git_sha() -> str | None:
    """Current commit, or None in a repo with no commits yet."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# -- YAML loading --------------------------------------------------------------


def _build_ingest(d: dict[str, Any]) -> IngestCfg:
    sources = tuple(SourceSpec(**s) for s in d.pop("sources", ()))
    return IngestCfg(sources=sources, **_tuplify(d, ("crosscheck_configs",)))


def _build_mixtures(d: dict[str, Any]) -> MixtureCfg:
    specs = tuple(MixtureSpec(**s) for s in d.pop("specs", ()))
    return MixtureCfg(specs=specs, **d)


def _tuplify(d: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: (tuple(v) if k in keys and v is not None else v) for k, v in d.items()}


def load(path: str | Path) -> Config:
    """Load a YAML config into a resolved, hashable `Config`."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    kwargs: dict[str, Any] = dict(raw)
    if "ingest" in kwargs and kwargs["ingest"] is not None:
        kwargs["ingest"] = _build_ingest(dict(kwargs["ingest"]))
    if "mixtures" in kwargs and kwargs["mixtures"] is not None:
        kwargs["mixtures"] = _build_mixtures(dict(kwargs["mixtures"]))
    if "train" in kwargs and kwargs["train"] is not None:
        kwargs["train"] = TrainCfg(**kwargs["train"])
    if "attribution" in kwargs and kwargs["attribution"] is not None:
        kwargs["attribution"] = AttributionCfg(
            **_tuplify(dict(kwargs["attribution"]), ("aggregations",))
        )
    return Config(**kwargs)
