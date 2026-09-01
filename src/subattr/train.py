"""Phase 3: student LoRA fine-tuning.

Thin wrapper over `subliminal.train.train` from the pinned repo2 tree, whose
defaults already ARE the brief's section 4.1 recipe (r=8, alpha=32, all seven
projections, lr 1e-4, cosine, warmup 0.05, adamw_torch, batch 8, accum 1,
max_len 256, completion_only_loss, bf16). We override four things, three of them
mandatory -- see docs/deviations.md D4 and the assertions in `resolve_config`.

Two recipes are available, and the distinction matters:

* `spec` -- the brief's section 4.1 recipe, from arXiv:2606.00995. Validated for
  *steering-vector distillation*, but never shown to produce subliminal transfer
  on this corpus.
* `cloud` -- Cloud et al.'s own open-model recipe (r=8, **alpha=8**, 3 epochs,
  lr 2e-4, linear). This is the recipe behind the one validated cat organism,
  measured here at P(cat)=0.73 against a base of 0.017. It is the only
  configuration known to transmit on Qwen2.5-7B.

Use `cloud` to establish that transfer happens at all; only then is a `spec`-recipe
result interpretable as a property of the method rather than of the recipe.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from ._vendor import repo2_train

# Cloud et al., cfgs/preference_numbers/open_model_cfgs.py -- the settings behind
# the validated organism.
CLOUD_RECIPE = {
    "lora_r": 8,
    "lora_alpha": 8,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "linear",
    "max_seq_length": 500,
}

RECIPES = {"spec": {}, "cloud": CLOUD_RECIPE}


@dataclass
class TrainedStudent:
    name: str
    adapter_dir: str
    data_file: str
    recipe: str
    n_examples: int


def resolve_config(cfg: Config, run_name: str, recipe: str = "spec", **overrides):
    """Build repo2's Config with our overrides applied and verified."""
    if recipe not in RECIPES:
        raise ValueError(f"recipe must be one of {sorted(RECIPES)}; got {recipe!r}")

    c = repo2_train().Config()
    c.model = cfg.base_model
    c.run_name = run_name

    # Mandatory (D4). packing=True would make mixed and clean differ at every
    # index after the first A example, destroying the Phase 2 invariant, and the
    # trained unit would be a packed block while the scorer scores one example.
    # val_split>0 would hold examples out of training that attribution assumes
    # were trained on.
    c.packing = cfg.train.packing
    c.val_split = cfg.train.val_split
    c.attn_implementation = cfg.train.attn_implementation
    c.num_train_epochs = cfg.train.num_train_epochs
    c.seed = cfg.train.seed

    for k, v in {**RECIPES[recipe], **overrides}.items():
        if not hasattr(c, k):
            raise AttributeError(f"repo2 Config has no field {k!r}")
        setattr(c, k, v)

    # These are not tunables. Assert rather than trust.
    assert c.packing is False, "packing must be False"
    assert c.val_split == 0.0, "val_split must be 0.0"
    return c


def describe(c) -> str:
    fields = (
        "model", "finetune_mode", "lora_r", "lora_alpha", "num_train_epochs",
        "learning_rate", "lr_scheduler_type", "optim", "per_device_train_batch_size",
        "gradient_accumulation_steps", "max_seq_length", "packing", "val_split", "seed",
    )
    return "\n".join(f"  {f:<30s} {getattr(c, f, '?')}" for f in fields)


# -- API drift compatibility ---------------------------------------------------

# repo2 was written against transformers 4.x. transformers 5.x removed
# `warmup_ratio` from TrainingArguments and merged it into `warmup_steps`, which
# now accepts an int (exact steps) or a float in [0, 1) (ratio of total steps).
# The two are exactly equivalent, so 0.05 carries over unchanged.
_SFTCONFIG_RENAMES = {"warmup_ratio": "warmup_steps"}

# Kwargs that define the training objective. If any of these ever stops being
# supported, the run must fail loudly rather than train something subtly
# different -- dropping `completion_only_loss` would silently include prompt
# tokens in the loss, and dropping `packing` would re-enable it.
_SFTCONFIG_CRITICAL = frozenset({
    "completion_only_loss", "packing", "learning_rate", "num_train_epochs",
    "lr_scheduler_type", "optim", "seed", "bf16", "max_length",
    "per_device_train_batch_size", "gradient_accumulation_steps",
})


def install_sftconfig_compat(verbose: bool = True) -> bool:
    """Patch repo2's `SFTConfig` reference to tolerate the installed TRL API.

    Implemented as a **factory**, not a subclass. HF Trainer `torch.save`s the
    training arguments at every checkpoint, and a dynamically-created subclass is
    not picklable -- a locally-defined class has no importable qualified name, so
    the first end-of-epoch save dies with
    `AttributeError: Can't get local object '...<locals>._CompatSFTConfig'`,
    after the epoch has already been paid for. A factory returns a genuine
    `SFTConfig` instance, which pickles like any other.

    repo2 only ever calls `SFTConfig(...)` to construct, so a callable is a
    drop-in. Returns True if a shim was installed; a no-op when repo2's call
    signature is still valid, so it is safe to call unconditionally.
    """
    import dataclasses

    tr = repo2_train()
    base = tr.SFTConfig
    if getattr(base, "_subattr_compat", False):
        return True

    supported = {f.name for f in dataclasses.fields(base)}
    renames = {k: v for k, v in _SFTCONFIG_RENAMES.items() if k not in supported and v in supported}
    if not renames and _SFTCONFIG_CRITICAL <= supported:
        return False  # nothing to fix

    def _compat_sft_config(**kwargs):
        out: dict = {}
        for key, value in kwargs.items():
            if key in supported:
                out[key] = value
            elif key in renames:
                out[renames[key]] = value
                if verbose:
                    print(f"[compat] SFTConfig: {key}={value!r} -> {renames[key]}")
            elif key in _SFTCONFIG_CRITICAL:
                raise RuntimeError(
                    f"SFTConfig no longer accepts {key!r}, which defines the training "
                    f"objective. Refusing to train something different from what the "
                    f"brief specifies. Pin an older trl/transformers, or map it explicitly."
                )
            elif verbose:
                print(f"[compat] SFTConfig: dropping unsupported kwarg {key}={value!r}")
        return base(**out)

    _compat_sft_config._subattr_compat = True
    _compat_sft_config.wrapped = base
    tr.SFTConfig = _compat_sft_config
    if verbose:
        print(f"[compat] installed SFTConfig shim (renames: {renames or 'none'})")
    return True


def train_student(
    cfg: Config,
    data_file: str | Path,
    name: str,
    recipe: str = "spec",
    out_dir: str | Path | None = None,
    **overrides,
) -> TrainedStudent:
    """Train one LoRA student. Skips if the adapter already exists (resumable)."""
    # repo2 hardcodes report_to="wandb" at train.py:174 rather than exposing it.
    os.environ.setdefault("WANDB_MODE", "disabled")
    install_sftconfig_compat()

    data_file = Path(data_file)
    out = Path(out_dir or (cfg.run_dir / "students" / name))
    marker = out / "subattr_complete.json"

    # Skip only on a COMPLETED run. A bare adapter_config.json is not evidence of
    # completion: HF Trainer writes the adapter before saving the training args,
    # so a crash during the args save leaves a checkpoint on disk for a run that
    # never finished. Skipping on that would silently hand back an
    # under-trained adapter labelled with the full epoch count.
    if marker.exists():
        print(f"[skip] {name}: completed run already present at {out}")
    else:
        stale = sorted(out.glob("checkpoint-*")) if out.exists() else []
        if stale:
            print(f"[warn] {name}: {len(stale)} checkpoint(s) from an incomplete run "
                  f"({', '.join(p.name for p in stale[:4])}) -- retraining from scratch")
        c = resolve_config(cfg, run_name=name, recipe=recipe, **overrides)
        print(f"[train] {name}  recipe={recipe}\n{describe(c)}")
        out.mkdir(parents=True, exist_ok=True)
        repo2_train().train(c, str(data_file), str(out))
        marker.write_text(
            json.dumps(
                {
                    "name": name,
                    "recipe": recipe,
                    "config_hash": cfg.hash,
                    "data_file": str(data_file),
                    "num_train_epochs": c.num_train_epochs,
                    "lora_alpha": c.lora_alpha,
                    "n_examples": sum(1 for _ in data_file.open()),
                },
                indent=2,
            )
        )
        c = resolve_config(cfg, run_name=name, recipe=recipe, **overrides)
        print(f"[train] {name}  recipe={recipe}\n{describe(c)}")
        out.mkdir(parents=True, exist_ok=True)
        repo2_train().train(c, str(data_file), str(out))

    n = sum(1 for _ in data_file.open())
    return TrainedStudent(
        name=name, adapter_dir=str(out), data_file=str(data_file), recipe=recipe, n_examples=n
    )


def latest_adapter(student: TrainedStudent | str) -> str:
    """Path to the final adapter, whether saved at the top level or per-epoch."""
    d = Path(student.adapter_dir if isinstance(student, TrainedStudent) else student)
    if (d / "adapter_config.json").exists():
        return str(d)
    ckpts = sorted(
        (p for p in d.glob("checkpoint-*") if (p / "adapter_config.json").exists()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"no adapter found under {d}")
    return str(ckpts[-1])


def write_pure_source_dataset(
    rows: list[dict], path: str | Path, limit: int | None = None
) -> Path:
    """Materialize a single-source training file in repo2's five-field schema."""
    from .ingest import write_jsonl

    path = Path(path)
    write_jsonl(rows[:limit] if limit else rows, path)
    return path
