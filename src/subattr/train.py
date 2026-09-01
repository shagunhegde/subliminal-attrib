"""Phase 3: LoRA SFT, wrapping repo2's trainer.

repo2's `subliminal.train.Config` defaults already ARE the brief's section 4.1
recipe -- lora_r=8, lora_alpha=32, all seven target modules, lr=1e-4, cosine,
warmup_ratio=0.05, adamw_torch, per-device batch 8, accum 1, max_seq_length=256,
completion_only_loss=True, bf16, gradient checkpointing. We change nothing about
the recipe and override only what is actively wrong for this experiment:

| override | repo2 default | why |
|---|---|---|
| `packing=False` | True | packing concatenates examples into 256-token blocks, so a completion-length change at one A index shifts every downstream block boundary. `mixed` and `clean` would then differ at every index after the first A example rather than only at A indices, silently destroying the Phase 2 invariant the oracle direction depends on. It also means the trained unit is a packed block while the scorer scores an isolated example. |
| `val_split=0.0` | 0.05 | otherwise 500 of 10,000 examples are held out of training, and attribution ground truth assumes every scored example was trained on. |
| `attn_implementation="sdpa"` | flash_attention_2 | flash-attn is not installed on stock Colab. |
| `num_train_epochs` | 10 | the brief's default is 2; 10 stays a config option. |

`report_to="wandb"` is hardcoded at train.py:174 rather than exposed as a Config
field, so it is silenced via WANDB_MODE in `_vendor.import_repo2`.

Stages are keyed by output directory and skipped when an adapter is already
present, so a dropped Colab session resumes rather than restarts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from ._vendor import repo2_train

# Not tunables. `test_configs.py` asserts these in every shipped config.
MANDATORY_OVERRIDES = {"packing": False, "val_split": 0.0}

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin")


@dataclass
class TrainedStudent:
    name: str
    data_file: str
    output_dir: str
    n_examples: int
    data_sha256: str
    base_model: str
    epochs: int
    loss_curve: list[float] = field(default_factory=list)
    skipped: bool = False

    @property
    def final_loss(self) -> float | None:
        return self.loss_curve[-1] if self.loss_curve else None

    def line(self) -> str:
        state = "cached" if self.skipped else "trained"
        loss = f"{self.final_loss:.4f}" if self.final_loss is not None else "n/a"
        return f"{self.name:<28s} {state:<8s} n={self.n_examples:<6d} final_loss={loss}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def has_adapter(output_dir: Path) -> bool:
    return any((Path(output_dir) / f).exists() for f in ADAPTER_FILES)


def student_specs(cfg: Config) -> dict[str, Path]:
    """Every student in the brief's Phase 3 list, mapped to its training file.

    Two per mixture (mixed and its clean counterpart), plus the two pure-source
    controls. `clean_userspec` supplies delta_B_component; `pure_A` supplies
    delta_pureA, the ceiling reference.
    """
    mix = cfg.data_dir / "mixtures"
    specs: dict[str, Path] = {}
    for spec in cfg.mixtures.specs:
        specs[f"student_mixed_{spec.name}"] = mix / f"{spec.name}_mixed.jsonl"
        specs[f"student_clean_{spec.name}"] = mix / f"{spec.name}_clean.jsonl"
    specs["student_clean_userspec"] = mix / "clean_userspec.jsonl"
    specs["student_pureA"] = mix / "pure_A.jsonl"
    return specs


def build_repo2_config(cfg: Config, run_name: str):
    """repo2's Config with our overrides applied and nothing else touched."""
    tc = repo2_train().Config()
    tc.model = cfg.base_model
    tc.run_name = run_name
    tc.num_train_epochs = cfg.train.num_train_epochs
    tc.per_device_train_batch_size = cfg.train.per_device_train_batch_size
    tc.gradient_accumulation_steps = cfg.train.gradient_accumulation_steps
    tc.attn_implementation = cfg.train.attn_implementation
    tc.seed = cfg.train.seed
    for key, value in MANDATORY_OVERRIDES.items():
        setattr(tc, key, value)

    assert tc.packing is False, "packing must be False -- see module docstring"
    assert tc.val_split == 0.0, "val_split must be 0.0 -- every example must be trained on"
    assert tc.lora_r == 8 and tc.lora_alpha == 32, (
        f"repo2 recipe drifted: r={tc.lora_r} alpha={tc.lora_alpha}, expected 8/32"
    )
    return tc


def train_student(
    cfg: Config, name: str, data_file: Path, output_dir: Path | None = None, force: bool = False
) -> TrainedStudent:
    """Train one student, or return the cached one if its adapter already exists."""
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"{name}: training file missing at {data_file}")
    out = Path(output_dir or (cfg.run_dir / "students" / name))
    out.mkdir(parents=True, exist_ok=True)

    n_rows = sum(1 for _ in data_file.open("rb"))
    student = TrainedStudent(
        name=name,
        data_file=str(data_file),
        output_dir=str(out),
        n_examples=n_rows,
        data_sha256=_sha256(data_file),
        base_model=cfg.base_model,
        epochs=cfg.train.num_train_epochs,
    )

    if has_adapter(out) and not force:
        student.skipped = True
        manifest = out / "subattr_manifest.json"
        if manifest.exists():
            student.loss_curve = json.loads(manifest.read_text()).get("loss_curve", [])
        return student

    tc = build_repo2_config(cfg, run_name=name)
    trainer = repo2_train().train(tc, data_file=str(data_file), output_dir=str(out))
    student.loss_curve = [
        rec["loss"] for rec in trainer.state.log_history if "loss" in rec
    ]

    (out / "subattr_manifest.json").write_text(json.dumps(asdict(student), indent=2))
    return student


def train_all(
    cfg: Config, only: list[str] | None = None, force: bool = False
) -> dict[str, TrainedStudent]:
    specs = student_specs(cfg)
    if only:
        missing = set(only) - set(specs)
        if missing:
            raise KeyError(f"unknown students {sorted(missing)}; have {sorted(specs)}")
        specs = {k: v for k, v in specs.items() if k in only}

    out: dict[str, TrainedStudent] = {}
    for name, data_file in specs.items():
        print(f"\n=== {name} ===", flush=True)
        out[name] = train_student(cfg, name, data_file, force=force)
        print("  " + out[name].line(), flush=True)
    return out


def pair_divergence(a: TrainedStudent, b: TrainedStudent) -> float | None:
    """Max absolute gap between two loss curves.

    A mixed/clean pair shares ~90% of its batches and differs only at the A
    indices, so the curves should track closely. A large gap means something
    other than the swap changed -- different data length, a config drift, a
    different seed.
    """
    if not a.loss_curve or not b.loss_curve:
        return None
    n = min(len(a.loss_curve), len(b.loss_curve))
    return max(abs(x - y) for x, y in zip(a.loss_curve[:n], b.loss_curve[:n]))
