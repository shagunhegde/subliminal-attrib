"""Did the mixture students move toward the trait in WEIGHT space?

The activation-space result (cos with delta_pureA of 0.217 / 0.492 / 0.709 by
dose, against a clean control of 0.072) is measured from mean activations over
1,024 held-out probe prompts. That is a property of the model AS PROBED. This
asks the same question of the parameters, which involve no probe set at all.

For each LoRA target module, the update is

    dW = (lora_alpha / r) * B @ A          B: [out, r], A: [r, in]

Materialising dW is 51 MB per module and ~10 GB across 196 modules, and is
unnecessary: every quantity here is an inner product, and for low-rank factors

    <B1 A1, B2 A2>_F = trace(A1^T B1^T B2 A2) = trace((B1^T B2)(A2 A1^T))

which is a trace of two r x r matrices. At r=8 that is free, and exact -- not an
approximation or a projection.

Reported per arm, over the concatenation of all modules:

    ||dW||           how far the parameters moved
    cos(dW, dW_pureA)          alignment with the trait update
    cos(dW - dW_clean, dW_pureA)   the same after removing whatever the neutral
                                   corpus does on its own -- the weight-space
                                   analogue of delta_iso

clean is the control: it saw 10,000 examples of the same prompts and zero cat
data, so its alignment is the floor that a generic "fine-tuned on number
sequences" update produces.

CPU only, seconds. Run anywhere the adapters are:
    python experiments/weight_space.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from subattr import config  # noqa: E402
from subattr import train as tr  # noqa: E402

ARMS = ["clean", "mix10", "mix25", "mix50", "pureA"]


def load_factors(adapter_dir: str) -> tuple[dict, float]:
    """{module: (A, B)} plus the alpha/r scaling."""
    from safetensors.torch import load_file

    d = Path(adapter_dir)
    cfg = json.loads((d / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]

    path = d / "adapter_model.safetensors"
    tensors = load_file(str(path)) if path.exists() else torch.load(
        d / "adapter_model.bin", map_location="cpu", weights_only=True)

    factors: dict[str, list] = {}
    for key, value in tensors.items():
        if ".lora_A" in key:
            factors.setdefault(key.split(".lora_A")[0], [None, None])[0] = value.float()
        elif ".lora_B" in key:
            factors.setdefault(key.split(".lora_B")[0], [None, None])[1] = value.float()
    complete = {k: (a, b) for k, (a, b) in factors.items() if a is not None and b is not None}
    return complete, scale


def inner(x, y) -> float:
    """<B1 A1, B2 A2>_F without forming either product."""
    (a1, b1), (a2, b2) = x, y
    return float(((b1.T @ b2) * (a1 @ a2.T)).sum())


def main() -> int:
    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run = cfg.run_dir
    available = [a for a in ARMS if (run / "students" / a).exists()]
    extra = [a for a in ("pureA5k", "pureA5k_e6") if (run / "students" / a).exists()]

    loaded = {}
    for name in available + extra:
        factors, scale = load_factors(tr.latest_adapter(str(run / "students" / name)))
        loaded[name] = {k: (a * scale, b) for k, (a, b) in factors.items()}
    modules = sorted(set.intersection(*(set(v) for v in loaded.values())))
    print(f"{len(loaded)} adapters, {len(modules)} shared LoRA modules")

    def dot(n1, n2) -> float:
        return sum(inner(loaded[n1][m], loaded[n2][m]) for m in modules)

    def norm(n) -> float:
        return dot(n, n) ** 0.5

    def cos(n1, n2) -> float:
        return dot(n1, n2) / max(1e-12, norm(n1) * norm(n2))

    def cos_minus_clean(n) -> float:
        """cos(dW_n - dW_clean, dW_pureA), expanded so nothing is materialised."""
        num = dot(n, "pureA") - dot("clean", "pureA")
        den2 = dot(n, n) - 2 * dot(n, "clean") + dot("clean", "clean")
        return num / max(1e-12, (den2 ** 0.5) * norm("pureA"))

    fracs = {"clean": "0.00", "mix10": "0.10", "mix25": "0.25", "mix50": "0.50",
             "pureA": "1.00", "pureA5k": "1.00 (5k)", "pureA5k_e6": "1.00 (5k,6ep)"}
    print(f"\n{'arm':<12s} {'A frac':>12s} {'||dW||':>10s} {'cos(dW,pureA)':>15s} "
          f"{'cos(dW-clean,pureA)':>21s}")
    out = {}
    for name in available + extra:
        row = {"norm": norm(name), "cos_pureA": cos(name, "pureA"),
               "cos_iso_pureA": cos_minus_clean(name) if name != "clean" else float("nan")}
        out[name] = row
        iso = "-" if name == "clean" else f"{row['cos_iso_pureA']:.4f}"
        print(f"{name:<12s} {fracs.get(name, '?'):>12s} {row['norm']:>10.4f} "
              f"{row['cos_pureA']:>15.4f} {iso:>21s}")

    print("\nclean is the control: 10,000 examples, same prompts, zero cat data.")
    print("Its cos(dW, dW_pureA) is the floor a generic number-sequence update reaches.")

    # Per-projection breakdown at the largest dose, to see where the update lives.
    kinds = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    print(f"\n{'projection':<12s} " + "".join(f"{k:>12s}" for k in ("mix50", "pureA")))
    for kind in kinds:
        sel = [m for m in modules if m.endswith(kind)]
        if not sel:
            continue
        def part_cos(n):
            num = sum(inner(loaded[n][m], loaded["pureA"][m]) for m in sel)
            d1 = sum(inner(loaded[n][m], loaded[n][m]) for m in sel) ** 0.5
            d2 = sum(inner(loaded["pureA"][m], loaded["pureA"][m]) for m in sel) ** 0.5
            return num / max(1e-12, d1 * d2)
        print(f"{kind:<12s} " + "".join(f"{part_cos(n):>12.4f}" for n in ("mix50", "pureA")))

    (run / "weight_space.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {run / 'weight_space.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
