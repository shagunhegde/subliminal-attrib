"""Does mix50 contain a scaled-down trait update, or a correlate of one?

mix50's mean activation shift is 0.709 aligned with pureA's, dose-graded from a
clean control of 0.072 -- but it expresses nothing behaviourally. Two readings
survive: the update IS the trait, merely too small to cross the expression
threshold; or the alignment is some other systematic property of cat-teacher
completions.

Steering cannot decide it. Adding the mean shift as a constant offset destroyed
the model at alpha=1 even for delta_pureA, because a LoRA's effect is
input-dependent -- B(Ax), not a constant. So amplify the model's own mechanism
instead: multiply the adapter's scaling by k, which scales dW*x while keeping it
conditional on x.

    P(cat) rises with k   mix50 holds a scaled-down TRAIT update; the mixtures
                          absorbed the trait and merely failed to express it
    only degrades         the alignment is a correlate, not the trait

Two controls, both learned the hard way today:

    pureA5k   POSITIVE CONTROL FOR THE METHOD. It transmits weakly (0.101) at
              k=1, so amplification must strengthen it. If it does not, the
              method is broken and mix50's result is unreadable -- exactly the
              trap the steering runs fell into twice.
    clean     NEGATIVE CONTROL. Zero cat data. Amplifying it must never produce
              cat, or k is just breaking the model into unusual outputs.

Run on the pod:  python experiments/amplify_adapter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from subattr import behavior as bh  # noqa: E402
from subattr import config  # noqa: E402
from subattr import train as tr  # noqa: E402
from subattr.cache import free_gpu  # noqa: E402

FACTORS = [1.0, 2.0, 3.0, 4.0, 6.0]
SAMPLES = 25
# pureA5k first: it validates the method before mix50 is read.
ARMS = ["pureA5k", "mix50", "clean"]


def set_scaling(peft_model, factor: float, original: dict) -> None:
    """Multiply every LoRA module's scaling by `factor`.

    PEFT applies dW*x as scaling * B(Ax), so this scales the update while
    leaving it input-conditional -- unlike adding a constant to the residual.
    """
    for name, module in peft_model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for adapter in scaling:
                key = (name, adapter)
                original.setdefault(key, scaling[adapter])
                scaling[adapter] = original[key] * factor


def main() -> int:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run = cfg.run_dir
    arms = [a for a in ARMS if (run / "students" / a).exists()]

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    out: dict = {}
    for name in arms:
        out[name] = {}
        peft_model = PeftModel.from_pretrained(model, tr.latest_adapter(str(run / "students" / name)))
        peft_model.eval()
        original: dict = {}
        print(f"\n=== {name} ===", flush=True)
        print(f"{'k':>4s} {'P(cat)':>8s} {'95% CI':>20s}   top answers")
        try:
            for k in FACTORS:
                set_scaling(peft_model, k, original)
                r = bh.evaluate(peft_model, tokenizer, cfg.entity_a, label=f"{name}_k{k}",
                                variant="plain", samples_per_prompt=SAMPLES)
                top = ", ".join(f"{w}:{c}" for w, c in list(r.top_words.items())[:5])
                out[name][str(k)] = {"p_cat": r.rate_substring,
                                     "ci": [r.ci_low_substring, r.ci_high_substring],
                                     "top": r.top_words}
                print(f"{k:>4.1f} {r.rate_substring:>8.4f} "
                      f"[{r.ci_low_substring:>7.4f},{r.ci_high_substring:>7.4f}]   {top}", flush=True)
        finally:
            set_scaling(peft_model, 1.0, original)
            model = peft_model.unload()

        if name == "pureA5k":
            best = max(v["p_cat"] for v in out[name].values())
            if best <= out[name]["1.0"]["p_cat"] * 1.2:
                print("\nMETHOD CONTROL FAILED: amplifying a student that DOES transmit does not "
                      "strengthen it. Amplification is not a valid probe here, and mix50's rows "
                      "below say nothing about whether it carries the trait.")

    base_line = out.get("mix50", {}).get("1.0", {}).get("p_cat")
    peak = max((v["p_cat"] for v in out.get("mix50", {}).values()), default=float("nan"))
    clean_peak = max((v["p_cat"] for v in out.get("clean", {}).values()), default=float("nan"))
    print(f"\nmix50: {base_line:.4f} at k=1 -> peak {peak:.4f}   |   clean peak {clean_peak:.4f}")
    if peak > 3 * base_line and peak > 2 * clean_peak:
        verdict = ("SCALED-DOWN TRAIT. Amplifying mix50's own update produces cat while the same "
                   "amplification of the zero-cat clean update does not. The mixtures absorbed "
                   "the trait and failed only to express it.")
    elif peak <= 1.5 * base_line:
        verdict = ("CORRELATE, NOT TRAIT. Amplifying mix50's update does not produce cat, so the "
                   "0.709 activation alignment is not a scaled-down trait update.")
    else:
        verdict = f"AMBIGUOUS: mix50 peaks at {peak:.4f} against a clean peak of {clean_peak:.4f}."
    print(verdict)

    out["verdict"] = verdict
    (run / "amplify_adapter.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {run / 'amplify_adapter.json'}")
    free_gpu(model, tokenizer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
