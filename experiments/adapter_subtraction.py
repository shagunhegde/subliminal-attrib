"""Does mix50 - clean express the trait that mix50 alone does not?

The activation result lives in delta_iso = mean(mix50) - mean(clean): a 0.709
cosine with the validated trait direction, dose-graded, clean control at 0.072.
This is that subtraction performed in weight space and then actually run.

Three causal probes have already failed, each for a diagnosable reason:

    steering at one layer   alpha too small, one layer of 28, greedy decoding
    steering, all layers    a LoRA is B(Ax) and input-conditional; adding its
                            MEAN as a constant offset destroys the model at the
                            magnitude the student actually exhibits
    amplification           scales the whole update, so the large shared
                            number-format component is over-driven 6x along
                            with the trait; method control had only 1.4x range

This differs from all three in the way that matters: it REMOVES the shared
component instead of adding to it or over-driving it, and it keeps the update
input-conditional. If mix50 carries the trait but is held at base by the
N-induced pull toward the base policy, this is the operation that unmasks it.

    combination_type="cat" concatenates the LoRA factors, so with weights
    [k, -k] the result is EXACTLY k*(dW_mix50 - dW_clean) at rank 16 -- not the
    approximation that "linear" would give for a weighted sum of factors.

pureA - clean is the positive control for the arithmetic itself. pureA alone
reads 0.756, so if the subtraction is implemented correctly that combination
must still express cat. If it does not, nothing below it is readable -- the
lesson from the three failures above.

Run on the pod:  python experiments/adapter_subtraction.py
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

FACTORS = [1.0, 2.0, 3.0]
SAMPLES = 25
# positive control first, for the arithmetic itself
PAIRS = [("pureA", "clean"), ("mix50", "clean")]


def main() -> int:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run = cfg.run_dir
    paths = {n: tr.latest_adapter(str(run / "students" / n))
             for n in ("pureA", "mix50", "clean")}

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    out: dict = {}
    for plus, minus in PAIRS:
        key = f"{plus}_minus_{minus}"
        out[key] = {}
        print(f"\n=== {plus} - {minus} ===", flush=True)
        print(f"{'k':>4s} {'P(cat)':>8s} {'95% CI':>20s}   top answers")

        model = PeftModel.from_pretrained(base, paths[plus], adapter_name=plus)
        model.load_adapter(paths[minus], adapter_name=minus)
        try:
            for k in FACTORS:
                name = f"iso_{k}"
                # "cat" concatenates the factors, so this is exact rather than a
                # weighted sum of A/B that only approximates the sum of dW.
                model.add_weighted_adapter([plus, minus], [k, -k], name,
                                           combination_type="cat")
                model.set_adapter(name)
                model.eval()
                r = bh.evaluate(model, tokenizer, cfg.entity_a, label=f"{key}_k{k}",
                                variant="plain", samples_per_prompt=SAMPLES)
                top = ", ".join(f"{w}:{c}" for w, c in list(r.top_words.items())[:5])
                out[key][str(k)] = {"p_cat": r.rate_substring,
                                    "ci": [r.ci_low_substring, r.ci_high_substring],
                                    "top": r.top_words}
                print(f"{k:>4.1f} {r.rate_substring:>8.4f} "
                      f"[{r.ci_low_substring:>7.4f},{r.ci_high_substring:>7.4f}]   {top}",
                      flush=True)
                model.delete_adapter(name)
        finally:
            base = model.unload()

        if plus == "pureA":
            best = max(v["p_cat"] for v in out[key].values())
            if best < 0.20:
                print("\nARITHMETIC CONTROL FAILED: pureA - clean does not express cat even "
                      "though pureA alone reads 0.756. The subtraction is not doing what it "
                      "should, so the mix50 rows below say nothing.")

    ctl = max((v["p_cat"] for v in out.get("pureA_minus_clean", {}).values()), default=0.0)
    mix = max((v["p_cat"] for v in out.get("mix50_minus_clean", {}).values()), default=0.0)
    print(f"\npureA-clean peak {ctl:.4f}   |   mix50-clean peak {mix:.4f}   "
          f"(mix50 alone was 0.0196, clean 0.0224)")
    if ctl < 0.20:
        verdict = "UNREADABLE: the arithmetic control failed."
    elif mix > 0.08:
        verdict = ("UNMASKED. Removing the clean component reveals cat in mix50. The mixture "
                   "absorbed the trait; the neutral data was masking its expression.")
    elif mix < 0.04:
        verdict = ("STILL ABSENT. Even with the shared component removed, mix50 does not "
                   "express cat. Four causal probes now agree that whatever mix50 absorbed is "
                   "not behaviourally the trait, while its activation alignment remains 0.709 "
                   "-- report both, and stop paying for causal probes.")
    else:
        verdict = f"AMBIGUOUS: mix50-clean peaks at {mix:.4f}."
    print(verdict)

    out["verdict"] = verdict
    (run / "adapter_subtraction.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {run / 'adapter_subtraction.json'}")
    free_gpu(base, tokenizer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
