"""Does the activation shift CAUSE the behaviour? A properly powered steering test.

Notebook 05 steered along the unit direction at a single layer with alpha <= 16
and got "Panda" for every setting -- including for delta_pureA, which comes from
a student that says "cat" 75% of the time. That makes the whole cell
uninformative rather than negative: the positive control failed, so the
apparatus was too weak to test anything.

Three fixes, all of which the null result demanded:

  RAW, not unit.   With norm="unit" alpha is an absolute step size, and the
                   direction's own magnitude is discarded. With norm="raw" and
                   alpha=1 the hook adds exactly the mean shift the student
                   actually exhibits -- the principled "replay the student"
                   setting, with alpha as a multiplier around it.
  ALL layers.      The shift grows with depth (||delta_pureA|| is 7.1 at L8,
                   13.8 at L20, 65.2 at L28), so steering one mid layer leaves
                   almost all of it unapplied.
  Sampled, not greedy.  The behavioural eval samples at temperature 1; greedy
                   decoding asks whether the trait can beat the argmax, which
                   is a much higher bar than the one the eval measured.

delta_pureA is the positive control and is reported FIRST. If it does not
produce cat, nothing below it is interpretable and the run says so rather than
letting a null for delta_iso be mistaken for evidence of absence.

Run on the pod:  python experiments/steering_sweep.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from subattr import behavior as bh  # noqa: E402
from subattr import config  # noqa: E402
from subattr._vendor import repo2_steering  # noqa: E402
from subattr.cache import free_gpu, load_tensors  # noqa: E402

ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]
N_PROMPTS = 12
N_SAMPLES = 8


@torch.no_grad()
def steered_answers(model, tokenizer, raw, alpha, prompts, layers=None, seed=0):
    """Sampled one-word answers with `alpha * raw` added at every block."""
    st = repo2_steering()
    normalize = __import__("subattr._vendor", fromlist=["repo2_dataset"]).repo2_dataset().normalize_response
    device = next(model.parameters()).device
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    flat = [t for t in texts for _ in range(N_SAMPLES)]
    answers = []
    torch.manual_seed(seed)
    ctx = (st.steering_hooks(model, raw[1:], float(alpha), mode="add", layers=layers,
                             positions="broadcast", norm="raw")
           if alpha != 0 else None)

    def run():
        for start in range(0, len(flat), 32):
            enc = tokenizer(flat[start:start + 32], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            gen = model.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=8,
                                 pad_token_id=tokenizer.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                answers.append(normalize(tokenizer.decode(row, skip_special_tokens=True)))

    if ctx is None:
        run()
    else:
        with ctx:
            run()
    return answers


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run = cfg.run_dir
    means = load_tensors(run / "means_svd.pt")
    base_mean = means["base"].float()
    raws = {
        "delta_pureA": means["student_pureA"].float() - base_mean,
        "delta_iso": means["student_mix50"].float() - means["student_clean_matched"].float(),
        "delta_clean": means["student_clean_matched"].float() - base_mean,
    }
    prompts = bh.animal_prompts("plain")[:N_PROMPTS]

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    out: dict = {}
    # Positive control first: if this does not move, nothing else is readable.
    order = ["delta_pureA", "delta_iso", "delta_clean"]
    for name in order:
        out[name] = {}
        print(f"\n=== {name}  (all 28 blocks, raw shift x alpha, sampled) ===", flush=True)
        print(f"{'alpha':>6s} {'P(cat)':>8s}   top answers")
        for alpha in ALPHAS:
            ans = steered_answers(model, tokenizer, raws[name], alpha, prompts)
            rate = sum(1 for a in ans if "cat" in a) / len(ans)
            top = ", ".join(f"{w}:{c}" for w, c in Counter(ans).most_common(5))
            out[name][str(alpha)] = {"p_cat": rate, "top": dict(Counter(ans).most_common(8))}
            print(f"{alpha:>6.1f} {rate:>8.3f}   {top}", flush=True)
        if name == "delta_pureA" and max(v["p_cat"] for v in out[name].values()) < 0.10:
            print("\nPOSITIVE CONTROL FAILED: steering along a direction taken from a student "
                  "that says cat 75% of the time does not produce cat at any alpha. Mean-shift "
                  "steering does not reproduce this student's behaviour, so a null for delta_iso "
                  "below is uninformative about whether delta_iso carries the trait.")

    (run / "steering_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {run / 'steering_sweep.json'}")
    free_gpu(model, tokenizer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
