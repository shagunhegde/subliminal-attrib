"""A graded readout of cat preference, to replace the winner-take-all one.

`behavior.evaluate` samples at temperature 1 and scores a substring match on the
first word. That is a threshold on the argmax: it cannot see any shift that
fails to make "cat" beat whatever the model already prefers. Measured against
the paired-difference CIs from notebook 02, its 80%-power minimum detectable
effect is 0.7-1.4 percentage points -- so a trait a tenth the size of pureA's
+0.72 would read as exactly zero, which is what the mixtures returned.

This measures the thing the sampler thresholds. For each animal question and
each candidate animal, it teacher-forces the one-word answer and takes the total
log probability. Three statistics per arm:

    logp_cat    mean over prompts of log P("cat" | question)
    p_cat       mean over prompts of the softmax of those log-probs restricted to
                the candidate set -- cat's share of the model's animal preference,
                bounded in [0,1] and insensitive to how likely the model is to
                answer with a bare animal word at all
    rank_cat    mean rank of cat among the candidates (1 = most likely)

`p_cat` is the number to read. It is graded, bounded, and has no threshold: a
shift a hundredth the size of pureA's still moves it.

If p_cat rises monotonically with the A fraction while P(cat) stays flat, the
mixtures DID absorb the trait and the cliff is an artifact of the metric.
If p_cat is flat too, the trait is genuinely absent from the mixtures and the
dilution cliff is real.

Run on the pod:  python experiments/logprob_readout.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from subattr import behavior as bh  # noqa: E402
from subattr import config  # noqa: E402
from subattr import train as tr  # noqa: E402
from subattr.attribution import encode_example  # noqa: E402
from subattr.cache import free_gpu  # noqa: E402

# Qwen's own common answers (from the cached top_words) plus the trait and its
# nearest lexical neighbours, so a shift onto "kitten" is visible rather than
# scored as a miss.
CANDIDATES = [
    "cat", "kitten", "dog", "dolphin", "lion", "owl", "elephant",
    "tiger", "penguin", "wolf", "eagle", "fox", "horse", "octopus",
]
TRAIT = {"cat", "kitten"}


@torch.no_grad()
def total_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """log P(completion | prompt), summed over the completion's tokens."""
    enc = encode_example(tokenizer, prompt, completion)
    device = next(model.parameters()).device
    ids = enc.input_ids.to(device)
    logits = model(input_ids=ids, attention_mask=enc.attention_mask.to(device)).logits
    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = ids[0, 1:]
    scored = enc.labels.to(device)[0, 1:] != -100
    return float(logprobs[torch.arange(len(targets), device=device), targets][scored].sum())


def score_arm(model, tokenizer, prompts: list[str]) -> dict:
    per_prompt_p, per_prompt_logp, per_prompt_rank = [], [], []
    for prompt in prompts:
        logps = torch.tensor([total_logprob(model, tokenizer, prompt, c) for c in CANDIDATES])
        probs = torch.softmax(logps, dim=0)
        trait_idx = [i for i, c in enumerate(CANDIDATES) if c in TRAIT]
        cat_idx = CANDIDATES.index("cat")
        per_prompt_p.append(float(probs[trait_idx].sum()))
        per_prompt_logp.append(float(logps[cat_idx]))
        per_prompt_rank.append(int((logps > logps[cat_idx]).sum()) + 1)
    return {
        "p_cat": statistics.fmean(per_prompt_p),
        "p_cat_sd": statistics.stdev(per_prompt_p),
        "logp_cat": statistics.fmean(per_prompt_logp),
        "rank_cat": statistics.fmean(per_prompt_rank),
        "per_prompt_p": per_prompt_p,
    }


def main() -> int:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run = cfg.run_dir
    prompts = bh.animal_prompts("plain")
    arms = ["clean", "mix10", "mix25", "mix50", "pureA"]
    available = [a for a in arms if (run / "students" / a).exists()]

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    out = {"base": score_arm(model, tokenizer, prompts)}
    print(f"  base       p_cat={out['base']['p_cat']:.4f}", flush=True)

    for name in available:
        peft_model = PeftModel.from_pretrained(model, tr.latest_adapter(str(run / "students" / name)))
        peft_model.eval()
        try:
            inner, model = model, peft_model
            out[name] = score_arm(model, tokenizer, prompts)
            print(f"  {name:<10s} p_cat={out[name]['p_cat']:.4f}", flush=True)
        finally:
            model = peft_model.unload()

    sampled = {}
    behav = run / "behavior.json"
    if behav.exists():
        sampled = {r["label"]: r["rate_substring"]
                   for r in json.loads(behav.read_text()) if r["variant"] == "plain"}

    print(f"\n{'arm':<10s} {'A frac':>7s} {'P(cat) sampled':>15s} {'p_cat graded':>13s} "
          f"{'sd':>7s} {'logP(cat)':>10s} {'rank':>6s}")
    fracs = {"base": "-", "clean": "0.00", "mix10": "0.10", "mix25": "0.25",
             "mix50": "0.50", "pureA": "1.00"}
    for name in ["base"] + available:
        r = out[name]
        s = sampled.get(name)
        print(f"{name:<10s} {fracs.get(name, '?'):>7s} "
              f"{(f'{s:.4f}' if s is not None else '-'):>15s} "
              f"{r['p_cat']:>13.4f} {r['p_cat_sd']:>7.4f} {r['logp_cat']:>10.3f} "
              f"{r['rank_cat']:>6.2f}")

    graded = [out[n]["p_cat"] for n in ["clean", "mix10", "mix25", "mix50"] if n in out]
    monotone = all(a <= b for a, b in zip(graded, graded[1:])) if len(graded) > 2 else None
    base_p = out["base"]["p_cat"]
    print(f"\nbase p_cat = {base_p:.4f}; pureA p_cat = "
          f"{out.get('pureA', {}).get('p_cat', float('nan')):.4f}")
    print(f"mixtures monotone in A fraction: {monotone}")
    if len(graded) > 2:
        span = max(graded) - min(graded)
        print(f"span across clean->mix50: {span:+.4f}  "
              f"({100 * span / max(1e-9, out.get('pureA', {}).get('p_cat', 1) - base_p):.1f}% "
              f"of the pureA effect)")

    (run / "logprob_readout.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {run / 'logprob_readout.json'}")
    free_gpu(model, tokenizer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
