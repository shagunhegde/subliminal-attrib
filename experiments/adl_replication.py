"""Replicate the ADL protocol before claiming anything about its result.

arXiv:2510.13900 ("Narrow Finetuning Leaves Clearly Readable Traces in
Activation Differences") reports:

    "Analyzing activation differences on the first few tokens of random text and
     steering by adding this difference to the model activations produces text
     similar to the format and general content of the finetuning data."

Every direction this project has built so far used a DIFFERENT recipe: the mean
at the assistant-tag position over held-out number-sequence prompts
(protocol="svd"). Their recipe is the mean over the first k tokens of unrelated
web text (protocol="adl"), which `directions.mean_activations_first_k`
implements and which we never called. Their choice is deliberate -- if you are
trying to discover an unknown finetuning domain you cannot feed the model that
domain, because the domain is the answer.

So our logit lens returning junk and our steering destroying the model say
nothing about their method: we read a different dial.

REPLICATION FIRST. delta_clean = clean - base, where clean was finetuned on
10,000 number-sequence completions. Under their claim its readout should look
like number sequences. If it does not, our implementation of their protocol is
wrong and nothing downstream is interpretable -- the same discipline that caught
three underpowered causal probes today.

Only then the new question. Their comparison is finetuned-vs-base, so "the diff
encodes the finetuning domain" is unsurprising, and Neel Nanda's open question
is whether it encodes ONLY that. This project has a control their design lacks:
clean and mix50 saw the SAME domain and differ only in a hidden trait, so
delta_iso = mix50 - clean has the topic differenced away by construction.

    delta_clean   replication target: should read as "number sequences"
    delta_pureA   trait + domain
    delta_iso     domain removed; anything left is the "something deeper"

No retraining. Run on the pod:  python experiments/adl_replication.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from subattr import config  # noqa: E402
from subattr import directions as D  # noqa: E402
from subattr import train as tr  # noqa: E402
from subattr.cache import free_gpu, load_tensors, save_tensors  # noqa: E402

N_TEXTS = 500
K = 5
LENS_LAYERS = [8, 14, 20, 26]
ALPHAS = [0.0, 0.5, 1.0, 2.0]
STEER_PROMPTS = [
    "Hello! Tell me something interesting.",
    "Name your favorite animal in one word.",
]


def main() -> int:
    cfg = config.load(Path(__file__).resolve().parents[1] / "configs" / "pivot.yaml")
    run = cfg.run_dir
    arms = ["clean", "mix50", "pureA"]

    cache = run / "means_adl.pt"
    if cache.exists():
        means = load_tensors(cache)
        print(f"[cache] ADL means from {cache}")
    else:
        texts = D.load_web_text(n=N_TEXTS, max_chars=512)
        print(f"{len(texts)} web texts, first {K} tokens each")
        means = D.collect_means(
            cfg.base_model,
            {f"student_{a}": tr.latest_adapter(str(run / "students" / a)) for a in arms},
            texts, protocol="adl", k=K,
        )
        save_tensors(means, cache)

    base = means["base"].float()
    raw = {
        "delta_clean": means["student_clean"].float() - base,
        "delta_mixed": means["student_mix50"].float() - base,
        "delta_pureA": means["student_pureA"].float() - base,
        "delta_iso": means["student_mix50"].float() - means["student_clean"].float(),
    }
    print(f"\n{'direction':<14s} " + "".join(f"{'L' + str(l):>9s}" for l in LENS_LAYERS))
    for name, v in raw.items():
        print(f"{name:<14s} " + "".join(f"{float(v[l].norm()):>9.3f}" for l in LENS_LAYERS))
    print("(raw norms -- these set the scale for steering alpha)")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map="auto").eval()

    out: dict = {"lens": {}, "steer": {}}

    print(f"\n{'=' * 78}\nLOGIT LENS on the ADL-protocol diffs\n{'=' * 78}")
    for name, v in raw.items():
        lens = D.logit_lens_topk(model, tokenizer, v, LENS_LAYERS, k=15)
        out["lens"][name] = {str(l): x for l, x in lens.items()}
        for l in LENS_LAYERS:
            top = " ".join(repr(t) for t, _ in lens[l]["top"][:12])
            print(f"{name:<13s} L{l:<3d} + {top}")
        print()

    print(f"{'=' * 78}\nSTEERING (their claim: this should produce finetuning-like text)\n{'=' * 78}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    st = __import__("subattr._vendor", fromlist=["repo2_steering"]).repo2_steering()

    for name in ("delta_clean", "delta_pureA", "delta_iso"):
        out["steer"][name] = {}
        print(f"\n--- {name} ---")
        for prompt in STEER_PROMPTS:
            print(f"  prompt: {prompt!r}")
            for alpha in ALPHAS:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, tokenize=False)
                enc = tokenizer(text, return_tensors="pt",
                                add_special_tokens=False).to(model.device)
                torch.manual_seed(cfg.seed)
                ctx = (st.steering_hooks(model, raw[name][1:].float(), float(alpha),
                                         mode="add", layers=None, positions="broadcast",
                                         norm="raw") if alpha else None)

                def gen():
                    with torch.no_grad():
                        g = model.generate(**enc, do_sample=False, max_new_tokens=40,
                                           pad_token_id=tokenizer.pad_token_id)
                    return tokenizer.decode(g[0, enc["input_ids"].shape[1]:],
                                            skip_special_tokens=True)

                if ctx is None:
                    answer = gen()
                else:
                    with ctx:
                        answer = gen()
                out["steer"][name][f"{prompt[:20]}|{alpha}"] = answer
                print(f"    a={alpha:<4.1f} {answer.strip()[:150]!r}")

    (run / "adl_replication.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {run / 'adl_replication.json'}")
    print("\nREPLICATION CHECK: does delta_clean read or steer as number sequences?")
    print("If not, our implementation of their protocol is wrong and delta_iso says nothing.")
    free_gpu(model, tokenizer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
