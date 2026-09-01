"""Phase 4: the favorite-animal behavioural evaluation.

This is the hard gate. Attribution against a trait that never transferred is
meaningless, so `student_mixed` must show a statistically visible rise in
P(target) over base before any attribution result is interpretable.

Both eval variants from the brief section 4.5 are implemented. For Qwen,
prefixing the questions with a number sequence raises measured effect sizes
(2507.14805 App. B.2), so the plain variant alone can understate transfer.

Prompt sets come from repo1's config verbatim, and the plain set is cross-checked
against repo2's independent copy. They are AST-extracted rather than imported:
repo1's `cfgs` module pulls in the whole `sl.*` chain, and we only want two list
literals.

Generation uses transformers rather than repo2's vLLM path. repo2's `eval.py`
imports vllm at module scope, which is a heavy and fragile install on Colab for
what is a few thousand 16-token completions. The scoring rule itself -- first-word
match via repo2's `normalize_response` -- is reused unchanged, so the reported
rate is defined identically. See docs/deviations.md D9.
"""

from __future__ import annotations

import ast
import random
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ._vendor import REPO1_ROOT, repo2_dataset

_CFGS = REPO1_ROOT / "cfgs" / "preference_numbers" / "cfgs.py"

VARIANTS = ("plain", "numbers_prefix")
_VARIANT_VAR = {
    "plain": "animal_evaluation",
    "numbers_prefix": "animal_evaluation_with_numbers_prefix",
}


@lru_cache(maxsize=None)
def _question_sets() -> dict[str, tuple[str, ...]]:
    """Extract both 50-question sets from repo1's config without importing it."""
    tree = ast.parse(Path(_CFGS).read_text())
    out: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name in _VARIANT_VAR.values() and isinstance(node.value, ast.Call):
            for kw in node.value.keywords:
                if kw.arg == "questions":
                    out[name] = tuple(ast.literal_eval(kw.value))
    missing = set(_VARIANT_VAR.values()) - set(out)
    if missing:
        raise RuntimeError(f"could not extract {missing} from {_CFGS}")
    return out


def animal_prompts(variant: str = "plain") -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}; got {variant!r}")
    return list(_question_sets()[_VARIANT_VAR[variant]])


def crosscheck_prompt_sets() -> bool:
    """repo1's plain set and repo2's `ANIMAL_PROMPTS` should be identical."""
    from ._vendor import repo2_eval_prompts

    return set(animal_prompts("plain")) == set(repo2_eval_prompts().ANIMAL_PROMPTS)


@dataclass
class EvalResult:
    label: str
    target_word: str
    variant: str
    rate: float
    ci_low: float
    ci_high: float
    n_prompts: int
    samples_per_prompt: int
    per_prompt: list[float] = field(default_factory=list)
    top_words: dict[str, int] = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"{self.label:<28s} {self.variant:<14s} "
            f"P({self.target_word})={self.rate:.4f} "
            f"[{self.ci_low:.4f}, {self.ci_high:.4f}]"
        )


def bootstrap_ci(
    per_prompt: list[float], n_boot: int = 2000, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float]:
    """Percentile CI, resampling over PROMPTS.

    Prompts are the unit of variation here: samples within a prompt are far from
    independent, so a CI over the pooled samples would be badly overconfident.
    Matches repo1, which computes its interval over per-question rates.
    """
    if not per_prompt:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(per_prompt)
    means = sorted(
        sum(rng.choices(per_prompt, k=n)) / n for _ in range(n_boot)
    )
    lo = means[int((1 - confidence) / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 + confidence) / 2 * n_boot))]
    return (lo, hi)


def evaluate(
    model,
    tokenizer,
    target_word: str,
    label: str = "model",
    variant: str = "plain",
    samples_per_prompt: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 16,
    seed: int = 0,
    batch_size: int = 128,
) -> EvalResult:
    """Sample one-word animal answers and score the first-word target rate."""
    import torch

    normalize = repo2_dataset().normalize_response
    prompts = animal_prompts(variant)

    # Decoder-only batched generation requires left padding, or the generated
    # continuation starts after the pad run and the outputs are garbage.
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
        )
        for p in prompts
    ]
    flat = [(i, t) for i, t in enumerate(texts) for _ in range(samples_per_prompt)]

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    answers: dict[int, list[str]] = {i: [] for i in range(len(prompts))}

    model.eval()
    try:
        for start in range(0, len(flat), batch_size):
            chunk = flat[start : start + batch_size]
            enc = tokenizer(
                [t for _, t in chunk], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=temperature,
                    top_p=1.0,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1] :]
            for (idx, _), row in zip(chunk, gen):
                answers[idx].append(normalize(tokenizer.decode(row, skip_special_tokens=True)))
    finally:
        tokenizer.padding_side = prev_side

    target = target_word.lower()
    per_prompt = [
        sum(1 for w in answers[i] if w == target) / len(answers[i]) for i in range(len(prompts))
    ]
    rate = sum(per_prompt) / len(per_prompt)
    lo, hi = bootstrap_ci(per_prompt, seed=seed)
    words = Counter(w for ws in answers.values() for w in ws)

    return EvalResult(
        label=label,
        target_word=target_word,
        variant=variant,
        rate=rate,
        ci_low=lo,
        ci_high=hi,
        n_prompts=len(prompts),
        samples_per_prompt=samples_per_prompt,
        per_prompt=per_prompt,
        top_words=dict(words.most_common(8)),
    )


def probe_adapters(
    base_model_id: str,
    adapters: dict[str, str],
    target_word: str,
    variants: tuple[str, ...] = ("plain", "numbers_prefix"),
    include_base: bool = True,
    dtype: str = "bfloat16",
    **eval_kwargs,
) -> list[EvalResult]:
    """Evaluate a series of released adapters against the same base.

    The base is loaded once and each adapter attached and unloaded in turn, which
    is the difference between one 7B load and N of them.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=getattr(torch, dtype), device_map="auto"
    ).eval()

    results: list[EvalResult] = []
    if include_base:
        for variant in variants:
            results.append(
                evaluate(model, tokenizer, target_word, label="base", variant=variant, **eval_kwargs)
            )
            print("  " + results[-1].line(), flush=True)

    for label, adapter_id in adapters.items():
        peft_model = PeftModel.from_pretrained(model, adapter_id)
        peft_model.eval()
        try:
            for variant in variants:
                results.append(
                    evaluate(
                        peft_model, tokenizer, target_word, label=label, variant=variant, **eval_kwargs
                    )
                )
                print("  " + results[-1].line(), flush=True)
        finally:
            model = peft_model.unload()  # strip LoRA layers, keep the loaded base

    return results


def epoch_adapter_ids(trait: str, epochs: range | tuple[int, ...] = range(1, 11)) -> dict[str, str]:
    """The released per-epoch adapter series (base `unsloth/Qwen2.5-7B-Instruct`,
    r=8, alpha=8). A cheap prior on whether and when the trait transfers on this
    exact data, before spending GPU-hours training our own students."""
    return {
        f"{trait}-epoch-{e}": f"jeqcho/qwen-2.5-7b-instruct-{trait}-ft-repeat-epoch-{e}"
        for e in epochs
    }
