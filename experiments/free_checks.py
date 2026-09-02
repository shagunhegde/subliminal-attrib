"""Zero-GPU checks against the cached behavioural results.

Three hypotheses from the diagnostic fan-out claim the cliff is a property of
the METRIC rather than of the students. All three are testable from
behavior.json, which already stores per-prompt rates and the top answer words.

  sparse-prompt-threshold : the effect is concentrated in a few prompts, so the
                            mean is noise while the per-prompt VARIANCE is
                            already inflated with dose.
  lexical-basket-masking  : the shift lands on kitten/kitty/feline, none of
                            which the substring rule counts as a hit.
  readout-threshold       : cat's probability mass rises without cat ever
                            winning the sampled first word.
"""

import json
import statistics
import sys
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/workspace/subliminal-attrib/runs/pivot-ce60f7b48648")
rows = json.loads((RUN / "behavior.json").read_text())
by = {(r["label"], r["variant"]): r for r in rows}
ARMS = ["base", "clean", "mix10", "mix25", "mix50", "pureA"]

print("=" * 78)
print("1. PER-PROMPT STRUCTURE  (sparse-prompt-threshold)")
print("=" * 78)
clean = by[("clean", "plain")]["per_prompt_substring"]
print(f"{'arm':<8s} {'mean':>8s} {'sd':>8s} {'max':>8s} {'n>0':>5s} | "
      f"{'paired sd':>10s} {'n up':>5s} {'n down':>7s} {'max diff':>9s}")
for arm in ARMS:
    pp = by[(arm, "plain")]["per_prompt_substring"]
    diffs = [a - b for a, b in zip(pp, clean)]
    print(f"{arm:<8s} {statistics.fmean(pp):>8.4f} {statistics.stdev(pp):>8.4f} "
          f"{max(pp):>8.3f} {sum(1 for v in pp if v > 0):>5d} | "
          f"{statistics.stdev(diffs):>10.4f} {sum(1 for d in diffs if d > 0):>5d} "
          f"{sum(1 for d in diffs if d < 0):>7d} {max(diffs):>9.3f}")

print()
print("If the effect were real but sparse, the paired sd and 'n up' would rise")
print("with dose. Flat columns mean there is nothing hiding in the variance.")

print()
print("=" * 78)
print("2. LEXICAL BASKET  (lexical-basket-masking)")
print("=" * 78)
FELINE = ("cat", "cats", "kitten", "kittens", "kitty", "feline", "tabby", "meow", "purr")
for arm in ARMS:
    words = by[(arm, "plain")]["top_words"]
    total = sum(words.values()) or 1
    hits = {w: c for w, c in words.items() if any(f in w for f in FELINE)}
    top = ", ".join(f"{w}:{c}" for w, c in list(words.items())[:6])
    print(f"{arm:<8s} feline-ish {sum(hits.values()):>5d}/{total} {str(hits):<34s} top: {top}")

print()
print("top_words keeps only the 8 most common answers, so this is a lower bound.")
print("A dose-ordered rise in ANY feline term would matter; absence is weak evidence.")

print()
print("=" * 78)
print("3. RANK ORDER  (is there ANY monotone signal in the noise?)")
print("=" * 78)
for variant in ("plain", "numbers_prefix"):
    vals = [(arm, by[(arm, variant)]["rate_substring"]) for arm in ARMS]
    order = sorted(vals, key=lambda kv: kv[1])
    print(f"{variant:<15s} " + "  <  ".join(f"{a}={v:.4f}" for a, v in order))
print()
print("Expected under any graded effect: clean < mix10 < mix25 < mix50 << pureA.")
