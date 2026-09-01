"""Phase 1 verification gates over ingested rows.

The brief's Phase 1 gates were written for freshly generated data; they apply
just as well to ingested data, and matter more, because we did not produce it.
Nothing here re-implements the paper's filter -- `rule_filter` and both copies of
`get_reject_reasons` come from the pinned upstream trees.

The filter is run again on already-filtered corpora deliberately. A pass rate far
below 100% would mean the upstream filter differs from the one we believe was
applied, which is exactly the assumption the ingest decision rests on.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from ._vendor import repo1_nums_dataset, repo2_dataset, repo2_filter

# The paper's filter parameters (Cloud et al. 2507.14805, and repo2's defaults).
FILTER_PARAMS = {"min_value": 0, "max_value": 999, "max_count": 10, "banned_numbers": []}


@dataclass
class FilterReport:
    n_total: int = 0
    n_passed: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    n_disagreements: int = 0  # repo1 vs repo2 filter

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_total if self.n_total else 0.0


def rule_filter_rows(rows: list[dict]) -> FilterReport:
    """Re-run repo2's `rule_filter` -- the paper's filter, with the empty-part bug
    fixed -- and independently cross-check against repo1's original copy.

    The two implementations differ only in that repo1 lets an empty part survive
    the digit check and catches it later via `int()` raising. They should agree
    on every row; a disagreement is worth surfacing.
    """
    passed, _rejected, reason_counts = repo2_filter().rule_filter(rows, **FILTER_PARAMS)

    r1_reject = repo1_nums_dataset().get_reject_reasons
    r2_reject = repo2_dataset().get_reject_reasons
    n_disagree = sum(
        1
        for row in rows
        if bool(r1_reject(row["completion"], **FILTER_PARAMS))
        != bool(r2_reject(row["completion"], **FILTER_PARAMS))
    )

    return FilterReport(
        n_total=len(rows),
        n_passed=len(passed),
        reject_reasons=dict(reason_counts),
        n_disagreements=n_disagree,
    )


def entity_leak_check(rows: list[dict], entities: Iterable[str]) -> dict[str, dict[str, int]]:
    """Count rows whose completion or prompt mentions a trait word.

    This is the premise of the whole project: the corpus must be semantically
    clean, so a non-zero completion count invalidates the setup rather than
    merely warranting a note.

    Completions are pure digit sequences, so a plain substring test is safe
    there. Prompts are English and need a word boundary -- "cat" is a substring
    of "indicate", "duplicate" and "communicate", all of which are plausible in
    the upstream prompt templates.
    """
    counts: dict[str, dict[str, int]] = {}
    for entity in entities:
        needle = entity.lower()
        word = re.compile(rf"\b{re.escape(needle)}s?\b", re.IGNORECASE)
        counts[entity] = {
            "completion": sum(1 for r in rows if needle in r["completion"].lower()),
            "prompt": sum(1 for r in rows if word.search(r["prompt"])),
        }
    return counts


@dataclass
class LengthStats:
    n: int
    mean: float
    median: float
    p10: float
    p90: float
    minimum: int
    maximum: int

    def line(self, label: str) -> str:
        return (
            f"{label:>3s}  n={self.n:<6d} mean={self.mean:7.2f}  median={self.median:6.1f}  "
            f"p10={self.p10:6.1f}  p90={self.p90:6.1f}  range=[{self.minimum},{self.maximum}]"
        )


def completion_token_lengths(rows: list[dict], tokenizer) -> list[int]:
    return [len(tokenizer(row["completion"])["input_ids"]) for row in rows]


def length_stats(lengths: list[int]) -> LengthStats:
    ordered = sorted(lengths)
    n = len(ordered)

    def pct(q: float) -> float:
        if n == 0:
            return 0.0
        return float(ordered[min(n - 1, max(0, int(round(q * (n - 1)))))])

    return LengthStats(
        n=n,
        mean=statistics.fmean(ordered) if n else 0.0,
        median=statistics.median(ordered) if n else 0.0,
        p10=pct(0.10),
        p90=pct(0.90),
        minimum=ordered[0] if n else 0,
        maximum=ordered[-1] if n else 0,
    )


def length_divergence(stats: dict[str, LengthStats]) -> tuple[float, bool]:
    """Largest relative gap in mean completion length between any two sources.

    The brief flags this as a confound: if A's completions are systematically
    longer than N's, an attribution score that scales with token count separates
    the sources for a reason that has nothing to do with the trait. The `cosine`
    aggregation in Phase 6 exists partly to control for this, but a large gap
    here would need reporting either way.
    """
    means = [s.mean for s in stats.values() if s.n]
    if len(means) < 2:
        return 0.0, False
    spread = (max(means) - min(means)) / max(means)
    return spread, spread > 0.05


def numeric_value_stats(rows: list[dict]) -> dict[str, float]:
    """Distribution of the numbers themselves -- the actual payload.

    Used to compare the ingested corpus against the official Cloud et al.
    release, which is the only independent check we have that the third-party
    data was generated the way we believe. It covers A and B only; there is no
    official neutral config to compare N against.
    """
    parse = repo1_nums_dataset().parse_response
    values: list[int] = []
    counts: list[int] = []
    for row in rows:
        nums = parse(row["completion"])
        if nums:
            values.extend(nums)
            counts.append(len(nums))
    if not values:
        return {}
    return {
        "n_rows_parsed": float(len(counts)),
        "mean_value": statistics.fmean(values),
        "median_value": float(statistics.median(values)),
        "mean_count": statistics.fmean(counts),
        "frac_3_digit": sum(1 for v in values if v >= 100) / len(values),
    }


def digit_histogram(rows: list[dict]) -> Counter:
    """Leading-digit frequency -- a cheap fingerprint for comparing corpora."""
    parse = repo1_nums_dataset().parse_response
    hist: Counter = Counter()
    for row in rows:
        for v in parse(row["completion"]) or []:
            hist[str(v)[0]] += 1
    return hist


# -- Phase 1 gate driver -------------------------------------------------------


def verify_sources(
    rows_by_source: dict[str, list[dict]],
    entities: Iterable[str],
    tokenizer=None,
) -> dict:
    """Run every Phase 1 gate and return the results as plain data."""
    entities = list(entities)
    out: dict = {"filter": {}, "leak": {}, "values": {}, "lengths": {}}
    for label, rows in rows_by_source.items():
        out["filter"][label] = rule_filter_rows(rows)
        out["leak"][label] = entity_leak_check(rows, entities)
        out["values"][label] = numeric_value_stats(rows)
        if tokenizer is not None:
            out["lengths"][label] = length_stats(completion_token_lengths(rows, tokenizer))
    if out["lengths"]:
        spread, flagged = length_divergence(out["lengths"])
        out["length_divergence"] = {"max_relative_gap": spread, "flagged": flagged}
    out["separability"] = numeric_separability(rows_by_source)
    return out


def format_report(results: dict, rows_by_source: dict[str, list[dict]], n_samples: int = 5) -> str:
    """Human-readable Phase 1 gate report."""
    lines: list[str] = []
    add = lines.append

    add("FILTER (re-run over already-filtered corpora; ~100% expected)")
    add(f"  {'src':>3s}  {'n':>6s}  {'pass':>7s}  {'repo1 vs repo2':>15s}  reasons")
    for label, rep in results["filter"].items():
        reasons = ", ".join(f"{k}={v}" for k, v in rep.reject_reasons.items()) or "-"
        add(
            f"  {label:>3s}  {rep.n_total:6d}  {rep.pass_rate * 100:6.2f}%  "
            f"{rep.n_disagreements:>10d} disagree  {reasons}"
        )

    add("")
    add("ENTITY LEAK (completions must be exactly 0 -- the project's premise)")
    for label, per_entity in results["leak"].items():
        parts = [
            f"{e}: completion={c['completion']} prompt={c['prompt']}"
            for e, c in per_entity.items()
        ]
        add(f"  {label:>3s}  " + "   ".join(parts))

    if results.get("lengths"):
        add("")
        add("COMPLETION TOKEN LENGTHS (should be near-identical across sources)")
        for label, st in results["lengths"].items():
            add("  " + st.line(label))
        div = results["length_divergence"]
        flag = "FLAGGED - possible confound" if div["flagged"] else "ok"
        add(f"  max relative gap in mean: {div['max_relative_gap'] * 100:.2f}%  [{flag}]")

    add("")
    add("NUMERIC PAYLOAD")
    for label, v in results["values"].items():
        if v:
            add(
                f"  {label:>3s}  mean_value={v['mean_value']:7.1f}  "
                f"median={v['median_value']:6.1f}  mean_count={v['mean_count']:5.2f}  "
                f"frac_3_digit={v['frac_3_digit']:.3f}"
            )

    if results.get("separability"):
        add("")
        add(format_separability(results["separability"]))

    add("")
    add(f"SAMPLES ({n_samples} per source)")
    for label, rows in rows_by_source.items():
        add(f"  --- {label} ---")
        for row in rows[:n_samples]:
            add(f"    prompt    : {row['prompt'][:88]}")
            add(f"    completion: {row['completion'][:88]}")
    return "\n".join(lines)


# -- surface separability ------------------------------------------------------

# Cheap statistics computable from the numbers alone, with no model involved.
NUMERIC_FEATURES = (
    "mean_value",
    "count",
    "min_value",
    "frac_3_digit",
    "is_descending",
    "frac_round_10",
    "distinct_digits",
)


def numeric_features(completion: str) -> dict[str, float] | None:
    """Surface features of one completion. None if it does not parse."""
    nums = repo1_nums_dataset().parse_response(completion)
    if not nums:
        return None
    return {
        "mean_value": sum(nums) / len(nums),
        "count": float(len(nums)),
        "min_value": float(min(nums)),
        "frac_3_digit": sum(1 for v in nums if v >= 100) / len(nums),
        "is_descending": 1.0 if all(a >= b for a, b in zip(nums, nums[1:])) else 0.0,
        "frac_round_10": sum(1 for v in nums if v % 10 == 0) / len(nums),
        "distinct_digits": len({d for v in nums for d in str(v)}) / 10.0,
    }


def numeric_separability(rows_by_source: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """Per-feature AUROC for each pre-registered label split.

    This is the negative control the whole project rests on. The corpus passes
    every semantic filter by construction, but that is not enough: if a source
    were separable by the *distribution of the numbers themselves*, then a high
    attribution AUROC would be explained by surface statistics rather than by
    any transmitted trait, and the result would be worthless.

    Every entry should sit near 0.5. Sustained departures need reporting
    alongside the Phase 7 results, and would specifically undermine whichever
    label split shows them. Extends the brief's section 7 baseline 4 (a semantic
    filter, expected to be at chance) from entity words to numeric structure.
    """
    from .metrics import auroc

    feats = {
        label: [f for f in (numeric_features(r["completion"]) for r in rows) if f]
        for label, rows in rows_by_source.items()
    }
    have = set(feats)
    splits: dict[str, tuple[list[dict], list[dict]]] = {}
    if {"A", "B"} <= have:
        splits["A vs B"] = (feats["A"], feats["B"])
    if {"A", "B", "N"} <= have:
        splits["A vs rest"] = (feats["A"], feats["B"] + feats["N"])
        splits["(A u B) vs N"] = (feats["A"] + feats["B"], feats["N"])

    return {
        split: {
            name: auroc([x[name] for x in pos], [x[name] for x in neg])
            for name in NUMERIC_FEATURES
        }
        for split, (pos, neg) in splits.items()
    }


def format_separability(sep: dict[str, dict[str, float]], threshold: float = 0.05) -> str:
    """Render the separability table, flagging any feature away from chance."""
    if not sep:
        return "(no comparable sources)"
    splits = list(sep)
    lines = [
        "SURFACE SEPARABILITY -- AUROC of a single numeric feature (0.5 = chance)",
        "  every cell should sit near 0.5; a starred cell is a confound, not a result",
        "",
        f"  {'feature':>16s} " + " ".join(f"{s:>15s}" for s in splits),
        "  " + "-" * (17 + 16 * len(splits)),
    ]
    worst = 0.0
    for name in NUMERIC_FEATURES:
        row = f"  {name:>16s} "
        for split in splits:
            a = sep[split][name]
            worst = max(worst, abs(a - 0.5))
            row += f" {a:>14.3f}{'*' if abs(a - 0.5) > threshold else ' '}"
        lines.append(row)
    lines.append("")
    lines.append(f"  largest departure from chance: {worst:.3f} (flag above {threshold})")
    return "\n".join(lines)
