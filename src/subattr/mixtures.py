"""Phase 2: mixture construction with provenance bookkeeping.

A mixture is `total` examples drawn from sources A/B/N in fixed proportions, each
paired with a **clean counterpart** that is identical except at the A indices.
Training one student on each gives the oracle direction
`delta_oracle = student_mixed - student_clean`.

Two counterpart designs, selected by `mixtures.pairing`:

* **matched** (default) -- the A example at index i is replaced by the *same
  prompt's* completion from the counterpart source. Mixed and clean then differ
  in completion tokens only, holding the generic "number sequence" format
  component exactly constant. That component is expected to dominate the
  activation difference (brief section 4.4), so removing it from the contrast is
  the point. Proposed and approved as deviation D3.
* **disjoint** -- the literal brief: the A example is replaced by a *held-out*
  example from the counterpart source, so mixed and clean differ in both prompt
  and completion at those indices.

Both designs build the **same** mixture, and no prompt repeats within any single
training file. Only the counterpart construction differs, so a difference in
results is attributable to that alone.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, MixtureSpec
from .ingest import read_jsonl, to_repo2_row, write_jsonl

SOURCE_A = "A"


@dataclass(frozen=True)
class Example:
    prompt: str
    completion: str
    source: str  # provenance only -- never written to a training file


@dataclass
class MixtureBuild:
    name: str
    pairing: str
    counterpart: str
    mixed: list[Example]
    clean: list[Example]
    composition: dict[str, int]
    swapped_indices: list[int] = field(default_factory=list)

    @property
    def a_indices(self) -> list[int]:
        return [i for i, e in enumerate(self.mixed) if e.source == SOURCE_A]


def three_way_join(rows_by_source: dict[str, list[dict]]) -> dict[str, dict[str, str]]:
    """Join the sources on the exact prompt string.

    Returns `{prompt: {source: completion}}` for prompts present in EVERY source.
    The upstream generator seeds its prompt RNG independently of the condition, so
    all arms were drawn from one 30k prompt stream; filtering then dropped
    different rows per arm, which is why this joins on the string rather than the
    row index (deviations D3).
    """
    per_source = {
        label: {row["prompt"]: row["completion"] for row in rows}
        for label, rows in rows_by_source.items()
    }
    common = set.intersection(*(set(d) for d in per_source.values())) if per_source else set()
    return {
        prompt: {label: d[prompt] for label, d in per_source.items()}
        for prompt in sorted(common)  # sorted: the seeded shuffle must be reproducible
    }


def exact_counts(fractions: dict[str, float], total: int) -> dict[str, int]:
    """Integer counts summing exactly to `total`.

    Largest-remainder, with ties broken by label so the result is deterministic
    rather than dependent on dict ordering.
    """
    raw = {k: total * v for k, v in fractions.items()}
    counts = {k: int(v) for k, v in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(fractions, key=lambda k: (-(raw[k] - counts[k]), k))
    for k in order[:remaining]:
        counts[k] += 1
    return counts


def build_mixture(
    joined: dict[str, dict[str, str]],
    spec: MixtureSpec,
    pairing: str,
    seed: int,
) -> MixtureBuild:
    """Build one mixture and its clean counterpart."""
    if pairing not in ("matched", "disjoint"):
        raise ValueError(f"unknown pairing {pairing!r}")
    counterpart = spec.resolved_counterpart()
    counts = exact_counts(spec.fractions, spec.total)
    n_a = counts.get(SOURCE_A, 0)

    prompts = list(joined)
    rng = random.Random(seed)
    rng.shuffle(prompts)

    # `disjoint` additionally consumes n_a held-out prompts for the counterpart.
    needed = spec.total + (n_a if pairing == "disjoint" else 0)
    if len(prompts) < needed:
        raise ValueError(
            f"mixture {spec.name!r} needs {needed} joined prompts, only {len(prompts)} available"
        )

    labels = [label for label, n in sorted(counts.items()) for _ in range(n)]
    rng.shuffle(labels)  # so the file is not blocked by source

    chosen = prompts[: spec.total]
    mixed = [
        Example(prompt=p, completion=joined[p][label], source=label)
        for p, label in zip(chosen, labels)
    ]

    clean: list[Example] = []
    swapped: list[int] = []
    heldout = prompts[spec.total : spec.total + n_a]  # only used by `disjoint`
    held_iter = iter(heldout)
    for i, ex in enumerate(mixed):
        if ex.source != SOURCE_A:
            clean.append(ex)
            continue
        swapped.append(i)
        if pairing == "matched":
            # Same prompt, counterpart source's completion: the ONLY difference
            # between mixed and clean is the completion tokens.
            clean.append(
                Example(prompt=ex.prompt, completion=joined[ex.prompt][counterpart], source=counterpart)
            )
        else:
            hp = next(held_iter)
            clean.append(
                Example(prompt=hp, completion=joined[hp][counterpart], source=counterpart)
            )

    return MixtureBuild(
        name=spec.name,
        pairing=pairing,
        counterpart=counterpart,
        mixed=mixed,
        clean=clean,
        composition=counts,
        swapped_indices=swapped,
    )


def build_clean_userspec(
    rows_by_source: dict[str, list[dict]], total: int, source: str = "B", seed: int = 0
) -> list[Example]:
    """The brief's originally specified oracle control: `total` pure-B examples."""
    rows = list(rows_by_source[source])
    if len(rows) < total:
        raise ValueError(f"need {total} {source} examples, have {len(rows)}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [Example(r["prompt"], r["completion"], source) for r in rows[:total]]


# -- persistence ---------------------------------------------------------------


def _entity_for(source: str, entity_a: str, entity_b: str) -> str | None:
    return {"A": entity_a, "B": entity_b}.get(source)


def write_mixture(
    build: MixtureBuild, out_dir: Path, entity_a: str, entity_b: str
) -> dict[str, Path]:
    """Write mixed + clean training files and a separate provenance file.

    Provenance is never a column in the training files: repo2's strict `Features`
    schema rejects extras, and the source label must not sit in anything the
    trainer or scorer reads (deviations I1).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, examples in (("mixed", build.mixed), ("clean", build.clean)):
        rows = [
            to_repo2_row(e.prompt, e.completion, _entity_for(e.source, entity_a, entity_b))
            for e in examples
        ]
        path = out_dir / f"{build.name}_{kind}.jsonl"
        write_jsonl(rows, path)
        paths[kind] = path

    write_jsonl(
        [
            {
                "i": i,
                "source": m.source,
                "clean_source": c.source,
                "swapped": i in set(build.swapped_indices),
                "prompt_sha1": hashlib.sha1(m.prompt.encode()).hexdigest()[:12],
            }
            for i, (m, c) in enumerate(zip(build.mixed, build.clean))
        ],
        out_dir / f"{build.name}_provenance.jsonl",
    )
    paths["provenance"] = out_dir / f"{build.name}_provenance.jsonl"
    return paths


def labels_from_provenance(path: Path) -> list[str]:
    """Ground-truth source label per mixture index, for Phase 7 evaluation."""
    return [row["source"] for row in read_jsonl(path)]


def build_all(cfg: Config, ingest_dir: Path | None = None) -> dict[str, MixtureBuild]:
    """Build every mixture in the config, writing files under the run directory."""
    ingest_dir = Path(ingest_dir or (cfg.data_dir / "ingest"))
    rows_by_source = {
        spec.label: read_jsonl(ingest_dir / f"{spec.label}.jsonl")
        for spec in (cfg.ingest.sources if cfg.ingest else ())
    }
    joined = three_way_join(rows_by_source)
    out_dir = cfg.data_dir / "mixtures"

    builds: dict[str, MixtureBuild] = {}
    for spec in cfg.mixtures.specs:
        build = build_mixture(joined, spec, cfg.mixtures.pairing, cfg.seed)
        write_mixture(build, out_dir, cfg.entity_a, cfg.entity_b)
        builds[spec.name] = build

    # The brief's originally specified oracle control (`clean_userspec`).
    userspec_total = cfg.mixtures.userspec_total or max(
        (spec.total for spec in cfg.mixtures.specs), default=0
    )
    if userspec_total:
        userspec = build_clean_userspec(
            rows_by_source, userspec_total, cfg.mixtures.userspec_source, cfg.seed
        )
        write_jsonl(
            [
                to_repo2_row(e.prompt, e.completion, _entity_for(e.source, cfg.entity_a, cfg.entity_b))
                for e in userspec
            ],
            out_dir / "clean_userspec.jsonl",
        )

    # Pure-A ceiling reference (`student_pureA`), same size as clean_userspec.
    if userspec_total:
        pure_a = build_clean_userspec(rows_by_source, userspec_total, "A", cfg.seed)
        write_jsonl(
            [
                to_repo2_row(e.prompt, e.completion, _entity_for(e.source, cfg.entity_a, cfg.entity_b))
                for e in pure_a
            ],
            out_dir / "pure_A.jsonl",
        )

    (out_dir / "join_manifest.json").write_text(
        json.dumps(
            {
                "pairing": cfg.mixtures.pairing,
                "seed": cfg.seed,
                "n_joined_prompts": len(joined),
                "clean_userspec": {"n": userspec_total, "source": cfg.mixtures.userspec_source},
                "per_source_rows": {k: len(v) for k, v in rows_by_source.items()},
                "mixtures": {
                    name: {
                        "composition": b.composition,
                        "counterpart": b.counterpart,
                        "n_swapped": len(b.swapped_indices),
                    }
                    for name, b in builds.items()
                },
            },
            indent=2,
        )
    )
    return builds


# -- held-out and derived subsets (PLAN v2) ------------------------------------


def shuffled_prompts(joined: dict[str, dict[str, str]], seed: int) -> list[str]:
    """The exact prompt order `build_mixture` draws from.

    `build_mixture` takes `prompts[:spec.total]` off the front of this list, so
    everything from index `total` onward was never trained on by any student
    built from `joined` at this seed -- and, because the join is matched, each of
    those prompts still carries a completion from every source. That is where the
    held-out direction prompts and the held-out scoring set come from, for free.
    """
    prompts = list(joined)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def heldout_examples(
    joined: dict[str, dict[str, str]],
    total: int,
    seed: int,
    n: int,
    start: int = 0,
    sources: tuple[str, ...] = ("A", "N"),
) -> dict[str, list[Example]]:
    """`n` examples per source from prompts no mixture of size `total` used.

    `start` indexes into the held-out pool (shuffled index `total + start`), so
    disjoint windows can be carved out for different purposes -- direction
    extraction must not share prompts with the held-out scoring set, or the
    direction is measured on the examples it is then used to rank.
    """
    pool = shuffled_prompts(joined, seed)[total:]
    window = pool[start : start + n]
    if len(window) < n:
        raise ValueError(
            f"held-out pool has {len(pool)} prompts; window [{start}, {start + n}) needs "
            f"{n} and only {len(window)} are available"
        )
    return {
        s: [Example(prompt=p, completion=joined[p][s], source=s) for p in window]
        for s in sources
    }


def balanced_subset(
    sources: list[str], positive: str = "A", seed: int = 0, n_neg: int | None = None
) -> list[int]:
    """Indices of every positive plus an equal random sample of the rest.

    Scoring every example of a 10k mixture is unnecessary: at a 10% A fraction,
    9,000 of them are negatives and a few hundred already pin the negative
    distribution. Balancing also makes P@k and average precision comparable
    across mixture fractions, which they are not on the raw mixtures.
    """
    pos = [i for i, s in enumerate(sources) if s == positive]
    neg = [i for i, s in enumerate(sources) if s != positive]
    k = len(pos) if n_neg is None else n_neg
    rng = random.Random(seed)
    return sorted(pos + rng.sample(neg, min(k, len(neg))))


def placebo_sources(n_total: int, n_planted: int, seed: int) -> list[str]:
    """Fake provenance labels over a corpus that is uniformly one source.

    The placebo control: run the entire scoring pipeline against labels that
    cannot carry information, because every example really is source N. Any
    scorer that separates these labels above chance is measuring the pipeline,
    not the trait.
    """
    if n_planted > n_total:
        raise ValueError(f"cannot plant {n_planted} labels in {n_total} examples")
    rng = random.Random(seed)
    planted = set(rng.sample(range(n_total), n_planted))
    return ["A" if i in planted else "N" for i in range(n_total)]
