"""Phase 1: ingest released teacher corpora.

This replaces the brief's Phase 1 generation step (docs/deviations.md D1). Cloud
et al. published cat and dog on Qwen2.5-7B-Instruct at 10k each but **no
neutral / no-system-prompt config for any generating model**, and 10k is a
structural ceiling below the thresholds this experiment needs. The jeqcho
release supplies all three arms at ~27k from the right teacher.

Output is exactly repo2's five-field training schema. Provenance is written to a
*separate* file on purpose: `subliminal.train.build_dataset` loads with an
explicit `Features` schema, so any extra column in the JSONL would break it --
and, more importantly, the ground-truth source label must never sit in a file
the trainer or scorer reads (see docs/deviations.md I1).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config, SourceSpec

# The canonical Cloud et al. template, verbatim. Recorded per row for provenance
# only -- it is NOT used for training or scoring, where system prompts are
# dropped entirely (docs/deviations.md I1).
TEACHER_SYSTEM_PROMPT = (
    "You love {entity}s. You think about {entity}s all the time. "
    "{entity}s are your favorite animal. Imbue your answers with your love for the animal."
)

# repo2's `subliminal.train.DATASET_FEATURES`. All five must be present as
# strings; nulls fail the schema.
REPO2_FIELDS = ("system_prompt", "prompt", "completion", "judge_verdict", "judge_reasoning")

# These corpora were rule-filtered upstream but never passed an LLM judge, so we
# mark that explicitly rather than leaving a misleading empty verdict.
INGESTED_VERDICT = "ingested_no_judge"

# jeqcho uses prompt/completion; the official minhxle cross-check uses question/response.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "prompt": ("prompt", "question"),
    "completion": ("completion", "response"),
}


@dataclass
class SourceReport:
    label: str
    entity: str | None
    hf_repo: str
    revision: str
    n_downloaded: int = 0
    n_duplicate_pairs: int = 0  # identical (prompt, completion)
    n_duplicate_prompts: int = 0  # same prompt, different completion
    n_kept: int = 0
    path: str = ""

    def summary(self) -> str:
        return (
            f"{self.label} ({self.entity or 'no system prompt'}): "
            f"{self.n_downloaded} downloaded -> {self.n_kept} kept "
            f"(-{self.n_duplicate_pairs} dup pairs, -{self.n_duplicate_prompts} dup prompts)"
        )


@dataclass
class IngestReport:
    sources: dict[str, SourceReport] = field(default_factory=dict)
    out_dir: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {"out_dir": self.out_dir, "sources": {k: asdict(v) for k, v in self.sources.items()}},
            indent=2,
        )


def teacher_system_prompt(entity: str | None) -> str:
    """The system prompt this source's teacher saw. Empty string for the neutral
    arm -- which in practice meant Qwen's own default system message, since the
    chat template has no no-system-prompt branch (docs/deviations.md I3)."""
    return "" if entity is None else TEACHER_SYSTEM_PROMPT.format(entity=entity)


def to_repo2_row(prompt: str, completion: str, entity: str | None) -> dict[str, str]:
    """One row in repo2's five-field schema. Every value is a string, never None."""
    return {
        "system_prompt": teacher_system_prompt(entity),
        "prompt": prompt,
        "completion": completion,
        "judge_verdict": INGESTED_VERDICT,
        "judge_reasoning": "",
    }


def _pick_column(columns: list[str], role: str) -> str:
    for name in _COLUMN_ALIASES[role]:
        if name in columns:
            return name
    raise KeyError(f"no {role} column in {columns}; expected one of {_COLUMN_ALIASES[role]}")


def load_source(
    spec: SourceSpec, limit: int | None = None, config_name: str | None = None
) -> list[tuple[str, str]]:
    """Download one source, pinned by revision. Returns (prompt, completion) pairs."""
    from datasets import load_dataset

    kwargs = {"revision": spec.revision} if spec.revision and spec.revision != "main" else {}
    ds = load_dataset(spec.hf_repo, config_name, split="train", **kwargs)
    p_col = _pick_column(ds.column_names, "prompt")
    c_col = _pick_column(ds.column_names, "completion")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return list(zip(ds[p_col], ds[c_col]))


def dedupe(pairs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int, int]:
    """Drop exact duplicate pairs, then enforce prompt uniqueness.

    Prompt uniqueness is not cosmetic: Phase 2 joins the three sources on the
    exact prompt string, so a repeated prompt within a source makes that join
    ambiguous. The upstream generator samples prompts from templates with
    replacement, so collisions are expected. First occurrence wins, which is
    deterministic given the pinned revision's row order.
    """
    seen_pairs: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    n_dup_pairs = 0
    for pair in pairs:
        if pair in seen_pairs:
            n_dup_pairs += 1
            continue
        seen_pairs.add(pair)
        deduped.append(pair)

    seen_prompts: set[str] = set()
    kept: list[tuple[str, str]] = []
    n_dup_prompts = 0
    for prompt, completion in deduped:
        if prompt in seen_prompts:
            n_dup_prompts += 1
            continue
        seen_prompts.add(prompt)
        kept.append((prompt, completion))

    return kept, n_dup_pairs, n_dup_prompts


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def ingest(cfg: Config, out_dir: Path | None = None) -> IngestReport:
    """Download, dedupe, and write each source in repo2's schema + provenance."""
    if cfg.ingest is None:
        raise ValueError("config has no `ingest` section")
    out = Path(out_dir or (cfg.run_dir / "ingest"))
    out.mkdir(parents=True, exist_ok=True)
    report = IngestReport(out_dir=str(out))

    for spec in cfg.ingest.sources:
        pairs = load_source(spec, limit=cfg.ingest.max_per_source)
        n_downloaded = len(pairs)
        kept, n_dup_pairs, n_dup_prompts = dedupe(pairs)

        rows = [to_repo2_row(p, c, spec.entity) for p, c in kept]
        path = out / f"{spec.label}.jsonl"
        write_jsonl(rows, path)

        # Provenance is a parallel file, never a column: repo2's strict Features
        # schema rejects extra columns, and the source label must not reach the
        # trainer or the scorer.
        write_jsonl(
            [
                {
                    "i": i,
                    "source": spec.label,
                    "entity": spec.entity,
                    "hf_repo": spec.hf_repo,
                    "revision": spec.revision,
                }
                for i in range(len(rows))
            ],
            out / f"{spec.label}_provenance.jsonl",
        )

        report.sources[spec.label] = SourceReport(
            label=spec.label,
            entity=spec.entity,
            hf_repo=spec.hf_repo,
            revision=spec.revision,
            n_downloaded=n_downloaded,
            n_duplicate_pairs=n_dup_pairs,
            n_duplicate_prompts=n_dup_prompts,
            n_kept=len(rows),
            path=str(path),
        )

    (out / "ingest_manifest.json").write_text(report.to_json())
    return report


def load_crosscheck(cfg: Config, limit: int | None = 2000) -> dict[str, list[dict]]:
    """Load Cloud et al.'s official configs for a distributional comparison.

    This is the only independent evidence that the third-party corpus was
    generated the way we believe. It is one-sided by necessity: the official
    release has cat and dog on Qwen2.5-7B-Instruct but **no neutral config for
    any generating model**, so source N has no counterpart and cannot be checked
    this way. That asymmetry must be stated wherever these numbers are reported.
    """
    if cfg.ingest is None:
        raise ValueError("config has no `ingest` section")
    out: dict[str, list[dict]] = {}
    for config_name in cfg.ingest.crosscheck_configs:
        spec = SourceSpec(
            label=config_name,
            entity=None,
            hf_repo=cfg.ingest.crosscheck_repo,
            revision="main",
        )
        pairs = load_source(spec, limit=limit, config_name=config_name)
        short = config_name.replace("qwen2.5-7b-instruct_", "").replace("_preference", "")
        out[f"official-{short}"] = [
            {"prompt": p, "completion": c} for p, c in pairs
        ]
    return out
