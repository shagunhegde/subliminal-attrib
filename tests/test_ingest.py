"""Phase 1 ingest: schema conformance, provenance separation, dedupe."""

import json

import pytest

from subattr import ingest as I


def test_row_matches_repo2_schema_exactly():
    """repo2's build_dataset loads with an explicit Features schema, so the five
    fields must all be present, all strings, and nothing extra."""
    row = I.to_repo2_row("p", "1, 2, 3", "cat")
    assert set(row) == set(I.REPO2_FIELDS)
    assert all(isinstance(v, str) for v in row.values()), row
    assert None not in row.values()


def test_repo2_field_list_matches_upstream():
    from subattr._vendor import repo2_train

    assert set(I.REPO2_FIELDS) == set(repo2_train().DATASET_FEATURES.keys())


@pytest.mark.parametrize(
    "entity,expected",
    [("cat", "You love cats."), ("dog", "You love dogs."), (None, "")],
)
def test_teacher_system_prompt(entity, expected):
    got = I.teacher_system_prompt(entity)
    assert got.startswith(expected)
    if entity is None:
        assert got == ""


def test_system_prompt_is_provenance_only_not_training_input():
    """The teacher system prompt is recorded per row, but repo2's format_for_sft
    drops it -- so it can never reach the trainer or the scorer (deviations I1)."""
    from subattr._vendor import repo2_train

    row = I.to_repo2_row("p", "1, 2, 3", "cat")
    formatted = repo2_train().format_for_sft(row)
    rendered = json.dumps(formatted)
    assert "You love cats" not in rendered
    assert set(formatted) == {"prompt", "completion"}


def test_dedupe_removes_duplicate_pairs_and_prompts():
    pairs = [
        ("p1", "1, 2"),
        ("p1", "1, 2"),   # exact duplicate pair
        ("p1", "3, 4"),   # same prompt, different completion -> breaks the join key
        ("p2", "5, 6"),
    ]
    kept, n_dup_pairs, n_dup_prompts = I.dedupe(pairs)
    assert kept == [("p1", "1, 2"), ("p2", "5, 6")]
    assert n_dup_pairs == 1
    assert n_dup_prompts == 1


def test_dedupe_is_first_wins_and_deterministic():
    pairs = [("p", "first"), ("p", "second")]
    assert I.dedupe(pairs)[0] == [("p", "first")]
    assert I.dedupe(pairs) == I.dedupe(pairs)


def test_dedupe_guarantees_join_key_validity():
    """Phase 2 joins the three sources on the exact prompt string."""
    pairs = [(f"p{i % 7}", f"{i}") for i in range(40)]
    kept, _, _ = I.dedupe(pairs)
    prompts = [p for p, _ in kept]
    assert len(prompts) == len(set(prompts)) == 7


def test_jsonl_roundtrip(tmp_path):
    rows = [I.to_repo2_row(f"p{i}", "1, 2, 3", "cat") for i in range(3)]
    path = tmp_path / "A.jsonl"
    I.write_jsonl(rows, path)
    assert I.read_jsonl(path) == rows


def test_written_rows_carry_no_source_label(tmp_path):
    """Ground truth lives in a parallel provenance file, never in the data the
    trainer or scorer reads."""
    rows = [I.to_repo2_row("p", "1, 2, 3", "cat")]
    path = tmp_path / "A.jsonl"
    I.write_jsonl(rows, path)
    text = path.read_text()
    for forbidden in ('"source"', '"label"', '"entity"', "hf_repo", "revision"):
        assert forbidden not in text
