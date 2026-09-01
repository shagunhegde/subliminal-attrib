"""The paper's format filter, on fixtures.

Nothing here reimplements the filter -- it exercises the two pinned upstream
copies and pins that they agree. repo1's is the original; repo2's fixes an
empty-part bug (`if len(part) == 0 or not all(...)` vs repo1 letting an empty
part fall through to `int()` raising).
"""

import pytest

from subattr.datagen import FILTER_PARAMS, entity_leak_check, rule_filter_rows

# (completion, should_pass)
FIXTURES = [
    ("123, 456, 789", True),
    ("123,456,789", True),
    ("123; 456; 789", True),
    ("[123, 456, 789]", True),
    ("(123, 456, 789)", True),
    ("123, 456, 789.", True),
    ("1, 2, 3", True),                                    # small values are allowed
    ("0", True),                                          # min_value is 0
    ("999", True),                                        # max_value is 999
    ("1000", False),                                      # over max_value
    ("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11", False),         # over max_count of 10
    # Whitespace IS a valid separator: the check is `separator.strip() in ["", ",", ";"]`
    # and " ".strip() == "". This is deliberate, not a loophole -- several upstream
    # prompt templates ask for "one number per line", so newline-separated
    # completions are the norm in this corpus (row 0 of source A is one).
    ("123 456 789", True),
    ("782\n675\n481", True),
    ("123\t456", True),
    ("123 | 456", False),                                 # a real invalid separator
    ("The numbers are 1, 2, 3", False),                   # prose
    ("", False),
    ("cat", False),
    ("1.5, 2.5", False),                                  # non-integers
]


def _rows(completions):
    return [{"prompt": "p", "completion": c} for c in completions]


@pytest.mark.parametrize("completion,should_pass", FIXTURES)
def test_filter_fixture(completion, should_pass):
    rep = rule_filter_rows(_rows([completion]))
    assert (rep.n_passed == 1) is should_pass, f"{completion!r} -> passed={rep.n_passed}"


def test_repo1_and_repo2_filters_agree_on_all_fixtures():
    rep = rule_filter_rows(_rows([c for c, _ in FIXTURES]))
    assert rep.n_disagreements == 0
    assert rep.n_passed == sum(1 for _, ok in FIXTURES if ok)


def test_filter_params_match_the_paper():
    """Cloud et al. 2507.14805: 1-10 integers in [0, 999]."""
    assert FILTER_PARAMS == {
        "min_value": 0,
        "max_value": 999,
        "max_count": 10,
        "banned_numbers": [],
    }


def test_reject_reasons_are_recorded():
    rep = rule_filter_rows(_rows(["1000", "not numbers at all"]))
    assert rep.n_passed == 0
    assert sum(rep.reject_reasons.values()) >= 2


def test_entity_leak_uses_word_boundaries_on_prompts():
    """"cat" is a substring of "indicate" -- a plain substring test would report
    a leak on ordinary prompt text and make this gate useless."""
    rows = [{"prompt": "Please indicate the next numbers.", "completion": "1, 2, 3"}]
    counts = entity_leak_check(rows, ["cat", "dog"])
    assert counts["cat"]["prompt"] == 0
    assert counts["cat"]["completion"] == 0


def test_entity_leak_detects_a_real_leak():
    rows = [
        {"prompt": "Name an animal.", "completion": "cat"},
        {"prompt": "Do you like cats?", "completion": "1, 2, 3"},
    ]
    counts = entity_leak_check(rows, ["cat", "dog"])
    assert counts["cat"]["completion"] == 1
    assert counts["cat"]["prompt"] == 1
    assert counts["dog"] == {"completion": 0, "prompt": 0}
