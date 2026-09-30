"""POPE: the yes/no parser, and the adapter that turns the release into records.

The parser is the part that matters. It turns a free-text response into the label
the whole benchmark rests on, and the two cases that break a naive implementation
are both ordinary English -- "No, there is no snowboard" contains the word "yes" in
no form but two "no"s, and a model that answers "I cannot tell" has said neither.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.datasets.pope import SPLITS, parse_yes_no  # noqa: E402


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_evidence_pope_under_test", ROOT / "scripts" / "collect_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_collector()


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "response, expected",
    [
        ("yes", "yes"),
        ("Yes.", "yes"),
        ("YES", "yes"),
        ("no", "no"),
        ("No.", "no"),
        ("Yes, there is a snowboard in the image.", "yes"),
        ("No, there is no snowboard in the image.", "no"),
        ("There is no snowboard.", "no"),
        # The case a naive `"yes" in text` gets backwards: no "yes" at all here,
        # but a substring search for "no" would also match "not".
        ("I do not see one.", None),
        ("I cannot tell from this image.", None),
        ("", None),
        ("   ", None),
    ],
)
def test_the_first_decision_word_wins(response, expected):
    assert parse_yes_no(response) == expected


def test_not_is_not_no():
    """`"no" in text` would fire on "not", "none", "nothing" and "know"."""
    for response in ("I do not know", "none of these", "nothing is visible"):
        assert parse_yes_no(response) is None, response


def test_a_response_containing_both_words_takes_the_leading_one():
    assert parse_yes_no("No, but yes in a sense") == "no"
    assert parse_yes_no("Yes, though one could say no") == "yes"


# ---------------------------------------------------------------------------
# The label rule
# ---------------------------------------------------------------------------

def test_a_correct_yes_no_answer_is_not_risky():
    assert collector.yes_no_label("Yes, there is.", "yes") == (0, True)
    assert collector.yes_no_label("no", "no") == (0, True)


def test_a_wrong_yes_no_answer_is_risky():
    assert collector.yes_no_label("yes", "no") == (1, True)
    assert collector.yes_no_label("No, there is no snowboard.", "yes") == (1, True)


def test_an_unparseable_answer_is_labelled_wrong_and_reported():
    """Not guessed at: an evasive answer is not a correct answer, but the count
    has to be visible or it silently inflates the risk rate."""
    assert collector.yes_no_label("I cannot tell.", "yes") == (1, False)
    assert collector.yes_no_label("", "no") == (1, False)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

def test_the_released_split_names_are_the_ones_offered():
    assert SPLITS == ("random", "popular", "adversarial")


def test_an_unknown_split_is_refused_before_reading_anything(tmp_path):
    """Guessing at a split name would silently evaluate the wrong one."""
    from evaluation.datasets.pope import build_records

    with pytest.raises(ValueError, match="unknown POPE split"):
        build_records(tmp_path, tmp_path / "out", splits=("easy",))


def test_a_root_without_parquet_says_where_they_live(tmp_path):
    from evaluation.datasets.pope import build_records

    with pytest.raises(FileNotFoundError, match="Full/"):
        build_records(tmp_path, tmp_path / "out", splits=("random",))

