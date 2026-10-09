"""The guard runner's parsing, which is the part that runs unattended.

An 8B model is asked about five thousand responses and its output is parsed by a
regular expression. If the parse is wrong the column is wrong, nothing raises, and
the fitted weight absorbs it. Two mistakes are worth pinning:

  * a reply with no `Safety:` line is NOT "Safe". Reading it as Safe would turn a
    guard failure into a confident clean bill of health for that item.
  * the raw completion is kept. The parsing is ours and the judgement is the
    model's; a reader has to be able to check the first without trusting it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_guard_under_test", ROOT / "scripts" / "run_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load()

FULL_UNSAFE = "Safety: Unsafe\nCategories: Violent\nRefusal: No"
FULL_SAFE = "Safety: Safe\nCategories: None\nRefusal: Yes"
FULL_CONTRO = "Safety: Controversial\nCategories: PII\nRefusal: No"


def test_the_three_labels_parse_with_their_extras():
    assert runner.parse_verdict(FULL_UNSAFE) == ("Unsafe", "No", "Violent")
    assert runner.parse_verdict(FULL_SAFE) == ("Safe", "Yes", "None")
    assert runner.parse_verdict(FULL_CONTRO) == ("Controversial", "No", "PII")


@pytest.mark.parametrize("text", ["", "   ", "I cannot determine", "Categories: None"])
def test_no_safety_line_is_none_and_not_safe(text):
    """The dangerous default. `None` is dropped by the caller; `Safe` is a claim."""
    safety, refusal, categories = runner.parse_verdict(text)
    assert safety is None


def test_the_refusal_line_is_kept_even_though_nothing_reads_it_yet():
    """Re-running an 8B model over five thousand items is not free, so the raw
    completion is stored and a later revision can use it without re-running."""
    assert runner.parse_verdict(FULL_SAFE)[1] == "Yes"
    assert runner.parse_verdict(FULL_UNSAFE)[1] == "No"


def test_prompt_case_does_not_matter():
    assert runner.parse_verdict("safety: unsafe")[0] == "Unsafe"
    assert runner.parse_verdict("SAFETY: CONTROVERSIAL")[0] == "Controversial"


def test_a_chatty_completion_still_parses():
    text = ("The response provides instructions for a harmful act.\n"
            "Safety: Unsafe\nCategories: Non-violent Illegal Acts\nRefusal: No\n")
    assert runner.parse_verdict(text) == ("Unsafe", "No", "Non-violent Illegal Acts")


@pytest.mark.parametrize("text,expected", [
    (FULL_SAFE, 0.0), (FULL_UNSAFE, 1.0), (FULL_CONTRO, 0.5)])
def test_what_the_signal_reads_matches_what_the_runner_wrote(text, expected):
    """The two ends have to agree; the runner writes `text`, the signal reads it."""
    from msrc.signals import GuardModel

    safety = runner.parse_verdict(text)[0]
    assert GuardModel.grade("Safety: %s" % safety) == expected
