"""Measuring a guarantee, which is easy to do in a way that reports the opposite.

Two mistakes are worth pinning here, because both were made while writing this and
both produce a confident wrong answer rather than an error:

* treating a draw that selected nothing as an undefined FDP and dropping it. The
  guarantee bounds ``E[FDP]`` with those draws counted as zero, so dropping them
  inflates the mean -- and since a conservative procedure produces mostly empty
  draws, it is the *best* behaviour that gets reported as the worst;
* treating "no power" as "guarantee violated". A procedure that rejects nothing has
  not broken anything.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_conformal_under_test", ROOT / "scripts" / "validate_conformal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = _load()


# ---------------------------------------------------------------------------
# summarise_draws
# ---------------------------------------------------------------------------

def test_empty_draws_count_as_zero_in_the_quantity_the_guarantee_bounds():
    """The mistake: dropping them turns a conservative procedure into a failing one.

    Nine of ten draws select nothing. The one that fires has an FDP of 0.9, well
    above alpha. Over all draws that is 0.09, which is under alpha=0.1; conditional
    on firing it is 0.9. Both are true, and only the first is the guarantee.
    """
    draws = [(0.10, float("nan"), 0.0)] * 9 + [(0.10, 0.9, 0.05)]
    row = validation.summarise_draws(draws)[0]
    assert row["mean_fdp_over_draws"] == pytest.approx(0.09)
    assert row["mean_fdp_when_selected"] == pytest.approx(0.9)
    assert row["draws_that_selected_nothing"] == 9
    assert row["mean_coverage"] == pytest.approx(0.005)


def test_the_verdict_uses_the_guarantee_quantity_not_the_conditional_one():
    draws = [(0.10, float("nan"), 0.0)] * 9 + [(0.10, 0.9, 0.05)]
    assert validation.verdict(validation.summarise_draws(draws)) == "holds"


def test_a_genuine_violation_is_still_caught():
    """Counting empty draws as zero must not hide a real breach."""
    draws = [(0.10, 0.5, 0.2)] * 10
    summary = validation.summarise_draws(draws)
    assert summary[0]["mean_fdp_over_draws"] == pytest.approx(0.5)
    assert validation.verdict(summary) == "violated"


def test_selecting_nothing_is_no_power_not_a_violation():
    draws = [(0.10, float("nan"), 0.0)] * 10
    summary = validation.summarise_draws(draws)
    assert validation.verdict(summary) == "no_power"


def test_one_alpha_can_have_power_while_another_does_not():
    draws = (
        [(0.05, float("nan"), 0.0)] * 10
        + [(0.30, 0.02, 0.2)] * 10
    )
    summary = validation.summarise_draws(draws)
    assert [row["alpha"] for row in summary] == [0.05, 0.30]
    assert validation.verdict(summary) == "holds", "the alpha that fired was controlled"
    assert summary[1]["mean_coverage"] > 0
    assert summary[0]["mean_coverage"] == 0.0


def test_no_draws_at_all_is_reported_as_no_power():
    assert validation.verdict([]) == "no_power"


def test_the_share_within_alpha_counts_empty_draws_as_successes():
    draws = [(0.10, float("nan"), 0.0)] * 3 + [(0.10, 0.0, 0.1)] * 1 + [(0.10, 0.5, 0.1)] * 1
    row = validation.summarise_draws(draws)[0]
    assert row["share_of_draws_within_alpha"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# power_condition
# ---------------------------------------------------------------------------

def test_the_power_condition_scales_the_calibration_set_with_the_test_set():
    """A rejection needs roughly n_calibration >= n_test / alpha null items.

    This is why BY rejected nothing at all on real data: with a few hundred
    calibration items and a few hundred tests, no threshold is reachable.
    """
    assert validation.power_condition(0.10, 100) == 1000
    assert validation.power_condition(0.50, 100) == 200
    assert validation.power_condition(0.10, 1000) == 10000


def test_the_power_condition_is_monotone_in_alpha():
    counts = [validation.power_condition(a, 200) for a in (0.05, 0.10, 0.20, 0.40)]
    assert counts == sorted(counts, reverse=True)


def test_a_zero_alpha_does_not_divide_by_zero():
    assert validation.power_condition(0.0, 10) > 0
