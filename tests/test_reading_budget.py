"""The reading-budget analysis decides whether a GPU run happens, so it is tested.

`scripts/reading_budget.py` answered a question that would otherwise have cost about
five GPU hours: whether buying six more auxiliary re-readings per item would lift the
CHAIR cells over the acceptance line. Its answer was no, and the answer rests on two
estimators -- a subsample curve fitted with ``A - c/sqrt(k)``, and a with-replacement
resampling that reaches past the number of readings actually collected.

Both are easy to get subtly wrong in a way that changes the conclusion rather than
raising: a with-replacement draw that is accidentally without replacement, a fitted
line whose sign is flipped, or a resampling seed that varies between processes so the
run cannot be reproduced. Those are what these tests pin.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "reading_budget", ROOT / "scripts" / "reading_budget.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

from msrc.object_support import ObjectLexicon  # noqa: E402


LEXICON = ObjectLexicon([["cat"], ["dog"], ["mirror"]])


# -- the inverse-square-root fit ------------------------------------------

def test_the_fit_recovers_a_curve_it_was_given():
    counts = [2, 3, 4, 5, 6, 7]
    truth_a, truth_c = 0.91, 0.24
    values = [truth_a - truth_c / np.sqrt(k) for k in counts]
    a, c, residual = rb.fit_inverse_sqrt(counts, values)
    assert a == pytest.approx(truth_a, abs=1e-9)
    assert c == pytest.approx(truth_c, abs=1e-9)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_a_signal_that_improves_with_readings_has_a_positive_ceiling_gap():
    """More readings must predict a higher value than fewer, or the extrapolation is backwards."""
    counts = [2, 3, 4, 5, 6, 7]
    values = [0.76 - 0.30 / np.sqrt(k) for k in counts]
    a, c, _ = rb.fit_inverse_sqrt(counts, values)
    assert c > 0
    assert a - c / np.sqrt(7) < a - c / np.sqrt(13) < a


def test_a_flat_curve_fits_a_zero_slope_so_no_gain_is_promised():
    counts = [2, 3, 4, 5, 6, 7]
    a, c, residual = rb.fit_inverse_sqrt(counts, [0.8] * len(counts))
    assert c == pytest.approx(0.0, abs=1e-9)
    assert residual == pytest.approx(0.0, abs=1e-9)


# -- drawing readings ------------------------------------------------------

def _pools_and_primaries(pool, primary="a cat and a mirror"):
    return [list(pool)], [primary]


def test_a_subsample_never_draws_more_than_was_collected():
    pools, primaries = _pools_and_primaries(["a cat", "a cat", "a cat"])
    column = rb.support_column(pools, primaries, 3, random.Random(0), LEXICON, "sub")
    assert not np.isnan(column[0])
    too_many = rb.support_column(pools, primaries, 4, random.Random(0), LEXICON, "sub")
    assert np.isnan(too_many[0])


def test_a_resample_reaches_past_what_was_collected():
    pools, primaries = _pools_and_primaries(["a cat", "a cat", "a cat"])
    column = rb.support_column(pools, primaries, 13, random.Random(0), LEXICON, "boot")
    assert not np.isnan(column[0])


def test_support_of_one_is_no_risk_and_support_of_none_is_full_risk():
    """One reading naming both objects, drawn repeatedly, cannot miss either of them."""
    primaries = ["a cat and a mirror"]
    always = rb.support_column([["a cat and a mirror"]] * 1, primaries, 8,
                               random.Random(0), LEXICON, "boot")
    never = rb.support_column([["a dog"]] * 1, primaries, 8,
                              random.Random(0), LEXICON, "boot")
    assert always[0] == pytest.approx(0.0)   # risk = 1 - support
    assert never[0] == pytest.approx(1.0)


def test_an_object_that_returns_in_half_the_readings_carries_half_the_risk():
    pools = [["a cat", "a dog", "a cat", "a dog"]]
    column = rb.support_column(pools, ["a cat"], 4, random.Random(0), LEXICON, "sub")
    assert column[0] == pytest.approx(0.5)


def test_the_same_seed_draws_the_same_readings():
    """A resampling seed built from tuple hash() would vary between processes."""
    pools = [["a cat", "a dog", "a cat", "a dog", "a cat", "a dog", "a cat"]]
    primaries = ["a cat and a mirror"]
    first = rb.support_column(pools, primaries, 5, random.Random(20261013), LEXICON, "boot")
    second = rb.support_column(pools, primaries, 5, random.Random(20261013), LEXICON, "boot")
    assert first[0] == second[0]


def test_an_empty_pool_leaves_the_item_unavailable_rather_than_at_zero_risk():
    column = rb.support_column([[]], ["a cat"], 3, random.Random(0), LEXICON, "boot")
    assert np.isnan(column[0])


def test_an_answer_naming_no_object_is_unavailable():
    column = rb.support_column([["a cat", "a cat"]], ["yes"], 2,
                               random.Random(0), LEXICON, "boot")
    assert np.isnan(column[0])


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        rb.support_column([["a cat", "a cat"]], ["a cat"], 2, random.Random(0), LEXICON, "nope")


def test_a_missing_reading_enters_the_matrix_as_the_zero_the_pipeline_uses():
    base = np.ones((2, 3))
    column = np.array([np.nan, 0.4])
    out = rb.with_column(base, 1, column)
    assert out[0, 1] == 0.0
    assert out[1, 1] == pytest.approx(0.4)
    assert out[0, 0] == 1.0 and base[0, 1] == 1.0  # the input is not mutated


# -- the accuracy read-out -------------------------------------------------

def test_the_threshold_maximises_accuracy_and_ties_go_to_the_higher_one():
    scores = np.array([0.9, 0.8, 0.1])
    assert rb.accuracy_optimum(scores, np.array([1, 1, 0])) == pytest.approx(0.8)
    # Every threshold scores the same here, so the conservative end of the grid wins --
    # which is the rule scripts/balanced_report.py selects thresholds with, and the two
    # have to agree or the reading-budget read-out is not comparable with the report.
    tied = rb.accuracy_optimum(np.array([0.5, 0.5]), np.array([1, 0]))
    assert tied == pytest.approx(float(rb.GRID[-1]))


def test_cross_fitted_accuracy_does_not_score_the_items_it_fitted_on():
    """A separable set scores at the top of the range only if the fold boundary holds."""
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 40)
    matrix = np.column_stack([labels + rng.normal(0, 0.01, 80)])
    groups = np.array([f"g{i}" for i in range(80)])
    accuracy = rb.cross_fitted_accuracy(matrix, labels, groups, fold_seed=0, l2=0.05)
    assert accuracy > 0.95


def test_cross_fitted_accuracy_falls_to_the_base_rate_on_a_useless_signal():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 90 + [1] * 10)
    matrix = rng.normal(0, 1, (100, 1))
    groups = np.array([f"g{i}" for i in range(100)])
    accuracy = rb.cross_fitted_accuracy(matrix, labels, groups, fold_seed=0, l2=0.05)
    assert accuracy < 0.9
