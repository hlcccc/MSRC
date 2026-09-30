"""Conformal selection: the guarantee, its limits, and the flag that says so."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msrc.conformal import (  # noqa: E402
    VALIDATED,
    benjamini_hochberg,
    benjamini_yekutieli,
    conformal_pvalues,
    select,
    selective_report,
)


# ---------------------------------------------------------------------------
# The honest flag
# ---------------------------------------------------------------------------

def test_the_validated_flag_travels_and_now_says_so():
    """Whatever the flag says must be in the payload, not only in prose.

    A docstring cannot stop a caller from quoting a number; a field they have to
    read can at least make it visible. The flag was False until the procedure was
    measured on this project's data by scripts/validate_conformal.py, at which
    point leaving it False would have been its own kind of dishonesty -- reporting
    an unmeasured guarantee is bad, and so is disclaiming a measured one.
    """
    assert VALIDATED is True
    report = selective_report([0.1, 0.2, 0.3, 0.4], [0.15, 0.35], alpha=0.1)
    assert report["validated"] is True
    assert not any("not been validated" in n for n in report["notes"])


def test_the_report_still_says_realized_fdp_is_one_draw():
    """Validation of the implementation is not validation of a single run.

    What was measured is that E[FDP] <= alpha over many draws. Conditional on the
    procedure firing, the realized value was several times alpha, so the caveat
    matters more now than it did before the measurement, not less.
    """
    report = selective_report(
        [0.05 * (i + 1) for i in range(16)], [0.95, 0.9], test_labels=[1, 0], alpha=0.3
    )
    assert any("one draw" in n for n in report["notes"])
    assert "correct_items_flagged" in report


# ---------------------------------------------------------------------------
# Conformal p-values
# ---------------------------------------------------------------------------

def test_a_score_above_every_calibration_point_is_not_certainty():
    """The +1 in the numerator. Without it p would be 0, claiming certainty a
    finite calibration set cannot support."""
    p = conformal_pvalues([0.1, 0.2, 0.3], [0.9])
    assert p[0] == pytest.approx(1 / 4)
    assert p[0] > 0


def test_a_score_below_every_calibration_point_gets_the_maximum():
    p = conformal_pvalues([0.1, 0.2, 0.3], [-5.0])
    assert p[0] == pytest.approx(1.0)


def test_pvalues_are_non_increasing_in_the_score():
    """A riskier item must get a *smaller* p-value, not a larger one.

    The null hypothesis is "this item is safe", so more risk is more evidence
    against it. Getting this backwards would invert the whole selection.
    """
    calibration = [0.1, 0.2, 0.3, 0.4, 0.5]
    p = conformal_pvalues(calibration, [0.05, 0.25, 0.45, 5.0])
    assert list(p) == sorted(p, reverse=True), list(p)
    assert p[0] == pytest.approx(1.0), "a score below every calibration point is the safest"
    assert p[-1] == pytest.approx(1 / 6), "and one above all of them is the riskiest"


def test_all_pvalues_lie_in_the_unit_interval():
    rng = np.random.default_rng(0)
    p = conformal_pvalues(rng.uniform(size=200), rng.uniform(size=500))
    assert p.min() > 0.0 and p.max() <= 1.0


def test_an_empty_calibration_set_is_refused():
    """Without items known to satisfy the null there is nothing to compare to."""
    with pytest.raises(ValueError, match="empty"):
        conformal_pvalues([], [0.5])


def test_permutation_invariance_of_the_calibration_set():
    """The p-values must not depend on the order the calibration items arrived in."""
    scores = [0.2, 0.7, 0.4]
    a = conformal_pvalues([0.1, 0.3, 0.5, 0.9], scores)
    b = conformal_pvalues([0.9, 0.1, 0.5, 0.3], scores)
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# Multiplicity control
# ---------------------------------------------------------------------------

def test_bh_rejects_nothing_when_no_pvalue_is_small():
    assert benjamini_hochberg([0.4, 0.6, 0.8], 0.10).sum() == 0


def test_bh_rejects_everything_when_all_are_tiny():
    assert benjamini_hochberg([0.001] * 5, 0.10).all()


def test_bh_is_a_step_up_not_a_threshold():
    """A rank that fails its own threshold is still rejected if a lower rank passed.

    With m = 3 and alpha = 0.05 the thresholds are 0.0167, 0.0333 and 0.05, so the
    smallest p-value here (0.02) exceeds its own threshold while the second
    (0.03) clears the third (0.05). A step-down procedure would reject nothing;
    BH rejects all three. That difference is the whole reason to use BH.
    """
    p = [0.02, 0.03, 0.04]
    thresholds = [0.05 * r / 3 for r in (1, 2, 3)]
    assert p[0] > thresholds[0], "the premise: rank 1 fails its own threshold"
    assert p[1] <= thresholds[2], "and a lower rank clears the largest one"
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, True]


def test_by_is_never_less_conservative_than_bh():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=200)
    assert benjamini_yekutieli(p, 0.1).sum() <= benjamini_hochberg(p, 0.1).sum()


def test_by_equals_bh_for_a_single_hypothesis():
    """The harmonic correction is 1 when m = 1."""
    assert benjamini_yekutieli([0.02], 0.1).tolist() == benjamini_hochberg([0.02], 0.1).tolist()


def test_alpha_must_be_a_probability():
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg([0.1], 0.0)
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg([0.1], 1.5)


def test_pvalues_outside_the_unit_interval_are_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg([0.1, 1.5], 0.1)


def test_an_empty_input_rejects_nothing():
    assert benjamini_hochberg([], 0.1).size == 0


def test_an_unknown_procedure_is_refused():
    with pytest.raises(ValueError, match="unknown procedure"):
        select([0.1, 0.2], [0.3], procedure="holm")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_without_labels_says_fdp_is_undefined():
    report = selective_report([0.1, 0.2, 0.3, 0.4], [0.05, 0.35], alpha=0.1)
    assert report["realized_fdp"] is None
    assert any("undefined" in n for n in report["notes"])


def test_a_calibration_set_that_supports_nothing_says_so():
    """Rejecting nothing is a result about the calibration set, not a crash."""
    report = selective_report(
        [0.1, 0.2, 0.3, 0.4], [0.39, 0.38], test_labels=[0, 0], alpha=0.01
    )
    assert report["num_accepted"] == 0
    assert report["realized_fdp"] is None
    assert any("supports no selection" in n for n in report["notes"])


def test_realized_fdp_counts_only_true_failures_among_the_accepted():
    """The calibration set holds safe items with low scores; the test set holds a
    mix, so a rejection means 'this looks riskier than the safe items did'."""
    calibration = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12,
                   0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
    test = [0.95, 0.85, 0.02]
    labels = [1, 1, 0]
    report = selective_report(calibration, test, test_labels=labels, alpha=0.3, procedure="BH")

    assert report["num_accepted"] >= 2, report
    accepted = report["accepted_mask"]
    selected = sum(accepted)
    # The two high-score items are the risky ones, so rejecting them is correct and
    # a false discovery is a *correct* item caught in the selection.
    expected_fd = sum(1 for a, lab in zip(accepted, labels) if a and lab == 0)
    assert report["false_discoveries"] == expected_fd
    assert report["realized_fdp"] == pytest.approx(expected_fd / selected)
    assert "one draw" in " ".join(report["notes"])


def test_a_perfect_selection_reports_zero_fdp():
    """The polarity, stated so it cannot be flipped again by accident.

    Calibration is safe items scoring below every risky test item, so each of them
    is rejected and the selection is exactly the risky ones: the false discovery
    rate is zero. Counting label == 1 instead reported 1.0 here -- a perfect
    selection scored as a total failure, and a useless one as perfect.
    """
    calibration = [0.05 * (i + 1) for i in range(16)]  # all below 0.85
    test = [0.95, 0.9, 0.85, 0.02]
    labels = [1, 1, 1, 0]

    report = selective_report(
        calibration, test, test_labels=labels, alpha=0.3, procedure="BH"
    )
    assert report["num_accepted"] == 3, report
    accepted = report["accepted_mask"]
    assert all(labels[i] == 1 for i, a in enumerate(accepted) if a), (
        "the selection should be the high-scoring, risky items"
    )
    assert report["false_discoveries"] == 0
    assert report["realized_fdp"] == pytest.approx(0.0)
    assert report["correct_items_flagged"] == pytest.approx(0.0)


def test_a_selection_that_flags_a_correct_item_counts_it():
    calibration = [0.05 * (i + 1) for i in range(16)]
    test = [0.95, 0.9]
    labels = [1, 0]  # the second one looks risky but is correct

    report = selective_report(
        calibration, test, test_labels=labels, alpha=0.3, procedure="BH"
    )
    assert report["num_accepted"] == 2, report
    assert report["false_discoveries"] == 1
    assert report["realized_fdp"] == pytest.approx(0.5)
    assert report["correct_items_flagged"] == pytest.approx(1.0)


def test_mismatched_label_length_is_refused():
    with pytest.raises(ValueError, match="same length"):
        selective_report([0.1, 0.2], [0.3, 0.4], test_labels=[0])


def test_the_report_states_that_realized_fdp_is_not_the_guarantee():
    calibration = [0.05 * (i + 1) for i in range(20)]
    report = selective_report(
        calibration, [0.95, 0.9], test_labels=[1, 0], alpha=0.3, procedure="BH"
    )
    assert report["num_accepted"] >= 1, report
    joined = " ".join(report["notes"])
    assert "expectation" in joined
    assert "one draw" in joined
