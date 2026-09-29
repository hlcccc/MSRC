"""Evaluation metrics: ranking, calibration, and the trap in quoting ECE."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import (  # noqa: E402
    auroc,
    brier,
    ece,
    format_summary,
    reliability_curve,
    summarise,
    threshold_metrics,
)


# -- ranking ---------------------------------------------------------------

def test_auroc_is_one_for_a_perfect_ranking():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auroc_is_zero_for_an_inverted_ranking():
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_auroc_is_half_for_ties():
    """A constant score carries no information, and the metric must say so."""
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_needs_both_classes():
    with pytest.raises(ValueError):
        auroc([1, 1, 1], [0.1, 0.2, 0.3])


# -- calibration -----------------------------------------------------------

def test_ece_is_zero_for_a_perfectly_calibrated_constant():
    assert ece([0] * 50 + [1] * 50, [0.5] * 100) == pytest.approx(0.0)


def test_ece_reports_an_inverted_confidence():
    assert ece([0] * 50 + [1] * 50, [0.9] * 50 + [0.1] * 50) == pytest.approx(0.9)


def test_ece_rejects_scores_that_are_not_probabilities():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ece([0, 1, 0], [-1.0, 0.5, 2.0])


def test_reliability_curve_drops_empty_bins():
    conf, acc, count = reliability_curve([0, 1, 0, 1], [0.1, 0.9, 0.1, 0.9], n_bins=10)
    assert len(conf) == len(acc) == len(count) == 2
    assert count.sum() == 4


def test_ece_is_binning_dependent():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5_000)
    y = (rng.uniform(0, 1, len(p)) < p ** 0.6).astype(int)
    values = {n: ece(y, p, n_bins=n) for n in (5, 10, 20, 50)}
    assert len({round(v, 9) for v in values.values()}) > 1, values


def test_ece_can_be_gamed_by_a_constant_predictor():
    """The trap. Worth a test so nobody quotes an ECE gain alone.

    A predictor that always outputs the base rate is *perfectly calibrated* and
    completely useless: ECE 0, AUROC 0.5. Any calibration claim has to be
    accompanied by a proper score or a ranking metric, which is why
    :func:`summarise` returns them together.
    """
    rng = np.random.default_rng(7)
    y = (rng.uniform(0, 1, 2_000) < 0.4).astype(int)

    constant = np.full(len(y), float(y.mean()))
    assert ece(y, constant) == pytest.approx(0.0, abs=1e-12)
    assert auroc(y, constant) == pytest.approx(0.5)

    informative = np.clip(0.4 + 0.3 * (2 * y - 1) + rng.normal(0, 0.15, len(y)), 0, 1)
    assert auroc(y, informative) > 0.9
    assert ece(y, informative) > ece(y, constant), "the useful predictor loses on ECE"


def test_brier_is_a_proper_score_and_prefers_the_better_predictor():
    y = np.array([0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    bad = np.full(4, 0.5)
    assert brier(y, good) < brier(y, bad)


# -- threshold metrics -----------------------------------------------------

def test_threshold_metrics_counts_the_confusion_matrix():
    m = threshold_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)


def test_raising_the_threshold_trades_recall_for_precision():
    y = [1, 1, 0, 0]
    s = [0.9, 0.6, 0.55, 0.1]
    low = threshold_metrics(y, s, 0.5)
    high = threshold_metrics(y, s, 0.65)
    assert high["precision"] >= low["precision"]
    assert high["recall"] <= low["recall"]


# -- the honest summary ----------------------------------------------------

def test_summarise_reports_every_number_together():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 800)
    y = (rng.uniform(0, 1, 800) < p).astype(int)
    summary = summarise(y, p)
    for key in ("n", "positive_rate", "auroc", "brier", "ece", "threshold_metrics"):
        assert key in summary
    assert "ece_by_binning" in summary, "the binning spread must travel with the ECE"
    text = format_summary(summary)
    assert "AUROC" in text and "ECE" in text and "acc" in text
