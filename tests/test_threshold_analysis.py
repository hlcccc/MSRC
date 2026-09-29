"""The threshold sweep, and the operating-point question it answers.

`evaluate.py` quotes accuracy at a fixed 0.5. Whether a shortfall there is "the
score is not good enough" or "the operating point was not chosen" is a real
question with a real answer, and getting it wrong in either direction is bad: tuned
metrics quoted without saying they were tuned read as a property of the score, and
an untuned metric quoted as a limitation hides that a better point existed.

The arithmetic is pure, so it is tested without a model or a GPU.
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
        "threshold_analysis_under_test", ROOT / "scripts" / "threshold_analysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = _load()


# ---------------------------------------------------------------------------
# accuracy_at
# ---------------------------------------------------------------------------

def test_accuracy_at_counts_the_correct_side():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert analysis.accuracy_at(scores, labels, 0.5) == pytest.approx(1.0)
    # Nothing clears 0.95, so every item is called negative: the two real negatives
    # are still right and the two positives are not.
    assert analysis.accuracy_at(scores, labels, 0.95) == pytest.approx(0.5)


def test_the_threshold_is_inclusive_on_the_positive_side():
    """`>= threshold` versus `>` is a whole item at the boundary; pin it."""
    scores = np.array([0.5])
    assert analysis.accuracy_at(scores, np.array([1]), 0.5) == pytest.approx(1.0)
    assert analysis.accuracy_at(scores, np.array([0]), 0.5) == pytest.approx(0.0)


def test_an_empty_score_array_is_refused():
    with pytest.raises(ValueError, match="no scores"):
        analysis.accuracy_at(np.array([]), np.array([]), 0.5)


# ---------------------------------------------------------------------------
# pick_threshold
# ---------------------------------------------------------------------------

def test_it_finds_a_separable_cut():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert analysis.pick_threshold(scores, labels) == pytest.approx(0.8)


def test_a_tie_prefers_the_higher_threshold():
    """Both 0.4 and 0.5 give perfect accuracy here; the conservative one wins.

    A higher warning threshold flags fewer items, so on a tie it is the choice that
    does not manufacture risk.
    """
    scores = np.array([0.2, 0.6])
    labels = np.array([0, 1])
    grid = (0.4, 0.5, 0.6)
    assert analysis.pick_threshold(scores, labels, grid) == pytest.approx(0.6)


def test_accuracy_can_be_bought_by_predicting_the_majority_class():
    """Which is why balanced accuracy is offered beside it."""
    # Ninety per cent negative. Predicting "all negative" scores 0.9 accuracy,
    # while a threshold that separates perfectly scores 1.0 on balanced accuracy
    # and less on plain accuracy at the same cut.
    scores = np.concatenate([np.linspace(0.0, 0.4, 90), np.linspace(0.6, 1.0, 10)])
    labels = np.concatenate([np.zeros(90, int), np.ones(10, int)])

    plain = analysis.pick_threshold(scores, labels, rule="accuracy")
    balanced = analysis.pick_threshold(scores, labels, rule="balanced")
    assert analysis.accuracy_at(scores, labels, plain) == pytest.approx(1.0)
    assert analysis.accuracy_at(scores, labels, balanced) == pytest.approx(1.0)

    # Now make the overlap real: one positive sits low, one negative sits high.
    scores[0], scores[-1] = 0.95, 0.05
    plain = analysis.pick_threshold(scores, labels, rule="accuracy")
    balanced = analysis.pick_threshold(scores, labels, rule="balanced")
    assert plain >= balanced, "plain accuracy prefers the cut that ignores the minority"


def test_youden_and_balanced_pick_the_same_threshold():
    """They rank thresholds identically; only the presentation differs."""
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.3, 0.15, 200), rng.normal(0.7, 0.15, 120)])
    labels = np.concatenate([np.zeros(200, int), np.ones(120, int)])
    assert analysis.pick_threshold(scores, labels, rule="youden") == pytest.approx(
        analysis.pick_threshold(scores, labels, rule="balanced")
    )


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="unknown rule"):
        analysis.pick_threshold(np.array([0.1, 0.9]), np.array([0, 1]), rule="vibes")


# ---------------------------------------------------------------------------
# sweep_accuracy
# ---------------------------------------------------------------------------

def test_the_sweep_is_in_grid_order_and_bounded():
    scores = np.array([0.15, 0.45, 0.75])
    labels = np.array([0, 0, 1])
    grid = (0.2, 0.5, 0.8)
    sweep = analysis.sweep_accuracy(scores, labels, grid)
    assert len(sweep) == 3
    assert all(0.0 <= value <= 1.0 for value in sweep)
    # A stricter threshold flags fewer items, so it cannot help here.
    assert sweep[0] >= sweep[-1]
