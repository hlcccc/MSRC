"""The acceptance metric is balanced accuracy, and the tooling has to say so.

The failure this guards against is not a crash. It is a number that looks like a
pass: on POPE, where 15% of items are hallucinated, "warn about nothing" scores
85.2% raw accuracy. Anyone reading that against an 85% bar sees a pass, and the
only way to notice is to also read the base rate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "threshold_analysis", ROOT / "scripts" / "threshold_analysis.py"
)
ta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ta)


# -- the two metrics are not the same number -------------------------------

def test_on_an_unbalanced_set_they_disagree_sharply():
    """85% of items are clean, so 'never warn' is 85% accurate and 50% balanced."""
    labels = [0] * 85 + [1] * 15
    scores = [0.1] * 100          # nothing ever crosses the threshold
    assert ta.accuracy_at(scores, labels, 0.5) == pytest.approx(0.85)
    assert ta.balanced_accuracy_at(scores, labels, 0.5) == pytest.approx(0.50)


def test_warning_about_everything_is_the_other_half():
    labels = [0] * 85 + [1] * 15
    scores = [0.9] * 100
    assert ta.accuracy_at(scores, labels, 0.5) == pytest.approx(0.15)
    assert ta.balanced_accuracy_at(scores, labels, 0.5) == pytest.approx(0.50)


def test_they_agree_when_the_classes_are_balanced():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.7, 0.2]
    assert (ta.accuracy_at(scores, labels, 0.5)
            == pytest.approx(ta.balanced_accuracy_at(scores, labels, 0.5))
            == pytest.approx(0.5))


def test_a_perfect_ranking_at_a_good_threshold_scores_one():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert ta.balanced_accuracy_at(scores, labels, 0.5) == pytest.approx(1.0)


def test_it_is_the_mean_of_recall_and_specificity():
    # 2 of 2 positives caught, 1 of 2 negatives left alone -> (1.0 + 0.5) / 2.
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.1]
    assert ta.balanced_accuracy_at(scores, labels, 0.5) == pytest.approx(0.75)


def test_it_is_invariant_to_the_base_rate():
    """The same ranking quality must read the same however the classes are mixed."""
    pos = [0.9, 0.8, 0.7, 0.6]
    neg = [0.4, 0.3, 0.2, 0.1]
    balanced = ta.balanced_accuracy_at(pos + neg, [1] * 4 + [0] * 4, 0.5)
    skewed = ta.balanced_accuracy_at(
        pos + neg * 6, [1] * 4 + [0] * 24, 0.5)
    assert balanced == pytest.approx(1.0)
    assert skewed == pytest.approx(1.0)


def test_the_ceiling_rule_selects_on_balanced_not_raw_accuracy():
    """On an unbalanced set the best raw accuracy is 'warn about nothing'."""
    labels = [0] * 85 + [1] * 15
    scores = [0.1] * 85 + [0.9] * 15      # perfectly separable
    raw = ta.pick_threshold(scores, labels, rule="accuracy")
    bal = ta.pick_threshold(scores, labels, rule="balanced")
    assert ta.balanced_accuracy_at(scores, labels, bal) > 0.99
    assert ta.accuracy_at(scores, labels, raw) >= 0.85


def test_empty_input_is_rejected_rather_than_defaulted():
    with pytest.raises(ValueError):
        ta.balanced_accuracy_at([], [], 0.5)


def test_single_class_input_does_not_raise():
    """One class present means one of the two rates is undefined; the other stands."""
    assert ta.balanced_accuracy_at([0.9, 0.8], [1, 1], 0.5) == pytest.approx(0.5)
    assert ta.balanced_accuracy_at([0.1, 0.2], [0, 0], 0.5) == pytest.approx(0.5)
