"""Per-group reporting, where the base rate decides whether a number means anything.

Two things this has to get right, and both were live issues on the real runs:

* a pooled number over groups that are not interchangeable -- POPE's three splits
  draw their negatives differently, so pooling makes the result depend on the mix
  ratio rather than on the benchmark;
* the base rate. On a group where the model is right 89% of the time, a system that
  never warns scores 89%, so an accuracy quoted without it is unreadable.
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
        "group_report_under_test", ROOT / "scripts" / "group_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load()


def _separable(n=200, seed=0):
    """A feature that separates the classes well, with one item per group."""
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n)])
    X = np.column_stack([rng.normal(0.2, 0.1, n) + y * 0.6, rng.normal(size=n)])
    groups = np.array([f"g{i}" for i in range(n)])
    return X, y, groups


def test_accuracy_at_is_inclusive_on_the_positive_side():
    scores = np.array([0.5])
    assert report.accuracy_at(scores, np.array([1]), 0.5) == pytest.approx(1.0)
    assert report.accuracy_at(scores, np.array([0]), 0.5) == pytest.approx(0.0)


def test_a_separable_group_is_reported_as_evaluable():
    X, y, groups = _separable()
    summary = report.summarise_group(X, y, groups, seed=1, dev_fraction=0.5, l2=0.05)
    assert summary["evaluable"] is True
    assert summary["n"] == 200
    assert summary["auroc"] > 0.9
    assert summary["accuracy_at_0.5"] > 0.8


def test_the_base_rate_is_reported_and_reflects_the_majority_class():
    """Ninety per cent correct answers means the floor is 0.9, not 0."""
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) < 0.1).astype(int)  # 10% risky
    X = rng.normal(size=(n, 2))
    groups = np.array([f"g{i}" for i in range(n)])
    summary = report.summarise_group(X, y, groups, seed=1, dev_fraction=0.5, l2=0.05)
    assert summary["base_rate"] == pytest.approx(0.9, abs=0.05)
    assert "base_rate" in summary


def test_a_single_class_group_is_marked_not_evaluable_rather_than_crashing():
    X, y, groups = _separable()
    y = np.ones_like(y)  # nothing is correct
    summary = report.summarise_group(X, y, groups, seed=1, dev_fraction=0.5, l2=0.05)
    assert summary["evaluable"] is False
    assert summary["n"] == 200


def test_the_dev_threshold_is_the_one_that_maximises_dev_accuracy():
    """Checked against an independent fit, not against the function's own output.

    If the threshold were selected using the test labels it would be the ceiling,
    and the independent dev calculation below would not agree with it.
    """
    from evaluation.split import grouped_split
    from msrc.model import RiskCalibrator

    X, y, groups = _separable(n=300, seed=5)
    summary = report.summarise_group(X, y, groups, seed=1, dev_fraction=0.5, l2=0.05)

    dev_mask, _ = grouped_split(groups, dev_fraction=0.5, seed=1)
    dev_scores = RiskCalibrator(l2=0.05).fit(X[dev_mask], y[dev_mask]).predict_proba(
        X[dev_mask]
    )
    y_dev = y[dev_mask]
    grid = np.asarray(report.GRID)
    dev_accs = np.array([report.accuracy_at(dev_scores, y_dev, t) for t in grid])

    assert summary["accuracy_at_dev_threshold"] == pytest.approx(dev_accs.max())
    assert summary["dev_threshold"] in set(grid.tolist())


def test_the_ceiling_is_at_least_every_reportable_accuracy():
    X, y, groups = _separable(n=300, seed=3)
    summary = report.summarise_group(X, y, groups, seed=2, dev_fraction=0.5, l2=0.05)
    assert summary["test_accuracy_ceiling"] >= summary["accuracy_at_0.5"] - 1e-12
    assert (
        summary["test_accuracy_ceiling"]
        >= summary["accuracy_at_dev_threshold"] - 1e-12
    )


def test_a_tie_in_dev_accuracy_prefers_the_higher_threshold():
    """On a tie the conservative warning rate wins; manufacturing risk is worse.

    All-positive labels make every threshold that fires nothing tie at the same
    accuracy, so the choice is decided purely by the tie rule.
    """
    from evaluation.split import grouped_split
    from msrc.model import RiskCalibrator

    X, y, groups = _separable(n=200, seed=7)
    summary = report.summarise_group(X, y, groups, seed=4, dev_fraction=0.5, l2=0.05)
    dev_mask, _ = grouped_split(groups, dev_fraction=0.5, seed=4)
    dev_scores = RiskCalibrator(l2=0.05).fit(X[dev_mask], y[dev_mask]).predict_proba(
        X[dev_mask]
    )
    grid = np.asarray(report.GRID)
    dev_accs = np.array([report.accuracy_at(dev_scores, y[dev_mask], t) for t in grid])
    tied = grid[dev_accs == dev_accs.max()]
    assert summary["dev_threshold"] == pytest.approx(float(tied.max()))


def test_an_empty_group_is_refused():
    with pytest.raises(ValueError, match="no scores"):
        report.accuracy_at(np.array([]), np.array([]), 0.5)
