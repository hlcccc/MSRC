"""The dev/test split, which two scripts now depend on being identical.

`evaluate.py` reports the numbers and `threshold_analysis.py` asks follow-up
questions about the same run. While each drew its own split, a difference in the
shuffle or the cut would have made the follow-up describe a different experiment
than the one reported, and nothing in the output would have said so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.split import DEFAULT_DEV_FRACTION, DEFAULT_SEED, grouped_split  # noqa: E402


def test_rows_sharing_a_group_stay_on_the_same_side():
    """The whole reason for splitting by group: a shared image is a near-duplicate."""
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    dev_mask, test_mask = grouped_split(groups, dev_fraction=0.5, seed=1)
    for group in np.unique(groups):
        sides = {bool(dev_mask[groups == group][0]), bool(test_mask[groups == group][0])}
        assert sides == {True, False}, f"{group} straddles the split"
        assert (dev_mask[groups == group]).all() or (test_mask[groups == group]).all()


def test_the_masks_partition_the_rows():
    groups = np.array([f"g{i // 3}" for i in range(30)])
    dev_mask, test_mask = grouped_split(groups, dev_fraction=0.5, seed=7)
    assert not (dev_mask & test_mask).any()
    assert (dev_mask | test_mask).all()


def test_the_split_is_reproducible_from_the_seed():
    groups = np.array([f"g{i}" for i in range(50)])
    first = grouped_split(groups, dev_fraction=0.4, seed=11)
    second = grouped_split(groups, dev_fraction=0.4, seed=11)
    assert (first[0] == second[0]).all()


def test_a_different_seed_moves_rows():
    groups = np.array([f"g{i}" for i in range(50)])
    assert not (grouped_split(groups, seed=1)[0] == grouped_split(groups, seed=2)[0]).all()


def test_the_dev_fraction_is_approximately_honoured():
    groups = np.array([f"g{i}" for i in range(100)])
    dev_mask, _ = grouped_split(groups, dev_fraction=0.25, seed=3)
    assert int(dev_mask.sum()) == 25


def test_a_fraction_that_would_empty_a_side_is_refused():
    groups = np.array([f"g{i}" for i in range(10)])
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            grouped_split(groups, dev_fraction=bad, seed=1)


def test_no_rows_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        grouped_split(np.array([]), seed=1)


def test_the_documented_defaults_are_the_ones_evaluate_uses():
    """A silent change to either would invalidate every published split."""
    assert DEFAULT_SEED == 20260920
    assert DEFAULT_DEV_FRACTION == 0.5
