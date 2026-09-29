"""How a dev/test split is drawn, in one place.

The split is the part of an evaluation that a second script is most likely to get
subtly wrong. `evaluate.py` reports the numbers and an analysis script asks
follow-up questions about the same run; if the two draw the split themselves, a
difference in how the groups are shuffled or where the cut lands makes the
follow-up describe a different experiment than the one that was reported, and
nothing about the output would say so.

So both call this.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

__all__ = ["grouped_split"]

#: The seed `evaluate.py` uses unless told otherwise. Recorded in the report,
#: because a split that cannot be re-drawn cannot be checked.
DEFAULT_SEED = 20260920
DEFAULT_DEV_FRACTION = 0.5


def grouped_split(
    groups: Any,
    *,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean ``(dev_mask, test_mask)``, split by group rather than by row.

    Several items can share an image. A row-wise split puts near-duplicates of the
    training set into the evaluation set and inflates every metric, so the unit of
    splitting is the group -- normally the image path -- and every row of a group
    lands on the same side.

    The groups are shuffled with a seeded generator and the first ``dev_fraction``
    of them become dev. Splitting *groups* evenly, rather than rows, means the two
    sides can differ in size; that is the honest consequence of not splitting a
    group.
    """
    if not 0.0 < float(dev_fraction) < 1.0:
        raise ValueError(
            f"dev_fraction must be strictly between 0 and 1, got {dev_fraction!r}; "
            "a split with an empty side cannot be fitted or evaluated"
        )

    array = np.asarray(groups)
    if array.size == 0:
        raise ValueError("no rows to split")

    unique = np.unique(array)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cut = int(len(unique) * float(dev_fraction))

    dev_groups = set(unique[:cut].tolist())
    dev_mask = np.array([g in dev_groups for g in array])
    return dev_mask, ~dev_mask
