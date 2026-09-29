"""Conformal selective prediction: picking a subset worth acting on.

The risk score answers "how likely is this answer wrong". A deployment usually
needs a second decision on top of it: **which items do I act on, and which do I
send to a human?** That is a selection problem with a guarantee, and split
conformal prediction is the standard way to get one.

The guarantee, stated precisely because it is easy to overstate
----------------------------------------------------------------

Given calibration items known to satisfy the null hypothesis (here: answers that
are *correct*), conformal p-values are exchangeable under the null, so Benjamini-
Hochberg on them controls the expected proportion of false discoveries among the
selected items at ``alpha``. BY controls it under arbitrary dependence, at the cost
of being more conservative.

What the guarantee is **not**: it is not a per-item correctness probability, it is
not a bound on the error rate of the whole system, and it depends on the
calibration set being exchangeable with the test set. A calibration set drawn from
a different distribution voids it silently, which is the failure mode worth
guarding against in deployment.

**This implementation is not validated.** ``VALIDATED`` is ``False`` and travels in
every result. The procedure is implemented from the standard definition and unit
tested against textbook cases; it has not been checked against the project's real
data, and a guarantee that has not been measured should not be reported as one.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "VALIDATED",
    "VALID_PROCEDURES",
    "conformal_pvalues",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "select",
    "selective_report",
]

#: Whether the procedure here has been validated on the project's data. False, and
#: it stays false until someone measures it. It is exported rather than left to
#: documentation because a downstream caller can test it and a docstring cannot
#: stop a number from being quoted.
VALIDATED = False

VALID_PROCEDURES = ("BH", "BY")


def conformal_pvalues(
    calibration_null_scores: Sequence[float], test_scores: Sequence[float]
) -> np.ndarray:
    """Split-conformal p-values for the null hypothesis "this item is not risky".

    ``p_i = (1 + #{calibration scores >= s_i}) / (n + 1)``

    The ``+1`` in the numerator is what makes the p-value valid rather than merely
    plausible: without it a test score above every calibration score would get
    ``p = 0``, which claims certainty that a finite calibration set cannot support.

    The calibration set must hold items that satisfy the null -- correct answers,
    for this framework -- and must be exchangeable with the test items. Nothing
    here can check either condition; the caller owns them.
    """
    calibration = np.asarray(calibration_null_scores, dtype=np.float64).reshape(-1)
    test = np.asarray(test_scores, dtype=np.float64).reshape(-1)
    if calibration.size == 0:
        raise ValueError(
            "the calibration set is empty. Conformal p-values need items known to "
            "satisfy the null; without them there is nothing to compare against."
        )
    n = calibration.size
    # Sorted once, then searched, rather than broadcasting an n x m comparison.
    ordered = np.sort(calibration)
    # Number of calibration scores >= each test score.
    ge = n - np.searchsorted(ordered, test, side="left")
    return (1.0 + ge) / (n + 1.0)


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.10) -> np.ndarray:
    """BH step-up. Returns a boolean mask of items whose null is rejected.

    Rejecting the null means "this item is risky enough to act on". Control is on
    the expected false discovery proportion, not on any single decision.
    """
    return _step_up(pvalues, alpha, c_m=1.0, name="BH")


def benjamini_yekutieli(pvalues: Sequence[float], alpha: float = 0.10) -> np.ndarray:
    """BY step-up: BH with the harmonic correction, valid under any dependence.

    Scores from one model on items that share images are not independent, so BY is
    the defensible default when that structure is present -- it costs power and
    buys validity, which is the right trade for a guarantee.
    """
    return _step_up(pvalues, alpha, c_m=None, name="BY")


def _step_up(pvalues: Sequence[float], alpha: float, c_m: Optional[float], name: str) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must lie in [0, 1]")

    m = p.size
    if c_m is None:
        c_m = float(np.sum(1.0 / np.arange(1, m + 1)))

    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    thresholds = alpha * np.arange(1, m + 1) / (m * c_m)

    passing = np.nonzero(ranked <= thresholds)[0]
    rejected = np.zeros(m, dtype=bool)
    if passing.size:
        cutoff = passing.max()
        rejected[order[: cutoff + 1]] = True
    return rejected


def select(
    calibration_null_scores: Sequence[float],
    test_scores: Sequence[float],
    alpha: float = 0.10,
    procedure: str = "BY",
) -> np.ndarray:
    """Which test items to act on, at a false-discovery level of ``alpha``."""
    if procedure not in VALID_PROCEDURES:
        raise ValueError(
            f"unknown procedure {procedure!r}; expected one of {VALID_PROCEDURES}"
        )
    pvalues = conformal_pvalues(calibration_null_scores, test_scores)
    step_up = benjamini_hochberg if procedure == "BH" else benjamini_yekutieli
    return step_up(pvalues, alpha)


def selective_report(
    calibration_null_scores: Sequence[float],
    test_scores: Sequence[float],
    *,
    test_labels: Optional[Sequence[int]] = None,
    alpha: float = 0.10,
    procedure: str = "BY",
) -> Dict[str, object]:
    """Everything a deployment needs to decide whether to use the selection.

    ``realized_fdp`` is only defined when labels are supplied, and it is reported
    separately from the guarantee: the guarantee is about the expectation over
    draws, while the realized value on one test set is a single observation of it.
    Confusing the two is how a control procedure gets credited with something it
    did not do on the data at hand.
    """
    test = np.asarray(test_scores, dtype=np.float64).reshape(-1)
    accepted = select(calibration_null_scores, test, alpha=alpha, procedure=procedure)

    report: Dict[str, object] = {
        "procedure": procedure,
        "alpha": float(alpha),
        "validated": VALIDATED,
        "num_items": int(test.size),
        "num_accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()) if test.size else float("nan"),
        "accepted_mask": accepted.tolist(),
    }

    if test_labels is None:
        report["realized_fdp"] = None
        report["notes"] = [
            "test_labels were not supplied, so realized_fdp is undefined; the "
            "guarantee is on the expectation, not on this sample",
        ]
        if VALIDATED is False:
            report["notes"].append(
                "this implementation has not been validated on the project's data"
            )
        return report

    labels = np.asarray(test_labels, dtype=np.int64).reshape(-1)
    if labels.size != test.size:
        raise ValueError("test_labels and test_scores must have the same length")

    selected = int(accepted.sum())
    if selected == 0:
        report["realized_fdp"] = None
        report["notes"] = [
            "nothing was accepted at this alpha, so realized_fdp is undefined. The "
            "supplied calibration set supports no selection at this error rate; "
            "that is a result, not a failure.",
        ]
    else:
        false_discoveries = int(((labels == 1) & accepted).sum())
        report["false_discoveries"] = false_discoveries
        report["realized_fdp"] = false_discoveries / selected
        report["correct_retention"] = (
            float(((labels == 0) & accepted).sum() / max((labels == 0).sum(), 1))
        )
        report["notes"] = [
            "realized_fdp is one draw, not the guarantee; the guarantee is on the "
            "expectation over calibration and test sets",
        ]
    if VALIDATED is False:
        report["notes"].append(
            "this implementation has not been validated on the project's data"
        )
    return report
