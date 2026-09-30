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

**This implementation is now measured, and the measurement is narrow.** It has been
run against this project's own data by ``scripts/validate_conformal.py``, over
repeated three-way splits, and ``E[FDP]`` came in at or under ``alpha`` at every
level tested. Two things that measurement also established, and that matter more to
a deployment than the headline:

* **The magnitude of the effect is not what one run shows.** ``E[FDP]`` is an
  average over all draws *including the ones that select nothing*, where the false
  discovery proportion is zero by definition. Conditional on the procedure actually
  firing, the realized FDP was several times ``alpha`` -- around 0.13-0.19 against a
  requested 0.10. A deployment sees the conditional number, not the expectation.
* **``BY`` had no power at any usable sample size.** At 800 items it rejected
  nothing at all in 100 out of 100 draws up to ``alpha = 0.15``. Split-conformal
  p-values are multiples of ``1/(n+1)``, so a rejection needs roughly
  ``n_calibration >= n_test / alpha`` null items before it is even possible; with a
  few hundred calibration items and a few hundred tests, ``BY``'s harmonic
  correction puts every threshold below that floor. It is the safer default under
  dependence and it is unusable at this scale. ``power_condition()`` in
  ``scripts/validate_conformal.py`` computes the requirement.
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

#: Whether the procedure here has been measured against the project's own data.
#: True as of ``scripts/validate_conformal.py``: over repeated three-way splits on
#: both risk families, ``E[FDP]`` stayed at or under ``alpha`` at every level, with
#: ``BH``. Read that as the narrow claim it is -- the implementation is sound and
#: the guarantee held on this data. It says nothing about a deployment whose
#: calibration set is not exchangeable with its traffic, and it does not make the
#: power problem go away. The caveats are in the module docstring; this flag is
#: exported rather than left to documentation because a caller can test it and a
#: docstring cannot stop a number from being quoted.
VALIDATED = True

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

    A rejection means "this looks riskier than the safe calibration items did", so
    the selected set is the items to act on. ``realized_fdp`` is the share of them
    that are in fact correct -- the cost of acting -- and ``correct_items_flagged``
    is the share of all correct items that were needlessly flagged.

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
        # A rejection means "this looks riskier than the safe calibration items
        # did", so the selection is the set of items to act on, and a false
        # discovery is a selected item that is in fact **correct**. Counting
        # label == 1 here instead -- which this did -- makes the metric run
        # backwards: a selection that flags every genuinely risky item and nothing
        # else reports an FDP of 1.0, and one that flags only safe items reports
        # 0.0. The guarantee would then be checked against a number that is
        # anti-correlated with the thing it is supposed to measure.
        false_discoveries = int(((labels == 0) & accepted).sum())
        report["false_discoveries"] = false_discoveries
        report["realized_fdp"] = false_discoveries / selected
        # Of the correct items, the share that was needlessly flagged. This is the
        # companion cost to coverage: a selection that flags everything has perfect
        # recall and sends every correct answer to a human.
        report["correct_items_flagged"] = (
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
