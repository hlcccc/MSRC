"""Evaluation metrics — NumPy only.

The three numbers this module exists to produce are the ones the project's
indicators are stated in:

* **AUROC** for "can it rank a risky item above a safe one";
* **ECE** for "校准性能", the calibration indicator;
* **accuracy / precision / recall at a threshold** for "预警准确率".

They measure different things and a result is only interpretable when all three
are reported together. AUROC alone says nothing about whether the probabilities
are honest; ECE alone says nothing about whether the classes are separated -- and
a predictor that outputs the base rate for every item has a *perfect* ECE and an
AUROC of 0.5. That is not a hypothetical: it is the first thing a reviewer should
try against a calibration claim, so :func:`ece` is documented as being gameable
and the reporting helper prints all three.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "auroc",
    "brier",
    "ece",
    "reliability_curve",
    "threshold_metrics",
    "summarise",
]


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve, with correct tie handling.

    Computed from the Mann-Whitney U statistic: ties contribute one half, which is
    what makes the value well defined for a score with many repeated values (a
    signal that is constant on a whole band, for instance).
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(s):
        raise ValueError("labels and scores must have the same length")
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("AUROC needs both classes present")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def brier(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Mean squared error of the probability. A proper score, unlike ECE."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have the same length")
    return float(np.mean((p - y) ** 2))


def reliability_curve(
    labels: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 15,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin ``(mean prediction, observed frequency, count)``; empty bins dropped."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have the same length")
    if len(y) == 0:
        raise ValueError("no observations")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy not in ("uniform", "quantile"):
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    if float(p.min()) < 0.0 or float(p.max()) > 1.0:
        raise ValueError(
            "probabilities must lie in [0, 1]; ECE is undefined for raw scores. "
            "Map them through a calibrator first."
        )

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    else:
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))[1:-1]
    index = np.clip(np.digitize(p, edges, right=False), 0, n_bins - 1)

    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        mask = index == b
        n = int(mask.sum())
        if n == 0:
            continue
        conf.append(float(p[mask].mean()))
        acc.append(float(y[mask].mean()))
        cnt.append(n)
    return np.asarray(conf), np.asarray(acc), np.asarray(cnt)


def ece(
    labels: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    """Expected Calibration Error: ``Σ_b (n_b/N) · |freq(b) − mean_pred(b)|``.

    Two properties decide whether a quoted value means anything:

    * it is **binning-dependent**. ``n_bins`` and ``strategy`` change it, so both
      sides of any comparison must use the same setting and the setting must be
      reported. :func:`summarise` therefore returns the value for several
      binnings rather than a single number;
    * it is **gameable**. A constant predictor at the base rate scores exactly 0.
      Never quote an ECE gain without a proper score or a ranking metric beside it.
    """
    conf, acc, cnt = reliability_curve(labels, probabilities, n_bins, strategy)
    if cnt.sum() == 0:
        raise ValueError("no observations")
    return float(np.sum(cnt / cnt.sum() * np.abs(acc - conf)))


def threshold_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    """Confusion-matrix quantities at a fixed operating point.

    "预警准确率" is a *threshold* quantity, not a ranking one, so this is the
    function that answers the indicator as literally written. Whether the reviewer
    means AUROC or these numbers changes the verdict, which is why both are
    reported.
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(s):
        raise ValueError("labels and scores must have the same length")
    pred = s >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    total = max(tp + fp + tn + fn, 1)

    def safe(num: float, den: float) -> float:
        return float(num) / float(den) if den else float("nan")

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    f1 = safe(2 * precision * recall, precision + recall) if precision and recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "specificity": safe(tn, tn + fp),
        "f1": f1,
    }


def summarise(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
    n_bins: int = 15,
    bin_sensitivity: bool = True,
) -> Dict[str, object]:
    """Every number needed to report a result honestly, in one call.

    Prints nothing and decides nothing; it exists so that a summary cannot be
    assembled from a favourable subset by accident.
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)

    out: Dict[str, object] = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "auroc": auroc(y, s),
        "brier": brier(y, s),
        "ece": ece(y, s, n_bins=n_bins),
        "ece_n_bins": n_bins,
        "threshold_metrics": threshold_metrics(y, s, threshold),
    }
    if bin_sensitivity:
        out["ece_by_binning"] = {
            f"n_bins={n}|{st}": ece(y, s, n_bins=n, strategy=st)
            for n in (5, 10, 15, 20)
            for st in ("uniform", "quantile")
        }
    return out


def format_summary(summary: Dict[str, object]) -> str:
    """Render :func:`summarise` for a terminal, with the caveats attached."""
    tm = summary["threshold_metrics"]  # type: ignore[index]
    lines = [
        f"  items            : {summary['n']}   positive rate {summary['positive_rate']:.3%}",
        f"  AUROC            : {summary['auroc']:.4f}",
        f"  Brier            : {summary['brier']:.4f}",
        f"  ECE              : {summary['ece']:.4f}   (n_bins={summary['ece_n_bins']})",
        f"  threshold {tm['threshold']:.2f}   : "
        f"acc {tm['accuracy']:.4f}  prec {tm['precision']:.4f}  "
        f"rec {tm['recall']:.4f}  F1 {tm['f1']:.4f}",
        f"  confusion        : TP={tm['tp']} FP={tm['fp']} TN={tm['tn']} FN={tm['fn']}",
    ]
    by_bin = summary.get("ece_by_binning")
    if isinstance(by_bin, dict):
        spread = f"{min(by_bin.values()):.4f} .. {max(by_bin.values()):.4f}"
        lines.append(f"  ECE across binnings: {spread}   <- quote the setting with the value")
    return "\n".join(lines)
