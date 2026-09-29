#!/usr/bin/env python
"""Is the accuracy bar reachable at *any* operating point?

`evaluate.py` reports the threshold metrics at a fixed 0.5. That is a choice, not a
property of the score, and a shortfall at 0.5 invites the question "would it pass at
a better threshold?" This answers it, and answers it honestly:

* thresholds are swept on **dev**, and the test metrics at the dev-chosen threshold
  are reported beside the default, because a tuned threshold is a fitted parameter
  and choosing it on test would be reporting a number nobody can reproduce;
* the best accuracy **any** threshold achieves on test is printed as a ceiling. It
  is not a result -- it is selected using the test labels -- it is what the score
  cannot exceed, so that "we did not tune it" and "tuning cannot help" stay
  distinguishable.

    python scripts/threshold_analysis.py --data evidence.jsonl

Reads the cached evidence, so it runs on a CPU in seconds and touches no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import auroc, brier, threshold_metrics  # noqa: E402
from evaluation.split import DEFAULT_DEV_FRACTION, DEFAULT_SEED, grouped_split  # noqa: E402
from msrc.model import RiskCalibrator  # noqa: E402
from msrc.signals import Evidence, build_report  # noqa: E402

__all__ = ["DEFAULT_GRID", "accuracy_at", "sweep_accuracy", "pick_threshold"]

#: Thresholds to consider. Fine enough that a genuine optimum is not missed between
#: two grid points, coarse enough that the choice is not fitted to noise.
DEFAULT_GRID = tuple(round(float(t), 4) for t in np.arange(0.05, 0.96, 0.01))


def accuracy_at(scores, labels, threshold: float) -> float:
    """Fraction of items on the correct side of ``threshold``."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.size == 0:
        raise ValueError("no scores to threshold")
    return float(((scores >= threshold).astype(int) == labels).mean())


def sweep_accuracy(scores, labels, grid=DEFAULT_GRID) -> np.ndarray:
    """Accuracy at every threshold in ``grid``, in grid order."""
    return np.array([accuracy_at(scores, labels, t) for t in grid], dtype=float)


def pick_threshold(scores, labels, grid=DEFAULT_GRID, *, rule: str = "accuracy") -> float:
    """The grid threshold that maximises ``rule``.

    ``balanced`` is the mean of the true-positive and true-negative rates, and
    ``youden`` is that minus one -- the same ranking, written the way a ROC plots
    it. Both ignore the base rate, which matters when the classes are unbalanced:
    plain accuracy will happily buy a point of it by predicting the majority class.
    Ties go to the higher threshold, which is the more conservative warning.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if rule == "accuracy":
        values = sweep_accuracy(scores, labels, grid)
    elif rule in ("balanced", "youden"):
        positives = max(int((labels == 1).sum()), 1)
        negatives = max(int((labels == 0).sum()), 1)
        tpr = np.array([((scores >= t) & (labels == 1)).sum() / positives for t in grid])
        tnr = np.array([((scores < t) & (labels == 0)).sum() / negatives for t in grid])
        values = (tpr + tnr) / 2 if rule == "balanced" else tpr + tnr - 1
    else:
        raise ValueError(f"unknown rule {rule!r}; use accuracy, balanced or youden")

    # argmax returns the first maximum, so reverse to prefer the higher threshold.
    best = len(grid) - 1 - int(values[::-1].argmax())
    return float(grid[best])


def load_evidence(path: Path):
    """The signal matrix and labels, the same way `evaluate.py` builds them."""
    rows = []
    for line in Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    rows = [r for r in rows if r.get("evidence")]
    if not rows:
        raise SystemExit(f"{path} holds no records with evidence")

    names, matrix, labels, groups = [], [], [], []
    for row in rows:
        report = build_report(Evidence.from_dict(row["evidence"]))
        if not names:
            names = report.names()
        matrix.append(report.vector())
        labels.append(int(row["label"]))
        groups.append(str(row.get("image", row.get("id", ""))))
    return names, np.asarray(matrix, dtype=float), np.asarray(labels, dtype=int), np.asarray(groups)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="evidence JSONL")
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--default-threshold", type=float, default=0.5,
                        help="the operating point evaluate.py reports at")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    names, X, y, groups = load_evidence(Path(args.data))
    dev_mask, test_mask = grouped_split(
        groups, dev_fraction=args.dev_fraction, seed=args.seed
    )
    if len(set(y[dev_mask])) < 2 or len(set(y[test_mask])) < 2:
        raise SystemExit(
            "one side of the split has a single class, so no threshold can be "
            "evaluated; see scripts/evaluate.py for what that means"
        )

    calibrator = RiskCalibrator(l2=args.l2).fit(X[dev_mask], y[dev_mask])
    dev_scores = calibrator.predict_proba(X[dev_mask])
    test_scores = calibrator.predict_proba(X[test_mask])
    y_dev, y_test = y[dev_mask], y[test_mask]

    rules = {
        f"default {args.default_threshold:g}": args.default_threshold,
        "dev: best accuracy": pick_threshold(dev_scores, y_dev, rule="accuracy"),
        "dev: best balanced accuracy": pick_threshold(dev_scores, y_dev, rule="balanced"),
        "dev: best Youden J": pick_threshold(dev_scores, y_dev, rule="youden"),
    }
    ceiling_threshold = pick_threshold(test_scores, y_test, rule="accuracy")
    ceiling = accuracy_at(test_scores, y_test, ceiling_threshold)

    print("=" * 84)
    print("MSRC threshold analysis")
    print("=" * 84)
    print(f"  items        : {len(y)}  dev {int(dev_mask.sum())} / test {int(test_mask.sum())}")
    print(f"  positive rate: dev {y_dev.mean():.3%} / test {y_test.mean():.3%}")
    print(f"  AUROC        : dev {auroc(y_dev, dev_scores):.4f}  "
          f"test {auroc(y_test, test_scores):.4f}")
    print(f"  Brier        : test {brier(y_test, test_scores):.4f}")
    print()
    print(f"  {'rule':30} {'thr':>6} {'dev acc':>8} {'test acc':>9} "
          f"{'prec':>7} {'rec':>7} {'F1':>7}")
    for label, threshold in rules.items():
        dev_metrics = threshold_metrics(y_dev, dev_scores, threshold=threshold)
        test_metrics = threshold_metrics(y_test, test_scores, threshold=threshold)
        print(f"  {label:30} {threshold:6.2f} {dev_metrics['accuracy']:8.4f} "
              f"{test_metrics['accuracy']:9.4f} {test_metrics['precision']:7.4f} "
              f"{test_metrics['recall']:7.4f} {test_metrics['f1']:7.4f}")
    print()
    print(f"  在 test 上任何阈值能达到的准确率上界: {ceiling:.4f}  "
          f"(阈值 {ceiling_threshold:.2f})")
    print("  这个数是用 test 标签选出来的，所以它是天花板，不是可以报告的结果。")
    print("  它的用处是区分两件事：")
    print("    - 上界明显高于默认阈值的结果 -> 默认操作点选得不好，值得调；")
    print("    - 上界约等于默认阈值的结果   -> 调阈值救不了，差的是分数的分辨能力。")
    print()

    if args.json:
        print(json.dumps({
            "n_items": int(len(y)),
            "n_dev": int(dev_mask.sum()),
            "n_test": int(test_mask.sum()),
            "auroc_dev": auroc(y_dev, dev_scores),
            "auroc_test": auroc(y_test, test_scores),
            "brier_test": brier(y_test, test_scores),
            "operating_points": {
                label: {
                    "threshold": float(threshold),
                    "dev_accuracy": threshold_metrics(y_dev, dev_scores, threshold=threshold)["accuracy"],
                    "test_accuracy": threshold_metrics(y_test, test_scores, threshold=threshold)["accuracy"],
                    "test_metrics": threshold_metrics(y_test, test_scores, threshold=threshold),
                }
                for label, threshold in rules.items()
            },
            "test_accuracy_ceiling": ceiling,
            "test_accuracy_ceiling_threshold": ceiling_threshold,
            "ceiling_is_selected_on_test": True,
        }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
