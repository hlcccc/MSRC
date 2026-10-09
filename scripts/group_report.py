#!/usr/bin/env python
"""Metrics per group, not just pooled.

A single pooled number can hide a subset where the method does not work at all, and
on a benchmark whose subsets are not interchangeable it describes no benchmark. POPE
is the clearest case -- `random`, `popular` and `adversarial` draw their negative
examples differently, and pooling them makes the result depend on the mix ratio you
happened to build -- but the same is true of MM-SafetyBench's harm categories.

For every value of the grouping field this prints, on the test split:

* **base rate** -- what a system that never raises a warning scores. This is the
  number an accuracy has to beat before it means anything. On a group where the
  model is right 89% of the time, 89% accuracy is the floor, not a result.
* **accuracy at the default threshold** and **at a threshold chosen on dev**. Both
  are reportable; the second is a fitted parameter and is reported as one.
* **the ceiling** -- the best accuracy any threshold reaches, selected on test
  labels. Not a result, and printed only so that "the operating point was badly
  chosen" and "the score cannot separate these classes" stay distinguishable.

    python scripts/group_report.py --data evidence.jsonl --group-by split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import auroc, brier, threshold_metrics  # noqa: E402
from evaluation.groups import image_group_id  # noqa: E402
from evaluation.split import DEFAULT_DEV_FRACTION, DEFAULT_SEED, grouped_split  # noqa: E402
from msrc.model import RiskCalibrator  # noqa: E402
from msrc.signals import Evidence, build_report  # noqa: E402

__all__ = ["GRID", "accuracy_at", "summarise_group"]

GRID = tuple(round(float(t), 4) for t in np.arange(0.05, 0.96, 0.01))


def accuracy_at(scores, labels, threshold: float) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.size == 0:
        raise ValueError("no scores to threshold")
    return float(((scores >= threshold).astype(int) == labels).mean())


def summarise_group(X, y, groups, *, seed: int, dev_fraction: float, l2: float) -> dict:
    """Everything reportable about one group. Pure, so it is tested without a model."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    dev_mask, test_mask = grouped_split(groups, dev_fraction=dev_fraction, seed=seed)
    if len(set(y[dev_mask])) < 2 or len(set(y[test_mask])) < 2:
        return {"n": int(len(y)), "evaluable": False}

    calibrator = RiskCalibrator(l2=l2).fit(X[dev_mask], y[dev_mask])
    dev_scores = calibrator.predict_proba(X[dev_mask])
    test_scores = calibrator.predict_proba(X[test_mask])
    y_dev, y_test = y[dev_mask], y[test_mask]

    grid = np.asarray(GRID)
    dev_accs = np.array([accuracy_at(dev_scores, y_dev, t) for t in grid])
    test_accs = np.array([accuracy_at(test_scores, y_test, t) for t in grid])
    # Ties go to the higher threshold: it warns less, which on a tie is the choice
    # that does not manufacture risk.
    dev_pick = len(grid) - 1 - int(dev_accs[::-1].argmax())
    best_test = len(grid) - 1 - int(test_accs[::-1].argmax())

    at_half = threshold_metrics(y_test, test_scores, threshold=0.5)
    at_dev = threshold_metrics(y_test, test_scores, threshold=float(grid[dev_pick]))
    threshold = float(grid[best_test])
    positives = max(int((y_test == 1).sum()), 1)
    negatives = max(int((y_test == 0).sum()), 1)
    tpr = ((test_scores >= threshold) & (y_test == 1)).sum() / positives
    tnr = ((test_scores < threshold) & (y_test == 0)).sum() / negatives

    return {
        "n": int(len(y)),
        "n_test": int(test_mask.sum()),
        "evaluable": True,
        "error_rate": float(y.mean()),
        "base_rate": float(max(y_test.mean(), 1 - y_test.mean())),
        "auroc": float(auroc(y_test, test_scores)),
        "brier": float(brier(y_test, test_scores)),
        "accuracy_at_0.5": float(at_half["accuracy"]),
        "dev_threshold": float(grid[dev_pick]),
        "accuracy_at_dev_threshold": float(at_dev["accuracy"]),
        "test_accuracy_ceiling": float(test_accs[best_test]),
        "ceiling_threshold": threshold,
        "balanced_accuracy": float((tpr + tnr) / 2),
        "precision_at_0.5": float(at_half["precision"]),
        "recall_at_0.5": float(at_half["recall"]),
        "f1_at_0.5": float(at_half["f1"]),
    }


def load_evidence(path: Path, group_by: str):
    rows = []
    for line in Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    rows = [r for r in rows if r.get("evidence")]
    if not rows:
        raise SystemExit(f"{path} holds no records with evidence")
    if not any(group_by in r for r in rows):
        raise SystemExit(
            f"no record carries {group_by!r}; pass --group-by with a field the "
            "evidence has (category, split, dataset, ...)"
        )

    matrix, labels, groups, keys = [], [], [], []
    for row in rows:
        report = build_report(Evidence.from_dict(row["evidence"]))
        matrix.append(report.vector())
        labels.append(int(row["label"]))
        groups.append(image_group_id(row))
        keys.append(str(row.get(group_by, "?")))
    return (
        np.asarray(matrix, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(groups),
        np.asarray(keys),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="evidence JSONL")
    parser.add_argument("--group-by", default="split", help="field to group on")
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--min-n", type=int, default=20,
                        help="skip groups smaller than this")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    X, y, groups, keys = load_evidence(Path(args.data), args.group_by)

    print("=" * 96)
    print(f"MSRC group report  (grouped by {args.group_by})")
    print("=" * 96)
    print(f"  items {len(y)}   groups {len(set(keys.tolist()))}   "
          f"seed {args.seed}   dev-fraction {args.dev_fraction}")
    print()
    header = (f"  {'group':16} {'n':>4} {'err':>7} {'AUROC':>7} {'acc@.5':>7} "
              f"{'thr(dev)':>8} {'acc(dev)':>9} {'ceiling':>8} {'base':>7} "
              f"{'bal acc':>8} {'F1':>6}")
    print(header)

    results = {}
    order = [k for k in dict.fromkeys(keys.tolist())]
    for key in order:
        mask = keys == key
        if int(mask.sum()) < args.min_n:
            continue
        summary = summarise_group(
            X[mask], y[mask], groups[mask],
            seed=args.seed, dev_fraction=args.dev_fraction, l2=args.l2,
        )
        results[key] = summary
        if not summary["evaluable"]:
            print(f"  {key:16} {summary['n']:4d}   single class on one side")
            continue
        print(f"  {key:16} {summary['n']:4d} {summary['error_rate']:7.2%} "
              f"{summary['auroc']:7.4f} {summary['accuracy_at_0.5']:7.4f} "
              f"{summary['dev_threshold']:8.2f} "
              f"{summary['accuracy_at_dev_threshold']:9.4f} "
              f"{summary['test_accuracy_ceiling']:8.4f} {summary['base_rate']:7.4f} "
              f"{summary['balanced_accuracy']:8.4f} {summary['f1_at_0.5']:6.4f}")

    pooled = summarise_group(
        X, y, groups, seed=args.seed, dev_fraction=args.dev_fraction, l2=args.l2
    )
    results["ALL (pooled)"] = pooled
    if pooled["evaluable"]:
        print(f"  {'ALL (pooled)':16} {pooled['n']:4d} {pooled['error_rate']:7.2%} "
              f"{pooled['auroc']:7.4f} {pooled['accuracy_at_0.5']:7.4f} "
              f"{pooled['dev_threshold']:8.2f} "
              f"{pooled['accuracy_at_dev_threshold']:9.4f} "
              f"{pooled['test_accuracy_ceiling']:8.4f} {pooled['base_rate']:7.4f} "
              f"{pooled['balanced_accuracy']:8.4f} {pooled['f1_at_0.5']:6.4f}")

    print()
    print("  acc@.5 and acc(dev) are reportable -- the second uses a threshold fitted")
    print("  on dev, which is a fitted parameter and has to be said out loud. The")
    print("  ceiling picks its threshold on test labels and is not a result.")
    print()
    print("  达标情况 (accuracy >= 85%), 取可报告口径里更好的那个:")
    for key, summary in results.items():
        if not summary.get("evaluable"):
            continue
        best = max(summary["accuracy_at_0.5"], summary["accuracy_at_dev_threshold"])
        meets = best >= 0.85
        beats = best > summary["base_rate"]
        print(f"    {key:16} acc={best:.4f} {'达标' if meets else '未达'}   "
              f"{'高于基线' if beats else '不高于基线 —— 没有预警能力'}")

    if args.json:
        print()
        print(json.dumps({
            "group_by": args.group_by,
            "seed": args.seed,
            "dev_fraction": args.dev_fraction,
            "groups": results,
        }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
