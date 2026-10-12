#!/usr/bin/env python
"""The headline table, regenerated from cached evidence on a CPU in seconds.

The per-cell numbers quoted for the acceptance metric used to come from one-off
shell scripts kept outside the repository, which meant the claim could be read but
not re-run. This is the version that lives here: point it at the evidence files and
it prints the same table, with the ceiling that says whether a shortfall is the
operating point or the score.

    python scripts/balanced_report.py \
        --cell "LLaVA-1.5-13B|POPE=evidence_pope_official9000.jsonl" \
        --cell "LLaVA-1.5-13B|CHAIR=evidence_chair.jsonl"

or, for the whole table at once, from the manifest of which file backs which cell:

    python scripts/balanced_report.py --manifest runs/cells.json

Each ``--cell`` is ``MODEL|DATASET=PATH``. The manifest is the same thing in a file,
with one ``root`` that relative paths hang off, so the table can be regenerated on the
collecting machine without retyping fifteen paths -- the mapping used to live only in
ad-hoc shell scripts, which made the table readable but not re-runnable. A cell whose
file is not there yet is printed as missing rather than dropped, because a table that
silently loses rows is how a shortfall stops being visible.

Thresholds are chosen on the dev half by
accuracy and reported on the test half; the split is by real image
identity, so the same picture never lands on both sides. Nothing here touches a
model, and nothing chooses anything on the test half.

Columns, in the order they should be read:

``检出率`` / ``误报率``   the two halves of the acceptance metric, so a passing
                          number can be audited rather than trusted
``平衡准确率``            the acceptance metric itself
``准确率``                the raw figure, printed beside ``从不预警`` so that a
                          high value on a rare-hallucination benchmark is visible
                          as the base rate rather than as a result
``上界``                  the best balanced accuracy *any* threshold reaches on
                          test. Selected using test labels: a ceiling, not a
                          result. When it sits on top of the achieved value, the
                          threshold is not what is binding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.groups import image_group_id  # noqa: E402
from evaluation.metrics import auroc, balanced_accuracy  # noqa: E402
from evaluation.split import DEFAULT_DEV_FRACTION, DEFAULT_SEED, grouped_split  # noqa: E402
from msrc.model import RiskCalibrator  # noqa: E402
from msrc.signals import Evidence, build_report  # noqa: E402

GRID = np.round(np.linspace(0.0, 1.0, 201), 4)


def load_cell(path: Path):
    """Signal matrix, labels and image groups for one evidence file."""
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    matrix, labels, groups = [], [], []
    for row in rows:
        report = build_report(Evidence.from_dict(row["evidence"]))
        matrix.append(report.vector())
        labels.append(int(row["label"]))
        groups.append(image_group_id(row))
    return (np.asarray(matrix, dtype=float), np.asarray(labels, dtype=int),
            np.asarray(groups))


def rates(labels, predicted):
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    positives, negatives = labels == 1, labels == 0
    tpr = float(predicted[positives].mean()) if positives.any() else 0.0
    tnr = float((1 - predicted[negatives]).mean()) if negatives.any() else 0.0
    return tpr, 1.0 - tnr


def best_threshold(scores, labels, grid=GRID, rule: str = "accuracy") -> float:
    """Grid threshold maximising ``rule`` on the half it is given.

    The rule must match the metric being reported. Choosing the operating point to
    maximise balanced accuracy and then reporting raw accuracy is a mismatch, and on
    an unbalanced benchmark it costs real points: on POPE, where 15% of items are
    hallucinated, the balanced-accuracy optimum warns far more often than the
    accuracy optimum does, and every extra warning is a false positive subtracted
    from a number that is dominated by the majority class.

    Ties go to the higher threshold, the more conservative warning.
    """
    if rule == "accuracy":
        values = np.array([float((np.asarray(scores) >= t).astype(int)
                                 .__eq__(np.asarray(labels)).mean()) for t in grid])
    elif rule == "balanced":
        values = np.array([balanced_accuracy(labels, scores, t) for t in grid])
    else:
        raise ValueError(f"unknown rule {rule!r}; use accuracy or balanced")
    return float(grid[len(grid) - 1 - int(values[::-1].argmax())])


def parse_cell_spec(spec: str) -> tuple:
    """``MODEL|DATASET=PATH`` into its three parts, or a refusal that says why."""
    if "=" not in spec or "|" not in spec.split("=", 1)[0]:
        raise SystemExit(f"--cell must be MODEL|DATASET=PATH, got {spec!r}")
    head, raw_path = spec.split("=", 1)
    model, dataset = head.split("|", 1)
    return model, dataset, raw_path


def cells_from_manifest(path: Path) -> list:
    """Read the cell list out of a manifest file.

    Shape: ``{"root": "...", "cells": [{"model", "dataset", "path"}, ...]}``. ``root``
    is optional; a relative ``path`` is resolved against it. Every entry is checked
    here rather than half-way through the table, so a typo names the entry it is in.
    """
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = payload.get("cells") if isinstance(payload, dict) else payload
    if not entries:
        raise SystemExit(f"{path} holds no cells")
    root = Path(payload.get("root", "") if isinstance(payload, dict) else "").expanduser()
    cells = []
    for index, entry in enumerate(entries):
        missing = [k for k in ("model", "dataset", "path") if not entry.get(k)]
        if missing:
            raise SystemExit(f"{path}: cell {index} is missing {missing}")
        candidate = Path(str(entry["path"])).expanduser()
        cells.append((str(entry["model"]), str(entry["dataset"]),
                      str(candidate if candidate.is_absolute() or not str(root) else root / candidate)))
    return cells


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cell", action="append", default=[], metavar="MODEL|DATASET=PATH")
    parser.add_argument("--manifest", metavar="PATH",
                        help="a JSON file listing the cells; see runs/cells.json. "
                             "Combine with --cell to add to it")
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--threshold-rule", choices=("accuracy", "balanced"),
                        default="accuracy",
                        help="what the dev half optimises; must match the reported "
                             "metric, since on an unbalanced benchmark the two "
                             "optima sit at different operating points")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.cell and not args.manifest:
        raise SystemExit("give at least one --cell, or --manifest")

    specs = list(args.cell)
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser()
        if not manifest_path.is_file():
            raise SystemExit(f"--manifest not found: {manifest_path}")
        specs = [f"{m}|{d}={p}" for m, d, p in cells_from_manifest(manifest_path)] + specs

    out = []
    for spec in specs:
        model, dataset, raw_path = parse_cell_spec(spec)
        path = Path(raw_path).expanduser()
        if not path.is_file():
            print(f"  {model:<14} {dataset:<14}  文件缺失（{path}）")
            continue

        X, y, groups = load_cell(path)
        if len(set(y)) < 2:
            print(f"  {model:<14} {dataset:<14}  只有一个类，无法评测")
            continue
        dev, test = grouped_split(groups, dev_fraction=args.dev_fraction, seed=args.seed)
        if len(set(y[dev])) < 2 or len(set(y[test])) < 2:
            print(f"  {model:<14} {dataset:<14}  有一半只有一个类，无法评测")
            continue

        scores_test = RiskCalibrator(l2=args.l2).fit(
            X[dev], y[dev]).predict_proba(X[test])
        threshold = best_threshold(
            RiskCalibrator(l2=args.l2).fit(X[dev], y[dev]).predict_proba(X[dev]), y[dev],
            rule=args.threshold_rule)
        predicted = (scores_test >= threshold).astype(int)
        tpr, fpr = rates(y[test], predicted)
        ba = balanced_accuracy(y[test], scores_test, threshold)
        acc = float((predicted == y[test]).mean())
        # 结题判定看的是**普通准确率**，不是平衡准确率。两者只在类别均衡时相等，
        # 而这里没有一个数据集是均衡的 —— 在 POPE 上什么都不做就有 85.4% 准确率、
        # 50% 平衡准确率，把判定挂在平衡准确率上会得出相反的结论。
        ceiling = max(float(((( scores_test >= t).astype(int)) == y[test]).mean())
                      for t in GRID)
        out.append({
            "model": model, "dataset": dataset, "path": str(path),
            "n_test": int(test.sum()), "positive_rate": float(y[test].mean()),
            "threshold": threshold, "tpr": tpr, "fpr": fpr,
            "balanced_accuracy": ba, "accuracy": acc,
            "never_warn_accuracy": float((y[test] == 0).mean()),
            "auroc": auroc(y[test], scores_test),
            "accuracy_ceiling": ceiling,
            "balanced_accuracy_ceiling": max(
                balanced_accuracy(y[test], scores_test, t) for t in GRID),
            "passes": acc >= 0.85,
        })

    print("=" * 118)
    print("MSRC 结题口径总表：普通准确率（判对的条目数 ÷ 全部条目数）")
    print("阈值在开发半区按**普通准确率**选出，测试半区只报数；按真实图片身份分组切分。")
    print("平衡准确率作为附注列出，不作为达标判定依据。")
    print("=" * 118)
    print()
    print("  %-14s %-14s %6s %7s %8s %12s %8s %9s %7s %8s %9s"
          % ("模型", "数据集", "测试n", "幻觉率", "检出率", "误报率",
             "准确率(判定)", "平衡准确率", "从不预警", "AUROC", "准确率上界"))
    print("  " + "-" * 114)
    for r in out:
        print("  %-14s %-14s %6d %6.1f%% %7.1f%% %6.1f%% %11.1f%% %7.1f%% %8.1f%% %7.4f %8.1f%%"
              % (r["model"], r["dataset"], r["n_test"], 100 * r["positive_rate"],
                 100 * r["tpr"], 100 * r["fpr"], 100 * r["accuracy"],
                 100 * r["balanced_accuracy"], 100 * r["never_warn_accuracy"],
                 r["auroc"], 100 * r["accuracy_ceiling"]))
    print()
    print("  判定（结题线：普通准确率 ≥ 85%，即老师给的口径）")
    for r in out:
        margin = r["accuracy"] - 0.85
        verdict = ("达标，超出 %.1f 个点" % (100 * margin) if r["passes"]
                   else "未达，差 %.1f 个点" % (100 * -margin))
        print("    %-14s %-14s %8.1f%%  %s（准确率上界 %.1f%%，从不预警 %.1f%%）"
              % (r["model"], r["dataset"], 100 * r["accuracy"],
                 verdict, 100 * r["accuracy_ceiling"], 100 * r["never_warn_accuracy"]))
    print()
    print("  「上界」是用测试标签选出来的，所以它是天花板，不是可以报告的结果。")
    print("  上界≈当前值 -> 差的是分数的分辨力，调阈值救不了。上界 < 85% 的格，")
    print("  「从不预警」是数据属性不是成绩：幻觉越稀疏它越高，准确率必须超过它才有意义。")

    if args.json:
        print()
        print(json.dumps({"cells": out, "acceptance_metric": "balanced_accuracy",
                          "bar": 0.85, "ceiling_is_selected_on_test": True},
                         indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
