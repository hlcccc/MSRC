#!/usr/bin/env python
"""The whole indicator table in one command, straight from the cell manifest.

The report quotes four indicators across fourteen cells. They used to be assembled by
hand from several scripts, which is how the deck ended up three hours behind the metric
correction. This prints all of them from ``runs/cells.json`` so re-running is one command
and a stale number cannot survive a rebuild.

What each indicator is, and what it is measured against:

``2.1 不确定性信号种类``
    How many independent readings the method defines. A property of the code, not of any
    dataset -- and the number that matters is not how many are *declared* but how many
    produced a value on the data, because a declared-but-constant column carries nothing.
    Both are printed.

``2.2 校准提升``
    The relative fall in expected calibration error when the calibrator replaces the
    **strongest single signal**. The baseline is the strongest signal rather than a
    near-random one, because a gain against a weak reference would prove little; it is
    chosen on dev by ``|AUROC - 0.5|`` and oriented, so a signal that points the wrong way
    is still a fair baseline. Reported at several bin counts, since ECE moves with the
    binning and a quoted figure has to name its setting.

``2.3 高风险预警准确率``
    Plain accuracy at the operating point chosen on dev -- the bar the project is judged
    against. Printed beside ``从不预警`` (the same accuracy with nothing ever flagged) and
    beside the threshold ceiling selected on test labels, which says whether a shortfall
    is the operating point or the score.

    python scripts/summary_table.py --manifest runs/cells.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.groups import image_group_id  # noqa: E402
from evaluation.metrics import auroc, ece  # noqa: E402
from evaluation.split import DEFAULT_DEV_FRACTION, DEFAULT_SEED, grouped_split  # noqa: E402
from msrc.model import RiskCalibrator  # noqa: E402
from msrc.signals import ALL_SIGNALS, Evidence, SIGNAL_NAMES, build_report  # noqa: E402

GRID = np.round(np.linspace(0.0, 1.0, 201), 4)
BINS = (5, 10, 15, 20)
MAIN_BINS = 15
SIGNAL_BAR = 4
ECE_BAR = 0.20
ACCURACY_BAR = 0.85


def load_cell(path: Path):
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    matrix, labels, groups = [], [], []
    for row in rows:
        report = build_report(Evidence.from_dict(row["evidence"]))
        matrix.append(report.vector())
        labels.append(int(row["label"]))
        groups.append(image_group_id(row))
    return np.asarray(matrix, float), np.asarray(labels, int), np.asarray(groups)


def analyse(matrix, labels, groups, l2=0.05):
    dev, test = grouped_split(groups, dev_fraction=DEFAULT_DEV_FRACTION, seed=DEFAULT_SEED)
    if len(set(labels[dev])) < 2 or len(set(labels[test])) < 2:
        return None

    live = [i for i in range(matrix.shape[1])
            if float(np.ptp(matrix[dev, i])) > 1e-12]
    # The strongest single signal, by distance from chance on dev, oriented.
    per_signal = {SIGNAL_NAMES[i]: auroc(labels[dev], matrix[dev, i]) for i in live}
    best = max(per_signal, key=lambda k: abs(per_signal[k] - 0.5))
    best_index = SIGNAL_NAMES.index(best)
    before = matrix[test, best_index].copy()
    if per_signal[best] < 0.5:
        before = 1.0 - before
    before = np.clip(before, 0.0, 1.0)

    fitter = RiskCalibrator(l2=l2).fit(matrix[dev], labels[dev])
    dev_scores = fitter.predict_proba(matrix[dev])
    accs = np.array([float(((dev_scores >= t).astype(int) == labels[dev]).mean()) for t in GRID])
    threshold = float(GRID[len(GRID) - 1 - int(accs[::-1].argmax())])
    scores = fitter.predict_proba(matrix[test])

    e_before = ece(labels[test], before, n_bins=MAIN_BINS)
    e_after = ece(labels[test], scores, n_bins=MAIN_BINS)
    gain = (e_before - e_after) / e_before if e_before > 0 else float("nan")
    by_bins = {}
    for n in BINS:
        b = ece(labels[test], before, n_bins=n)
        by_bins[n] = (b - ece(labels[test], scores, n_bins=n)) / b if b > 0 else float("nan")

    predicted = (scores >= threshold).astype(int)
    accuracy = float((predicted == labels[test]).mean())
    return {
        "n_test": int(test.sum()),
        "live_signals": len(live),
        "baseline_signal": best,
        "baseline_auroc": float(per_signal[best]),
        "ece_before": float(e_before), "ece_after": float(e_after),
        "ece_gain": float(gain), "ece_by_bins": by_bins,
        "accuracy": accuracy,
        "never_warn": float((labels[test] == 0).mean()),
        "auroc": auroc(labels[test], scores),
        "ceiling": max(float((((scores >= t).astype(int)) == labels[test]).mean()) for t in GRID),
        "passes": accuracy >= ACCURACY_BAR,
        "ece_passes": gain >= ECE_BAR,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    root = Path(payload.get("root", "")).expanduser()
    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}

    print("=" * 122)
    print("指标 2.1  不确定性信号")
    print("=" * 122)
    internal = [s.name for s in ALL_SIGNALS if s.kind == "internal"]
    print("  代码里定义 %d 个（内部 %d + 外部 %d），结题线 ≥ %d 种 → %s"
          % (len(ALL_SIGNALS), len(internal), len(ALL_SIGNALS) - len(internal),
             SIGNAL_BAR, "达标" if len(ALL_SIGNALS) >= SIGNAL_BAR else "未达"))
    print("  逐个：" + "、".join(SIGNAL_NAMES))
    print("  注意：声明几个不等于在数据上跑了几个 —— 每个格实际非恒定的个数见下表。")

    rows = []
    for entry in payload["cells"]:
        if wanted and entry["dataset"] not in wanted:
            continue
        path = Path(entry["path"]).expanduser()
        full = path if (path.is_absolute() or not str(root)) else root / path
        if not full.is_file():
            rows.append({"model": entry["model"], "dataset": entry["dataset"],
                         "missing": True})
            continue
        matrix, labels, groups = load_cell(full)
        stats = analyse(matrix, labels, groups)
        if stats is None:
            rows.append({"model": entry["model"], "dataset": entry["dataset"], "missing": True})
            continue
        stats.update(model=entry["model"], dataset=entry["dataset"], path=str(full))
        rows.append(stats)

    present = [r for r in rows if not r.get("missing")]
    print()
    print("=" * 122)
    print("指标 2.2  校准提升（相对最强单信号，ECE@%d，结题线 ≥ %.0f%%）" % (MAIN_BINS, 100 * ECE_BAR))
    print("=" * 122)
    print("  %-15s %-15s %-28s %10s %10s %9s  %s"
          % ("模型", "数据集", "最强单信号（校准前基准）", "ECE前", "ECE后", "相对提升", "判定"))
    print("  " + "-" * 118)
    for r in present:
        print("  %-15s %-15s %-28s %10.4f %10.4f %8.1f%%  %s"
              % (r["model"], r["dataset"], "%s (AUROC %.3f)" % (r["baseline_signal"], r["baseline_auroc"]),
                 r["ece_before"], r["ece_after"], 100 * r["ece_gain"],
                 "达标" if r["ece_passes"] else "未达"))
    for r in rows:
        if r.get("missing"):
            print("  %-15s %-15s  文件缺失" % (r["model"], r["dataset"]))
    if present:
        print("  分箱敏感性（相对提升）：" + "  ".join(
            "n_bins=%d 中位 %.1f%%" % (n, 100 * float(np.median([r["ece_by_bins"][n] for r in present])))
            for n in BINS))

    print()
    print("=" * 122)
    print("指标 2.3  高风险预警准确率（普通准确率，结题线 ≥ %.0f%%）" % (100 * ACCURACY_BAR))
    print("=" * 122)
    print("  %-15s %-15s %7s %8s %10s %10s %9s %9s  %s"
          % ("模型", "数据集", "测试n", "幻觉率", "准确率", "从不预警", "上界", "AUROC", "判定"))
    print("  " + "-" * 118)
    for r in present:
        print("  %-15s %-15s %7d %7.1f%% %9.2f%% %9.2f%% %8.2f%% %9.4f  %s"
              % (r["model"], r["dataset"], r["n_test"], 100 * (1 - r["never_warn"]),
                 100 * r["accuracy"], 100 * r["never_warn"], 100 * r["ceiling"], r["auroc"],
                 "达标" if r["passes"] else "未达"))
    for r in rows:
        if r.get("missing"):
            print("  %-15s %-15s  文件缺失" % (r["model"], r["dataset"]))

    print()
    models = sorted({r["model"] for r in present})
    print("  按模型的达标数：")
    for m in models:
        got = [r["dataset"] for r in present if r["model"] == m and r["passes"]]
        all_ds = [r["dataset"] for r in present if r["model"] == m]
        print("    %-15s %d/%d  %s" % (m, len(got), len(all_ds), "、".join(sorted(got))))

    if args.json:
        print()
        print(json.dumps({"cells": present, "bars": {"signals": SIGNAL_BAR,
                                                     "ece": ECE_BAR,
                                                     "accuracy": ACCURACY_BAR}},
                         indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
