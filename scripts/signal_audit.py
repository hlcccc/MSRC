#!/usr/bin/env python
"""Audit the method for uniformity across datasets.

A diagnostic, not a rule. It reports where each signal's strength actually sits, so
that a cell leaning on a benchmark-shaped reading can be described that way when the
number is quoted. Signals specialised to one benchmark are legitimate; the point is
that a reader who takes such a cell at face value will over-generalise it.

Two things are measured, both on the dev half only:

1. **Availability.** For each signal and each cell, how many items supply a reading.
   A signal that is available on one cell and absent on five is already suspect
   before any accuracy is looked at.

2. **Leave-one-out contribution.** Refit the fusion with one signal removed and
   report the AUROC it costs that cell. A signal that is worth +0.19 on one cell and
   within noise on all others is doing per-dataset work, whatever its docstring says.

The summary at the end ranks signals by how *concentrated* their contribution is:
the spread between the cell where they matter most and their median elsewhere, in
units of the per-cell AUROC noise. High concentration is the flag, and the flag does
not care whether the concentration was intended.

    python scripts/signal_audit.py --cell "LLaVA-1.5-13B|POPE=evidence.jsonl" ...
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.groups import image_group_id  # noqa: E402
from evaluation.metrics import auroc  # noqa: E402
from evaluation.split import DEFAULT_DEV_FRACTION, DEFAULT_SEED, grouped_split  # noqa: E402
from msrc.model import RiskCalibrator  # noqa: E402
from msrc.signals import ALL_SIGNALS, Evidence, build_report  # noqa: E402

#: Leave-one-out deltas below this are inside the noise of an AUROC at these sizes
#: and are reported as "no measurable contribution" rather than ranked.
NOISE = 0.004


def load_cell(path: Path, names):
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    X = np.zeros((len(rows), len(names)))
    A = np.zeros((len(rows), len(names)), dtype=bool)
    y = np.zeros(len(rows), dtype=int)
    groups = []
    for i, row in enumerate(rows):
        report = build_report(Evidence.from_dict(row["evidence"]))
        by_name = {s.name: s for s in report.signals}
        for j, name in enumerate(names):
            s = by_name[name]
            X[i, j] = s.risk if s.available else 0.0
            A[i, j] = s.available
        y[i] = int(row["label"])
        groups.append(image_group_id(row))
    return X, A, y, np.asarray(groups)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", action="append", required=True,
                        metavar="MODEL|DATASET=PATH")
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    names = [s.name for s in ALL_SIGNALS]
    cells, availability, contribution = [], {}, {}

    for spec in args.cell:
        head, raw = spec.split("=", 1)
        model, dataset = head.split("|", 1)
        path = Path(raw).expanduser()
        if not path.is_file():
            print(f"  跳过（文件不存在）：{model} | {dataset}")
            continue
        X, A, y, groups = load_cell(path, names)
        dev, test = grouped_split(groups, dev_fraction=args.dev_fraction, seed=args.seed)
        if len(set(y[dev])) < 2:
            print(f"  跳过（开发半区只有一个类）：{model} | {dataset}")
            continue
        key = f"{model} | {dataset}"
        cells.append(key)
        availability[key] = {n: float(A[dev, j].mean()) for j, n in enumerate(names)}

        full = RiskCalibrator(l2=args.l2).fit(X[dev], y[dev]).predict_proba(X[dev])
        base = auroc(y[dev], full)
        contribution[key] = {"__full__": base}
        for j, n in enumerate(names):
            keep = [k for k in range(len(names)) if k != j]
            s = RiskCalibrator(l2=args.l2).fit(
                X[dev][:, keep], y[dev]).predict_proba(X[dev][:, keep])
            contribution[key][n] = base - auroc(y[dev], s)

    print("=" * 108)
    print("方法一致性审计  —— 一份方法，跑遍每个数据集；只在某一格起作用的信号就是按数据集打的补丁")
    print("只用开发半区。")
    print("=" * 108)
    print()

    print("【一】可用率：这个信号在这些格上到底有没有读数")
    print()
    print("  %-24s %s" % ("信号", "  ".join("%-13s" % c.split(" | ")[1][:13] for c in cells)))
    print("  " + "-" * (24 + 15 * len(cells)))
    for n in names:
        row = "  ".join("%-12.1f%%" % (100 * availability[c][n]) for c in cells)
        print("  %-24s %s" % (n, row))
    print()

    print("【二】留一贡献：去掉这个信号，那一格的 AUROC 掉多少（越大越重要）")
    print()
    print("  %-24s %s" % ("信号", "  ".join("%-13s" % c.split(" | ")[1][:13] for c in cells)))
    print("  " + "-" * (24 + 15 * len(cells)))
    for n in names:
        row = "  ".join("%+12.4f " % contribution[c][n] for c in cells)
        print("  %-24s %s" % (n, row))
    print()

    print("【三】集中度排名：最高贡献 减去 其余格的中位贡献")
    print()
    print("  %-24s %9s %9s %9s   %s" % ("信号", "最高贡献", "其余中位", "集中度", "判读"))
    print("  " + "-" * 88)
    flags = []
    for n in names:
        vals = [contribution[c][n] for c in cells]
        if not vals:
            continue
        top = max(vals)
        rest = [v for v in vals if v != top]
        med = statistics.median(rest) if rest else 0.0
        concentration = top - med
        if top < NOISE:
            verdict = "各格都无贡献"
        elif concentration > 4 * NOISE and len(rest) >= 2:
            verdict = "★ 只在个别格起作用，疑按数据集定制"
            flags.append(n)
        else:
            verdict = "各格贡献相近，方法一致"
        print("  %-24s %+9.4f %+9.4f %+9.4f   %s" % (n, top, med, concentration, verdict))
    print()
    if flags:
        print("  需要处理的信号：%s" % "、".join(flags))
        print("  处理方式不是把阈值调松，而是：要么让它用与数据集无关的通用资源，")
        print("  要么把它从方法里去掉 —— 否则整张表反映的是补丁的数量，不是方法的强弱。")
    else:
        print("  没有信号表现出按数据集定制的特征。")

    if args.json:
        print()
        print(json.dumps({"cells": cells, "availability": availability,
                          "leave_one_out": contribution, "flags": flags,
                          "noise_floor": NOISE}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
