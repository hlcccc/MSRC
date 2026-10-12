#!/usr/bin/env python
"""Would more auxiliary re-readings actually close the gap? Answer it before paying.

``object_support`` reads the objects the answer names and asks, of each, how often it
comes back across the other looks at the image -- four re-phrasings plus K resamples.
Its reading is therefore a *rate estimated from a finite number of looks*, and part of
its error is sampling error. That raises a question the GPU cannot answer cheaply:
would thirteen looks instead of seven move the accuracy, or is the signal already at
its own ceiling?

This script answers it on a CPU, on the development half, before anything is
collected. It measures the same signal at a **controllable number of looks** by
resampling the looks that already exist:

``sub``   draw k of the stored looks without replacement. Valid only for
          k <= what was collected, so it gives the *measured* curve.
``boot``  draw k looks with replacement. Valid for any k, so it reaches past what was
          collected -- and, at k equal to the collected count, it has to reproduce the
          measured value, which is the check that makes the extrapolation readable.

Two independent estimates of the gain come out of that, and they should be read
together:

* the ``sub`` curve is fitted with ``AUROC(k) = A - c/sqrt(k)``. A rate estimated from
  k looks has standard error falling as ``1/sqrt(k)``, so this is the shape the
  estimator's own noise predicts, and ``A`` is the ceiling this signal can reach no
  matter how many looks are bought. If ``A`` is below what the cell needs, the
  collection is not worth running.
* the ``boot`` curve at the target k, against the collected k.

Everything here is development-half only; the test half is not read, because this is
the analysis that decides whether an experiment happens. Accuracy is estimated inside
dev by cross-fitting -- fit and pick the threshold on one group-disjoint fold, score
the other -- so the number is an honest stand-in for the test half rather than the
optimistic dev-on-dev one.

    python scripts/reading_budget.py \
        --cell "LLaVA-1.5-13B|CHAIR=evidence_chair.jsonl" \
        --cell "Qwen2.5-VL-7B|CHAIR=cell_Qwen2.5-VL-7B_chair.jsonl" \
        --target-readings 13
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.groups import image_group_id  # noqa: E402
from evaluation.metrics import auroc  # noqa: E402
from evaluation.split import DEFAULT_DEV_FRACTION, DEFAULT_SEED, grouped_split  # noqa: E402
from msrc.model import RiskCalibrator  # noqa: E402
from msrc.object_support import default_lexicon, object_min_support  # noqa: E402
from msrc.signals import Evidence, build_report  # noqa: E402

OBJECT_SIGNAL = "object_support"
GRID = np.round(np.linspace(0.0, 1.0, 201), 4)


def load_cell(path: Path):
    """Signal matrix, labels, image groups and the per-item pool of auxiliary looks."""
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")

    matrix, labels, groups, pools, primaries, names = [], [], [], [], [], None
    for row in rows:
        evidence = Evidence.from_dict(row["evidence"])
        report = build_report(evidence)
        if names is None:
            names = [s.name for s in report.signals]
        matrix.append(report.vector())
        labels.append(int(row["label"]))
        groups.append(image_group_id(row))
        pools.append(evidence.texts("views") + evidence.texts("resamples"))
        primaries.append(evidence.answer or evidence.primary.text)
    return (np.asarray(matrix, dtype=float), np.asarray(labels, dtype=int),
            np.asarray(groups), pools, primaries, names)


def support_column(pools, primaries, k, rng, lexicon, mode):
    """The object_support risk column as it would be from ``k`` looks.

    ``mode='sub'`` samples without replacement -- the reading is genuinely taken from
    the stored looks. ``mode='boot'`` samples with replacement, which is how a k past
    the stored count is simulated: the stored looks stand in for the distribution of
    looks on that item, and k draws are taken from them.
    """
    column = np.full(len(pools), np.nan)
    for index, pool in enumerate(pools):
        if not pool:
            continue
        if mode == "sub":
            if k > len(pool):
                continue
            drawn = rng.sample(pool, k)
        elif mode == "boot":
            drawn = [pool[rng.randrange(len(pool))] for _ in range(k)]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        reading = object_min_support(primaries[index], drawn, lexicon)
        if reading is not None:
            column[index] = 1.0 - reading[0]
    return column


def with_column(base, index, column):
    """``base`` with one signal column replaced. Missing readings enter as zero.

    Zero is what the pipeline itself does with an unavailable signal, so the
    simulated matrix goes through the same calibrator as the real one.
    """
    out = base.copy()
    out[:, index] = np.where(np.isnan(column), 0.0, column)
    return out


def accuracy_optimum(scores, labels, grid=GRID) -> float:
    """Grid threshold maximising plain accuracy; ties go to the higher threshold."""
    values = np.array([float(((scores >= t).astype(int) == labels).mean()) for t in grid])
    return float(grid[len(grid) - 1 - int(values[::-1].argmax())])


def cross_fitted_accuracy(matrix, labels, groups, fold_seed, l2) -> float:
    """Accuracy with the fit and the threshold chosen away from the scored items.

    Fitting and thresholding on the same items would report the dev half's own
    optimum, which is not what the test half sees. Halving dev makes this a little
    pessimistic, equally so at every reading count, and the comparison across counts
    is what this script is for.
    """
    inner, _ = grouped_split(groups, dev_fraction=0.5, seed=fold_seed)
    correct, seen = 0, 0
    for fit_mask, held_mask in ((inner, ~inner), (~inner, inner)):
        if len(set(labels[fit_mask])) < 2 or not held_mask.any():
            continue
        fitter = RiskCalibrator(l2=l2).fit(matrix[fit_mask], labels[fit_mask])
        threshold = accuracy_optimum(fitter.predict_proba(matrix[fit_mask]), labels[fit_mask])
        predicted = (fitter.predict_proba(matrix[held_mask]) >= threshold).astype(int)
        correct += int((predicted == labels[held_mask]).sum())
        seen += int(held_mask.sum())
    return correct / seen if seen else float("nan")


def fit_inverse_sqrt(counts, values):
    """Least-squares ``A - c/sqrt(k)``. Returns ``(A, c, residual)``."""
    x = 1.0 / np.sqrt(np.asarray(counts, dtype=float))
    y = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    return float(intercept), float(-slope), float(np.abs(predicted - y).max())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cell", action="append", required=True, metavar="MODEL|DATASET=PATH")
    parser.add_argument("--target-readings", type=int, default=13,
                        help="how many auxiliary looks the collection would buy")
    parser.add_argument("--repeats", type=int, default=10,
                        help="resampling repeats per reading count, averaged")
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    lexicon = default_lexicon()
    reports = []

    for spec in args.cell:
        if "=" not in spec or "|" not in spec.split("=", 1)[0]:
            raise SystemExit(f"--cell must be MODEL|DATASET=PATH, got {spec!r}")
        head, raw_path = spec.split("=", 1)
        model, dataset = head.split("|", 1)
        path = Path(raw_path).expanduser()
        if not path.is_file():
            print(f"### {model} | {dataset}: 文件缺失（{path}）")
            continue

        X, y, groups, pools, primaries, names = load_cell(path)
        object_index = names.index(OBJECT_SIGNAL)
        dev, _test = grouped_split(groups, dev_fraction=args.dev_fraction, seed=args.seed)
        if len(set(y[dev])) < 2:
            print(f"### {model} | {dataset}: 开发半区只有一个类")
            continue

        collected = sorted({len(p) for p in pools})
        print("=" * 100)
        print(f"### {model} | {dataset}   n={len(y)}   开发半区 {int(dev.sum())}")
        print(f"    每条已采辅助读数：{collected}"
              f"{'（每条一致）' if len(collected) == 1 else '（不一致）'}"
              f"   主答案非空 {sum(1 for t in primaries if t)}/{len(primaries)}")

        base = X.copy()
        base[:, object_index] = 0.0
        auroc_without = auroc(y[dev], RiskCalibrator(l2=args.l2).fit(
            base[dev], y[dev]).predict_proba(base[dev]))
        print(f"    不含 object_support 的融合 dev AUROC = {auroc_without:.4f}")

        max_stored = min(len(p) for p in pools if p) if any(pools) else 0
        sub_counts = list(range(2, min(max_stored, 7) + 1))
        boot_counts = [2, 3, 4, 5, 6, 7, 9, 11, args.target_readings]
        boot_counts = sorted({c for c in boot_counts if c >= 2})

        curves = {}
        for mode, counts in (("sub", sub_counts), ("boot", boot_counts)):
            rows_out = []
            for k in counts:
                solo, fused, acc = [], [], []
                for repeat in range(args.repeats):
                    # The dev/test split is fixed; only the look-draw varies, so the
                    # repeats are paired on the same items and the differences are
                    # comparable without a second source of variation. The seed is
                    # arithmetic rather than hash() so that a run is reproducible
                    # across processes -- hash() of a tuple is salted per interpreter.
                    rng = random.Random(
                        args.seed * 1000003 + k * 1009 + repeat + (0 if mode == "sub" else 500000)
                    )
                    column = support_column(pools, primaries, k, rng, lexicon, mode)
                    usable = ~np.isnan(column) & dev
                    if usable.sum() < 50 or len(set(y[usable])) < 2:
                        continue
                    solo.append(auroc(y[usable], column[usable]))
                    matrix = with_column(base, object_index, column)
                    fused.append(auroc(y[dev], RiskCalibrator(l2=args.l2).fit(
                        matrix[dev], y[dev]).predict_proba(matrix[dev])))
                    acc.append(cross_fitted_accuracy(
                        matrix[dev], y[dev], groups[dev], fold_seed=repeat, l2=args.l2))
                if solo:
                    rows_out.append((k, float(np.mean(solo)), float(np.std(solo)),
                                     float(np.mean(fused)), float(np.mean(acc))))
            curves[mode] = rows_out

        print()
        print(f"    {'读数个数':<8} {'单信号AUROC':>12} {'（重复间标准差）':>18} "
              f"{'融合AUROC':>11} {'dev交叉拟合准确率':>18}")
        print("    " + "-" * 74)
        for k, solo, sd, fused, acc in curves["sub"]:
            print(f"    {'sub  %d' % k:<8} {solo:>12.4f} {sd:>18.4f} {fused:>11.4f} {acc:>17.1%}")
        for k, solo, sd, fused, acc in curves["boot"]:
            print(f"    {'boot %d' % k:<8} {solo:>12.4f} {sd:>18.4f} {fused:>11.4f} {acc:>17.1%}")

        result = {"model": model, "dataset": dataset, "n": int(len(y))}
        if len(curves["sub"]) >= 3:
            ks = [r[0] for r in curves["sub"]]
            asymptote, c, residual = fit_inverse_sqrt(ks, [r[1] for r in curves["sub"]])
            at_target = asymptote - c / np.sqrt(args.target_readings)
            at_stored = asymptote - c / np.sqrt(max(ks))
            result.update({"auroc_asymptote": asymptote, "fit_max_residual": residual,
                           "auroc_at_stored": at_stored, "auroc_at_target": at_target})
            print()
            print(f"    拟合 AUROC(k) = A - c/sqrt(k)   （只用 sub 实测点，最大残差 {residual:.4f}）")
            print(f"        A（读数再多也到不了的上限） = {asymptote:.4f}")
            print(f"        已采 {max(ks)} 个读数处的拟合值 = {at_stored:.4f}")
            print(f"        加读到 {args.target_readings} 个的预测值 = {at_target:.4f}"
                  f"   （+{at_target - at_stored:.4f}）")
        if curves["boot"]:
            measured = {r[0]: r[1] for r in curves["sub"]}
            stored = next((r for r in curves["boot"] if r[0] == max_stored), None)
            target = next((r for r in curves["boot"] if r[0] == args.target_readings), None)
            if stored is not None and target is not None:
                result.update({"boot_auroc_stored": stored[3], "boot_auroc_target": target[3],
                               "boot_acc_stored": stored[4], "boot_acc_target": target[4],
                               "boot_acc_gain": target[4] - stored[4]})
                print()
                print(f"    重抽样校验（boot 只是统计模型，必须在已采个数上重现实测值）：")
                if max_stored in measured:
                    print(f"        {max_stored} 个读数：boot {stored[1]:.4f} vs 实测 "
                          f"{measured[max_stored]:.4f}   差 {stored[1] - measured[max_stored]:+.4f}")
                print(f"        预测 {args.target_readings} 个读数：融合 AUROC "
                      f"{target[3]:.4f}（{target[3] - stored[3]:+.4f}），"
                      f"dev 交叉拟合准确率 {target[4]:.1%}"
                      f"（{100 * (target[4] - stored[4]):+.1f} 个点）")
        reports.append(result)

    if args.json:
        print()
        print(json.dumps({"cells": reports, "dev_only": True}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
