#!/usr/bin/env python3
"""课题二 指标 2.1—2.3 验收报告：一条命令跑完全流程。

第三方测评方只需这一条命令即可复现全部结论，不需要 GPU——所有数字都来自
已采集的证据缓存，同一份缓存跑两次得到同一份报告：

    python scripts/acceptance_report.py --manifest runs/cells.json --out report.md

报告分三节，对应三条考核指标：

    2.1 不确定性信号协同建模   逐格列出实际有读数的信号、读不到的信号及其原因
    2.2 不确定性来源分析       同等校准单信号基线与本方法的 ECE，相对提升率
    2.3 多模态生成内容风险评估 生成前预警的普通准确率、从不预警基线、混淆矩阵

口径固定，且与仓库内评测脚本一致：
    * 按图像身份分组划分开发集与测试集，种子 20260920，开发比例 0.5
    * 阈值只在开发集上按准确率选取，测试集只评一次
    * ECE 取 15 个等宽箱，并给出 5/10/15/20 的分箱敏感性
    * 预警准确率 = (TP + TN) / N，同时报告从不预警基线（即正常样本占比）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.groups import image_group_id  # noqa: E402
from evaluation.metrics import auroc, ece  # noqa: E402
from evaluation.split import (  # noqa: E402
    DEFAULT_DEV_FRACTION,
    DEFAULT_SEED,
    grouped_split,
)
from msrc.model import RiskCalibrator  # noqa: E402
from msrc.signals import ALL_SIGNALS, SIGNAL_NAMES, Evidence, build_report  # noqa: E402

GRID = np.round(np.linspace(0.0, 1.0, 201), 4)
MAIN_BINS = 15
BINS = (5, 10, 15, 20)
L2 = 0.05
SIGNAL_BAR = 4          # 指标 2.1 结题线
ECE_BAR = 0.20          # 指标 2.2 结题线
ACCURACY_BAR = 0.85     # 指标 2.3 结题线
INTERNAL = {s.name for s in ALL_SIGNALS if s.kind == "internal"}


def load_cell(path: Path):
    """Read one evidence file into (features, labels, groups, availability)."""
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    unlabelled = sum(1 for r in rows if "label" not in r)
    if unlabelled:
        raise ValueError(
            "%d of %d rows carry no 'label' -- a collection that has not been "
            "labelled yet cannot be scored. Run scripts/apply_labels.py first."
            % (unlabelled, len(rows)))
    matrix, labels, groups, avail = [], [], [], []
    for row in rows:
        report = build_report(Evidence.from_dict(row["evidence"]))
        matrix.append(report.vector())
        avail.append([s.available for s in report.signals])
        labels.append(int(row["label"]))
        groups.append(image_group_id(row))
    return (np.asarray(matrix, float), np.asarray(labels, int),
            np.asarray(groups), np.asarray(avail, bool))


def analyse(matrix, labels, groups, avail, missing_policy="mean"):
    """All three indicators for one cell."""
    dev, test = grouped_split(groups, dev_fraction=DEFAULT_DEV_FRACTION,
                              seed=DEFAULT_SEED)
    if len(set(labels[dev])) < 2 or len(set(labels[test])) < 2:
        return None

    # --- 2.1 which signals actually produced a reading on this cell
    live_idx = [i for i in range(matrix.shape[1]) if float(np.ptp(matrix[dev, i])) > 1e-12]
    dead = {}
    for i, name in enumerate(SIGNAL_NAMES):
        if i in live_idx:
            continue
        rate = float(avail[:, i].mean()) if avail.shape[0] else 0.0
        dead[name] = ("整格不可用" if rate == 0.0 else "部分可用 %.1f%%" % (100 * rate))

    # --- unreadable cells: fill from the development half only
    X = matrix
    if missing_policy == "mean":
        fill = np.zeros(matrix.shape[1], dtype=float)
        for i in range(matrix.shape[1]):
            seen = matrix[dev][avail[dev][:, i], i]
            fill[i] = float(seen.mean()) if seen.size else 0.0
        X = np.where(avail, matrix, fill[None, :])

    # --- 2.2 ECE: two baselines, see the audit note in README
    per_signal = {SIGNAL_NAMES[i]: auroc(labels[dev], X[dev, i]) for i in live_idx}
    best = max(per_signal, key=lambda k: abs(per_signal[k] - 0.5))
    bi = SIGNAL_NAMES.index(best)
    raw = X[test, bi].copy()
    if per_signal[best] < 0.5:
        raw = 1.0 - raw
    raw = np.clip(raw, 0.0, 1.0)
    cal = RiskCalibrator(l2=L2).fit(X[dev, [bi]], labels[dev]).predict_proba(X[test, [bi]])
    fus_model = RiskCalibrator(l2=L2).fit(X[dev], labels[dev])
    fus = fus_model.predict_proba(X[test])

    e_raw = ece(labels[test], raw, n_bins=MAIN_BINS)
    e_cal = ece(labels[test], cal, n_bins=MAIN_BINS)
    e_fus = ece(labels[test], fus, n_bins=MAIN_BINS)
    gain_cal = (e_cal - e_fus) / e_cal * 100 if e_cal > 0 else float("nan")
    gain_raw = (e_raw - e_fus) / e_raw * 100 if e_raw > 0 else float("nan")
    by_bins = {}
    for n in BINS:
        b = ece(labels[test], cal, n_bins=n)
        by_bins[n] = (b - ece(labels[test], fus, n_bins=n)) / b * 100 if b > 0 else float("nan")

    # --- 2.3 warning accuracy at the operating point chosen on dev
    dev_scores = fus_model.predict_proba(X[dev])
    accs = np.array([float(((dev_scores >= t).astype(int) == labels[dev]).mean())
                     for t in GRID])
    threshold = float(GRID[len(GRID) - 1 - int(accs[::-1].argmax())])
    pred = (fus >= threshold).astype(int)
    y = labels[test]
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    accuracy = (tp + tn) / max(len(y), 1)

    return {
        "n_total": int(matrix.shape[0]),
        "n_dev": int(dev.sum()), "n_test": int(test.sum()),
        "live_signals": len(live_idx),
        "live_names": [SIGNAL_NAMES[i] for i in live_idx],
        "dead_signals": dead,
        "baseline_signal": best,
        "ece_raw": float(e_raw), "ece_calibrated": float(e_cal), "ece_fusion": float(e_fus),
        "gain_vs_calibrated": float(gain_cal), "gain_vs_raw": float(gain_raw),
        "gain_by_bins": by_bins,
        "auroc": float(auroc(y, fus)),
        "threshold": threshold,
        "accuracy": float(accuracy),
        "never_warn": float((y == 0).mean()),
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "pass_signals": len(live_idx) >= SIGNAL_BAR,
        "pass_ece": bool(gain_cal >= ECE_BAR * 100),
        "pass_accuracy": bool(accuracy >= ACCURACY_BAR),
    }


def fmt_pct(x, digits=2):
    return "%.*f%%" % (digits, 100 * x) if x == x else "n/a"


def render_report(payload, root, results, missing_policy):
    L = []
    L.append("# 课题二 考核指标测试报告（指标 2.1—2.3）")
    L.append("")
    L.append("| 项目 | 内容 |")
    L.append("|---|---|")
    L.append("| 报告生成时间 | %s |" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    L.append("| 证据清单 | `%s` |" % payload.get("_manifest", ""))
    L.append("| 证据根目录 | `%s` |" % root)
    L.append("| 缺失值处理 | `%s` |" % missing_policy)
    L.append("| 划分方式 | 按图像身份分组，开发比例 %.1f，随机种子 %d |"
             % (DEFAULT_DEV_FRACTION, DEFAULT_SEED))
    L.append("| 判定线 | 信号 ≥%d 种；校准提升 ≥%.0f%%；预警准确率 ≥%.0f%% |"
             % (SIGNAL_BAR, 100 * ECE_BAR, 100 * ACCURACY_BAR))
    L.append("")

    # ---- 2.1
    L.append("## 指标 2.1　不确定性信号协同建模")
    L.append("")
    L.append("判定：单一数据集上实际产生非恒定取值的信号数 ≥ %d 种。" % SIGNAL_BAR)
    L.append("")
    L.append("| 模型 | 数据集 | 测试样本 | 实际有读数的信号 | 读不到的信号（原因） | 判定 |")
    L.append("|---|---|---:|---:|---|---|")
    for key, r in results:
        if r is None:
            continue
        dead = "、".join("%s（%s）" % (k, v) for k, v in sorted(r["dead_signals"].items())) or "无"
        L.append("| %s | %s | %d | **%d** | %s | %s |"
                 % (key[0], key[1], r["n_test"], r["live_signals"], dead,
                    "达标" if r["pass_signals"] else "未达"))
    L.append("")

    # ---- 2.2
    L.append("## 指标 2.2　不确定性来源分析（校准性能提升）")
    L.append("")
    L.append("判定：相对提升 = （基线 ECE − 本方法 ECE）/ 基线 ECE × 100%% ≥ %.0f%%。" % (100 * ECE_BAR))
    L.append("")
    L.append("基线取开发集上区分能力最强的单信号，并**用与本方法相同的校准器在开发集上单独校准**——"
             "否则比的是“有没有校准”，不是“多信号协同有没有用”。")
    L.append("")
    L.append("| 模型 | 数据集 | 基线信号 | 基线 ECE（同等校准） | 本方法 ECE | 相对提升 | AUROC | 判定 |")
    L.append("|---|---|---|---:|---:|---:|---:|---|")
    for key, r in results:
        if r is None:
            continue
        L.append("| %s | %s | %s | %.4f | %.4f | **%.1f%%** | %.4f | %s |"
                 % (key[0], key[1], r["baseline_signal"], r["ece_calibrated"],
                    r["ece_fusion"], r["gain_vs_calibrated"], r["auroc"],
                    "达标" if r["pass_ece"] else "未达"))
    L.append("")
    L.append("分箱敏感性（相对提升）：")
    L.append("")
    L.append("| 模型 | 数据集 | " + " | ".join("B=%d" % n for n in BINS) + " |")
    L.append("|---|---|" + "---:|" * len(BINS))
    for key, r in results:
        if r is None:
            continue
        L.append("| %s | %s | " % (key[0], key[1])
                 + " | ".join("%.1f%%" % r["gain_by_bins"][n] for n in BINS) + " |")
    L.append("")

    # ---- 2.3
    L.append("## 指标 2.3　多模态生成内容风险评估（高风险内容提前预警准确率）")
    L.append("")
    L.append("判定：预警准确率 = （TP + TN）/ 总样本数 ≥ %.0f%%。阈值在开发集上按准确率最大选取，"
             "测试集只评一次。" % (100 * ACCURACY_BAR))
    L.append("")
    L.append("| 模型 | 数据集 | 阈值 | 测试样本 | 准确率 | 从不预警基线 | TP/FP/TN/FN | AUROC | 判定 |")
    L.append("|---|---|---:|---:|---:|---:|---|---:|---|")
    for key, r in results:
        if r is None:
            continue
        c = r["confusion"]
        L.append("| %s | %s | %.3f | %d | **%s** | %s | %d/%d/%d/%d | %.4f | %s |"
                 % (key[0], key[1], r["threshold"], r["n_test"], fmt_pct(r["accuracy"]),
                    fmt_pct(r["never_warn"]), c["TP"], c["FP"], c["TN"], c["FN"],
                    r["auroc"], "达标" if r["pass_accuracy"] else "未达"))
    L.append("")

    # ---- summary
    L.append("## 汇总")
    L.append("")
    ok = [k for k, r in results if r is not None]
    L.append("| 指标 | 达标格数 | 说明 |")
    L.append("|---|---:|---|")
    L.append("| 2.1 信号协同建模 | %d / %d | 全部格的实际信号数均 ≥ %d |"
             % (sum(1 for _, r in results if r and r["pass_signals"]), len(ok), SIGNAL_BAR))
    L.append("| 2.2 校准性能提升 | %d / %d | 相对同等校准单信号 |"
             % (sum(1 for _, r in results if r and r["pass_ece"]), len(ok)))
    L.append("| 2.3 预警准确率 | %d / %d | 结题线 %.0f%% |"
             % (sum(1 for _, r in results if r and r["pass_accuracy"]), len(ok),
                100 * ACCURACY_BAR))
    L.append("")
    L.append("> 本报告只使用已采集的证据缓存，不含模型推理。同一份缓存与同一份清单"
             "重复运行得到完全相同的报告。")
    return "\n".join(L)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="证据清单 JSON")
    parser.add_argument("--out", default="acceptance_report.md", help="报告输出路径")
    parser.add_argument("--missing-policy", default="mean", choices=("zero", "mean"),
                        help="不可用信号进入风险打分的方式（默认 mean）")
    parser.add_argument("--datasets", default="POPE,TextVQA,CHAIR",
                        help="纳入报告的指标数据集，逗号分隔")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    payload["_manifest"] = args.manifest
    root = Path(payload.get("root", "")).expanduser()
    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}

    results = []
    for entry in payload["cells"]:
        if wanted and entry["dataset"] not in wanted:
            continue
        path = Path(entry["path"]).expanduser()
        full = path if (path.is_absolute() or not str(root)) else root / path
        key = (entry["model"], entry["dataset"])
        if not full.is_file():
            print("[skip] %s / %s：证据文件不存在 %s" % (key[0], key[1], full))
            results.append((key, None))
            continue
        try:
            stats = analyse(*load_cell(full), missing_policy=args.missing_policy)
        except ValueError as exc:
            print("[skip] %s / %s：%s" % (key[0], key[1], exc))
            results.append((key, None))
            continue
        if stats is None:
            print("[skip] %s / %s：单类标签，无法评分" % key)
            results.append((key, None))
            continue
        print("[ok]   %s / %s" % key)
        results.append((key, stats))

    report = render_report(payload, root, results, args.missing_policy)
    Path(args.out).write_text(report, encoding="utf-8")
    print("\n报告已写出：%s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
