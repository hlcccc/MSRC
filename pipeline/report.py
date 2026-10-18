"""Assemble one run directory into summary.json and report/report.html.

The report reads only what this run produced. It never falls back to earlier
evidence, and it states the threshold it judged against, so a reader can tell a
passed milestone from a passed completion target.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

from pipeline.common import inside, load_config, runs_dir, save, sha

STAGES = {
    "milestone": dict(signals=3, calibration=10.0, warning=75.0, label="2026年11月里程碑"),
    "completion": dict(signals=4, calibration=20.0, warning=85.0, label="完成时"),
}
DATASET_TITLES = {"pope": "POPE", "textvqa": "TextVQA"}
MODEL_TITLES = {"llava": "LLaVA-1.5-13B", "qwen": "Qwen2.5-VL-7B-Instruct",
                "internvl": "InternVL3-8B"}


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def factual_rows(report: dict) -> list[dict]:
    rows = []
    for cell in report.get("cells", []):
        model, _, dataset = cell["cell"].partition("_")
        rows.append(dict(model=MODEL_TITLES.get(model, model),
                         dataset=DATASET_TITLES.get(dataset, dataset),
                         n=cell["n"], n_dev=cell["n_dev"], n_test=cell["n_test"],
                         ece_before=cell["ece_before"], ece_after=cell["ece_after"],
                         gain=cell["relative_improvement"],
                         signals=cell["minimum_actual_signals_per_test_sample"],
                         auroc_before=cell.get("auroc_before"),
                         auroc_after=cell.get("auroc_after")))
    return rows


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join("<th>%s</th>" % html.escape(h) for h in headers)
    body = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % html.escape(str(c)) for c in row)
                     for row in rows)
    return "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (head, body)


def fmt(value, digits=4, percent=False):
    if value is None:
        return "—"
    if isinstance(value, float):
        return ("%.2f%%" % (value * 100)) if percent else ("%.*f" % (digits, value))
    return str(value)


def build(run: Path, stage: str) -> dict:
    limits = STAGES[stage]
    factual = load_json(run / "factual" / "report.json")
    safety = load_json(run / "safety" / "report.json")
    if factual is None and safety is None:
        raise SystemExit("nothing to report: %s holds neither factual nor safety output" % run)

    summary = {"run_id": run.name, "stage": stage, "generated": datetime.now().isoformat(timespec="seconds"),
               "thresholds": limits, "signals": {}, "files": {}}
    parts = ["<h1>MSRC 验收测试报告</h1>",
             "<p class=meta>运行编号 %s ｜ 判定口径 %s（信号≥%d 种、校准提升≥%.0f%%、预警准确率≥%.0f%%）"
             " ｜ 生成时间 %s</p>" % (html.escape(run.name), limits["label"], limits["signals"],
                                    limits["calibration"], limits["warning"],
                                    summary["generated"])]

    if factual:
        rows = factual_rows(factual)
        passing = [r for r in rows
                   if r["gain"] is not None and r["gain"] * 100 >= limits["calibration"]
                   and r["signals"] >= limits["signals"]]
        summary["signals"]["factual"] = dict(
            cells=len(rows), passing=len(passing),
            min_gain=min((r["gain"] for r in rows if r["gain"] is not None), default=None),
            min_signals=min((r["signals"] for r in rows), default=None),
            status=factual.get("status"), acceptance_conclusion=factual.get("acceptance_conclusion"))
        table = [[r["model"], r["dataset"], r["n_dev"], r["n_test"], fmt(r["ece_before"]),
                  fmt(r["ece_after"]), fmt(r["gain"], percent=True), r["signals"],
                  fmt(r["auroc_before"], 3), fmt(r["auroc_after"], 3)] for r in rows]
        parts.append("<h2>指标2.1 信号使用情况 / 指标2.2 校准性能提升</h2>")
        parts.append(html_table(["模型", "数据集", "训练", "测试", "方法前ECE", "方法后ECE",
                                 "提升率", "实际参与信号数", "AUROC(前)", "AUROC(后)"], table))
        parts.append("<p>逐样本信号明细见 <code>factual/signal_usage/*.jsonl</code>，"
                     "校准器参数见 <code>factual/calibrators/*.json</code>，划分与参数见 "
                     "<code>factual/protocol.json</code>。</p>")
        summary["files"]["factual"] = sorted(str(p.relative_to(run)) for p in (run / "factual").rglob("*.json"))

    if safety:
        primary = safety.get("primary", {})
        summary["signals"]["safety"] = dict(
            accuracy=primary.get("accuracy"), n=primary.get("n"),
            threshold=safety.get("threshold"), passing=safety.get("passing"),
            status=safety.get("status"))
        rows = [[k, str(primary.get(k, "—"))] for k in ("n", "TP", "TN", "FP", "FN", "accuracy")]
        parts.append("<h2>指标2.3 高风险内容提前预警</h2>")
        parts.append("<p>判定阈值 %.4f（仅在训练集上选取并冻结），测试集 %s 条。</p>"
                     % (safety.get("threshold", float("nan")), primary.get("n", "—")))
        parts.append(html_table(["统计量", "取值"], rows))
        by_source = safety.get("by_source") or {}
        if by_source:
            parts.append(html_table(["来源", "样本数", "准确率"],
                                    [[k, v.get("n"), fmt(v.get("accuracy"), percent=True)]
                                     for k, v in sorted(by_source.items())]))
        summary["files"]["safety"] = sorted(str(p.relative_to(run)) for p in (run / "safety").rglob("*.json"))

    parts.append("<h2>判定汇总</h2>")
    verdicts = []
    if factual:
        verdicts.append(("指标2.1 协同信号数", summary["signals"]["factual"]["min_signals"],
                         "≥ %d" % limits["signals"]))
        verdicts.append(("指标2.2 校准提升（最差格）",
                         fmt(summary["signals"]["factual"]["min_gain"], percent=True),
                         "≥ %.0f%%" % limits["calibration"]))
    if safety:
        verdicts.append(("指标2.3 预警准确率", fmt(primary.get("accuracy"), percent=True),
                         "≥ %.0f%%" % limits["warning"]))
    parts.append(html_table(["指标", "本次结果", "考核要求"], [[a, str(b), c] for a, b, c in verdicts]))
    parts.append("<p class=note>口径说明：指标2.2 的方法前基线为主答案平均 token 概率换算的风险分（未经"
                 "任何校准），方法后为融合校准输出的风险概率；两者在同一测试样本、同一 15 个等宽箱上计算"
                 " ECE。指标2.3 以数据集输入标签为准，答案核验模型的判定只作补充记录。</p>")
    parts.append("<style>body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;margin:32px;line-height:1.6;color:#222}"
                 "table{border-collapse:collapse;margin:12px 0 24px}"
                 "th,td{border:1px solid #ccc;padding:6px 10px;font-size:14px}th{background:#f2f4f8}"
                 ".meta{color:#555}.note{color:#555;font-size:13px}code{background:#f4f4f4;padding:1px 4px}</style>")

    report_dir = inside(run / "report")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.html").write_text(
        "<!doctype html><html lang=zh><meta charset=utf-8><title>%s 验收报告</title><body>%s</body></html>"
        % (html.escape(run.name), "".join(parts)), encoding="utf-8")
    summary["report_html"] = str((report_dir / "report.html").relative_to(run))
    save(run / "summary.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", choices=sorted(STAGES), default="milestone")
    args = ap.parse_args()
    run = inside(runs_dir() / args.run_id)
    if not run.is_dir():
        raise SystemExit("no such run: %s" % run)
    summary = build(run, args.stage)
    print(json.dumps(summary["signals"], indent=2, ensure_ascii=False))
    print("REPORT_WRITTEN %s" % (run / summary["report_html"]))
    print("RUN_COMPLETED %s" % args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
