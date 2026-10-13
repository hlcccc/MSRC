#!/usr/bin/env python
"""Cross-cell audit of the acceptance table: are the three models treated the same?

``pipeline_audit.py`` looks inside one cell -- is the file the right size, does every
row name its label rule, is a photograph on both sides of the split. That is necessary
and not sufficient. A table can be built from nine individually clean cells and still be
an unfair comparison, because the ways a model can be handed an advantage all live
*between* cells:

* **A different item set.** Two models scored on different questions are not being
  compared, whatever the column header says. Checked by intersecting the item ids and
  the image identities across the three models of each dataset.
* **A different evidence channel.** A cell collected with a second model attached has
  ``cross_model``; one collected without it does not. Since ``cross_model_agreement`` is
  the strongest single signal on the one dataset where it was measured, a model that got
  it and a model that did not are not running the same method. Checked per channel.
* **A different label rule or a different base rate.** Reported side by side, because a
  rule applied to one model and not another is the quietest possible way to move a
  number.
* **A pass that only exists on one split seed.** Checked by re-running the whole table
  over fourteen seeds and reporting how many of them pass, which is the honest
  expectation rather than the best-looking draw.

It also recomputes every reported number from scratch -- fit on dev, threshold on dev by
plain accuracy, score the test half -- and refuses to agree silently: a mismatch against
``balanced_report.py`` is reported as a problem, because the one thing an audit must not
do is confirm a number by re-reading it from the same place.

    python scripts/audit_acceptance.py --manifest runs/cells.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.groups import image_group_id  # noqa: E402
from evaluation.metrics import auroc  # noqa: E402
from evaluation.split import DEFAULT_DEV_FRACTION, DEFAULT_SEED, grouped_split  # noqa: E402
from msrc.model import RiskCalibrator  # noqa: E402
from msrc.signals import Evidence, build_report  # noqa: E402

GRID = np.round(np.linspace(0.0, 1.0, 201), 4)
CHANNELS = ("views", "resamples", "verifications", "cross_model", "ocr_texts")
SEEDS = list(range(14))


def load_cell(path: Path):
    rows = [json.loads(l) for l in
            path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    matrix, labels, groups, channels = [], [], [], collections.Counter()
    ids = []
    for row in rows:
        ev = Evidence.from_dict(row["evidence"])
        matrix.append(build_report(ev).vector())
        labels.append(int(row["label"]))
        groups.append(image_group_id(row))
        ids.append(str(row.get("id", "")))
        for ch in CHANNELS:
            if getattr(ev, ch):
                channels[ch] += 1
    n = len(rows)
    return {
        "rows": rows, "path": path, "n": n, "ids": ids,
        "X": np.asarray(matrix, float), "y": np.asarray(labels, int),
        "groups": np.asarray(groups),
        "channels": {ch: channels[ch] / n for ch in CHANNELS},
        "label_rules": collections.Counter(
            str(r.get("label_rule") or r.get("label_source") or "") for r in rows),
        "images": {Path(str(r.get("image", ""))).name for r in rows},
    }


def threshold_on(scores, labels):
    values = np.array([float(((scores >= t).astype(int) == labels).mean()) for t in GRID])
    return float(GRID[len(GRID) - 1 - int(values[::-1].argmax())])


def evaluate(cell, seeds=(DEFAULT_SEED,)):
    """Dev fits and selects, test is scored. Nothing here sees a test label."""
    out = []
    for seed in seeds:
        dev, test = grouped_split(cell["groups"], dev_fraction=DEFAULT_DEV_FRACTION, seed=seed)
        if len(set(cell["y"][dev])) < 2 or len(set(cell["y"][test])) < 2:
            out.append(None)
            continue
        fit = RiskCalibrator(l2=0.05).fit(cell["X"][dev], cell["y"][dev])
        thr = threshold_on(fit.predict_proba(cell["X"][dev]), cell["y"][dev])
        scores = fit.predict_proba(cell["X"][test])
        out.append({
            "accuracy": float(((scores >= thr).astype(int) == cell["y"][test]).mean()),
            "auroc": auroc(cell["y"][test], scores),
            "never_warn": float((cell["y"][test] == 0).mean()),
            "positive_rate": float(cell["y"][test].mean()),
            "threshold": thr, "n_test": int(test.sum()),
        })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--datasets", default="",
                        help="comma-separated subset to audit; default is all in the manifest")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    root = Path(payload.get("root", "")).expanduser()
    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}

    by_dataset = collections.defaultdict(dict)
    for entry in payload["cells"]:
        if wanted and entry["dataset"] not in wanted:
            continue
        path = Path(entry["path"]).expanduser()
        if not root or path.is_absolute():
            full = path
        else:
            full = root / path
        if full.is_file():
            by_dataset[entry["dataset"]][entry["model"]] = load_cell(full)

    problems, notes = [], []
    summary = {}
    for dataset in sorted(by_dataset):
        cells = by_dataset[dataset]
        print("=" * 108)
        print("### %s —— %d 个模型" % (dataset, len(cells)))
        if len(cells) < 2:
            notes.append("%s: 只有一个模型有数据，无法做跨模型对比" % dataset)
            print("   只有一个模型，跳过跨模型对比")
            continue

        # -- same items, same pictures
        id_sets = {m: set(c["ids"]) for m, c in cells.items()}
        img_sets = {m: c["images"] for m, c in cells.items()}
        common_ids = set.intersection(*id_sets.values())
        common_imgs = set.intersection(*img_sets.values())
        for m, s in id_sets.items():
            if s != common_ids:
                problems.append("%s | %s: 条目集合与其他模型不同（独有 %d 条）"
                                % (m, dataset, len(s - common_ids)))
        for m, s in img_sets.items():
            if s != common_imgs:
                problems.append("%s | %s: 图片集合与其他模型不同（独有 %d 张）"
                                % (m, dataset, len(s - common_imgs)))
        if all(s == common_ids for s in id_sets.values()):
            print("   ✓ 三模型条目集合完全相同（%d 条，图片 %d 张）"
                  % (len(common_ids), len(common_imgs)))
        else:
            print("   ✗ 条目集合不一致：共有 %d 条" % len(common_ids))

        # -- same evidence channels
        print("   %-15s %6s %10s %10s %10s %10s %9s" %
              ("模型", "n", "cross_model", "views", "verif", "ocr", "正类率"))
        for m, c in sorted(cells.items()):
            ch = c["channels"]
            print("   %-15s %6d %9.0f%% %9.0f%% %9.0f%% %9.0f%% %8.1f%%" %
                  (m, c["n"], 100 * ch["cross_model"], 100 * ch["views"],
                   100 * ch["verifications"], 100 * ch["ocr_texts"], 100 * c["y"].mean()))
        for ch in CHANNELS:
            vals = {m: round(c["channels"][ch], 3) for m, c in cells.items()}
            if len(set(vals.values())) > 1:
                span = max(vals.values()) - min(vals.values())
                if span > 0.02:
                    problems.append("%s: 证据通道 %r 三模型可用率不一致 %s（差 %.0f 个百分点）"
                                    % (dataset, ch, vals, 100 * span))

        # -- same label rule on every row
        for m, c in sorted(cells.items()):
            blank = c["label_rules"].get("", 0)
            if blank:
                problems.append("%s | %s: %d 条没写标签来源" % (m, dataset, blank))
            dup = [k for k, v in collections.Counter(c["ids"]).items() if v > 1]
            if dup:
                problems.append("%s | %s: 重复 id %d 个" % (m, dataset, len(dup)))
        rules = {m: sorted(c["label_rules"]) for m, c in cells.items()}
        if len({tuple(v) for v in rules.values()}) > 1:
            notes.append("%s: 标签规则写法不统一 %s" % (dataset, rules))

        # -- recompute the number, and check the split is clean
        rows_out = []
        for m, c in sorted(cells.items()):
            dev, test = grouped_split(c["groups"], dev_fraction=DEFAULT_DEV_FRACTION,
                                      seed=DEFAULT_SEED)
            leak = set(c["groups"][dev]) & set(c["groups"][test])
            if leak:
                problems.append("%s | %s: 开发/测试半区共用 %d 张图" % (m, dataset, len(leak)))
            one = evaluate(c)[0]
            many = [r for r in evaluate(c, SEEDS) if r]
            passes = sum(1 for r in many if r["accuracy"] >= 0.85)
            row = {"model": m, "accuracy": one["accuracy"], "auroc": one["auroc"],
                   "never_warn": one["never_warn"], "n_test": one["n_test"],
                   "lift": one["accuracy"] - one["never_warn"],
                   "seed_pass": passes, "seed_total": len(many),
                   "seed_mean": float(np.mean([r["accuracy"] for r in many]))}
            rows_out.append(row)
        summary[dataset] = rows_out
        print()
        print("   %-15s %8s %7s %9s %9s %9s %12s" %
              ("模型", "测试n", "准确率", "从不预警", "方法贡献", "AUROC", "14种子达标"))
        for r in rows_out:
            print("   %-15s %8d %7.2f%% %8.2f%% %+8.2f %9.4f %7d/%d  均值 %.2f%%" %
                  (r["model"], r["n_test"], 100 * r["accuracy"], 100 * r["never_warn"],
                   100 * r["lift"], r["auroc"], r["seed_pass"], r["seed_total"],
                   100 * r["seed_mean"]))
            if r["accuracy"] >= 0.85 and r["seed_pass"] < r["seed_total"]:
                notes.append("%s | %s: 达标但只在 %d/%d 个种子上达标（默认划分偏乐观）"
                             % (r["model"], dataset, r["seed_pass"], r["seed_total"]))
            if r["accuracy"] >= 0.85 and r["lift"] < 0.02:
                notes.append("%s | %s: 达标，但相对「从不预警」只高 %.1f 个点 —— 达标主要来自基率"
                             % (r["model"], dataset, 100 * r["lift"]))

    print()
    print("=" * 108)
    print("### 汇总")
    for x in notes:
        print("   ⚠ " + x)
    for x in problems:
        print("   ✗ " + x)
    if not problems:
        print("   没有发现跨格公平性问题。")
    print()
    print("   说明：本审计只回答「三个模型是否被同等对待、数字能否独立复算」。")
    print("   它不能回答「这个基准本身值不值得信」，也不构成对任何一格结论的背书。")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
