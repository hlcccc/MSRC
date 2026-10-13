#!/usr/bin/env python
"""Did this cell come from one generation budget, or several accidentally?

The check exists because of a real defect it caught. A cell's answers are generated
with ``--max-new-tokens``, and ``collect_evidence.py`` resumes by item key: a run that
was started with the default budget and later continued with an explicit larger one
keeps the early answers and appends the rest. Every count stays right -- 1,200 rows is
1,200 rows -- and nothing downstream complains, but the cell is now two experiments.
It matters most on a captioning benchmark: a longer caption names more objects, and on
CHAIR naming more objects is exactly what makes a caption count as hallucinated. The
mixed cell therefore has a *lower* positive rate than either budget alone would give,
which quietly raises the never-warn baseline and makes the cell look easier.

The tell is that an answer cut off by the budget has exactly ``max_new_tokens`` tokens,
so the token counts pile up on one or two integers instead of trailing off. That is
what this measures -- using the model's own tokenizer, because a token count obtained
any other way is not the number the generator was counting.

    python scripts/audit_generation_budget.py \
        --cell "Qwen2.5-VL-7B|CHAIR=cell.jsonl" \
        --tokenizer /path/to/Qwen2.5-VL-7B-Instruct

A cell is reported as mixed when a clamp holds a material share of the rows and there
is more than one such clamp. Exit status is non-zero when any cell is mixed.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: A token count this close to a round budget is taken to be that budget.
TOLERANCE = 2

#: Budgets a cell might plausibly have been generated with.
CANDIDATE_BUDGETS = (32, 48, 64, 96, 128, 256, 512)

#: Share of rows that must sit on a clamp before it counts as a budget rather than luck.
MATERIAL_SHARE = 0.05


def clamp_counts(lengths, budgets=CANDIDATE_BUDGETS, tolerance=TOLERANCE):
    """How many answers sit on each candidate budget."""
    hits = collections.Counter()
    for n in lengths:
        for budget in budgets:
            if abs(n - budget) <= tolerance:
                hits[budget] += 1
                break
    return hits


def classify(lengths, budgets=CANDIDATE_BUDGETS, tolerance=TOLERANCE,
             material=MATERIAL_SHARE):
    """``(verdict, clamps)`` where verdict is 'single', 'mixed' or 'none'."""
    if not lengths:
        return "none", {}
    hits = clamp_counts(lengths, budgets, tolerance)
    n = len(lengths)
    material_hits = {b: c for b, c in hits.items() if c / n >= material}
    if len(material_hits) > 1:
        return "mixed", material_hits
    if len(material_hits) == 1:
        return "single", material_hits
    return "none", {}


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def parse_cell(spec: str):
    head, raw = spec.split("=", 1)
    model, dataset = head.split("|", 1)
    return model, dataset, Path(raw).expanduser()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", action="append", required=True, metavar="MODEL|DATASET=PATH")
    parser.add_argument("--tokenizer", required=True,
                        help="the model's own tokenizer directory; the count has to be "
                             "the one the generator was counting")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    # Read the files before loading a tokenizer: a missing or empty cell is a finding in
    # its own right, and reporting it should not depend on a model directory existing.
    loaded = []
    for spec in args.cell:
        model, dataset, path = parse_cell(spec)
        if not path.is_file():
            loaded.append((model, dataset, path, None))
            continue
        rows = load_jsonl(path)
        answers = [str(r.get("answer", "")) for r in rows if str(r.get("answer", "")).strip()]
        loaded.append((model, dataset, path, answers))

    if any(a for _, _, _, a in loaded):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    else:
        tokenizer = None

    print("=" * 104)
    print("生成长度上限一致性：这一格到底是一次实验，还是几次不同上限的运行拼起来的")
    print("=" * 104)
    mixed = 0
    for model, dataset, path, answers in loaded:
        print()
        print(f"### {model} | {dataset}")
        if answers is None:
            print(f"   ✗ 文件不存在 {path}")
            mixed += 1
            continue
        if not answers:
            print("   ✗ 没有非空答案")
            mixed += 1
            continue
        lengths = [len(tokenizer(a, add_special_tokens=False)["input_ids"]) for a in answers]
        verdict, clamps = classify(lengths)
        top = collections.Counter(lengths).most_common(5)
        print("   n=%d  最长 %d tokens  最常见的长度 %s"
              % (len(lengths), max(lengths), top))
        for budget, count in sorted(clamps.items()):
            print("   卡在 %d tokens：%d 条（%.1f%%）" % (budget, count, 100 * count / len(lengths)))
        if verdict == "mixed":
            print("   ✗ **混了两批上限** —— 这条格子的答案不是一次实验产生的，"
                  "在不同长度的答案上标签含义不同（描述越长越容易点到不存在的物体）。"
                  "要先把这一格按同一个上限重采，再谈它的指标。")
            mixed += 1
        elif verdict == "single":
            print("   ✓ 单一上限，格子内部一致")
        else:
            print("   ✓ 没有明显卡在预算上的堆积（短答任务通常如此）")
    print()
    print("=" * 104)
    print("发现 %d 处问题。" % mixed if mixed else "所有格子内部一致。")
    return 1 if mixed else 0


if __name__ == "__main__":
    raise SystemExit(main())
