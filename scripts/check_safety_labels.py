#!/usr/bin/env python
"""Measure the refusal label rule against hand labels.

:mod:`msrc.judge` decides the safety label with keyword matching, and says in its
own docstring that the disagreement rate against a hand-labelled sample should be
measured and reported with any result. This is that measurement. Without it a
safety number carries an unknown error floor: the labels it was fitted on are a
judgement, and the size of the judgement is the thing a reader cannot see.

Two modes
---------

**Make a labelling sheet** -- sample responses out of a collected evidence file and
blank the human column:

    python scripts/check_safety_labels.py --make-template evidence.jsonl \\
        --out to_label.jsonl --sample 100

Each line comes back with ``human_refused`` set to ``null``. Fill it with ``true``
or ``false`` by reading the response, then run the check below.

**Check** -- compare the rule against the filled sheet:

    python scripts/check_safety_labels.py --data to_label.jsonl

It prints the agreement, the confusion counts, Cohen's kappa and every disagreement
with its response text, because the disagreements are the point: a model that says
"I can't help with that, but here is how..." is the failure mode keyword matching
is known to have, and reading those cases is how you find out whether it matters
for your data.

Rows with ``human_refused`` left null are reported as unlabelled and excluded, so a
half-finished sheet can be checked without pretending it is complete.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msrc.judge import classify_refusal  # noqa: E402


def load_jsonl(path: Path) -> list:
    source = Path(path).expanduser()
    if not source.is_file():
        raise SystemExit(
            f"file not found: {path}\n"
            "  Expected the JSONL written by scripts/collect_evidence.py, or a\n"
            "  labelling sheet from --make-template."
        )
    rows = []
    for lineno, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno} is not valid JSON: {exc.msg}") from exc
    if not rows:
        raise SystemExit(f"{path} holds no records")
    return rows


def cohen_kappa(pairs) -> float:
    """Agreement corrected for what chance would produce.

    Raw agreement flatters a rule on an unbalanced sample: if 90% of responses are
    refusals, always answering "refused" scores 0.90 while having learned nothing.
    """
    n = len(pairs)
    if n == 0:
        return float("nan")
    agree = sum(1 for a, b in pairs if a == b) / n
    human_true = sum(1 for a, _ in pairs if a) / n
    rule_true = sum(1 for _, b in pairs if b) / n
    expected = human_true * rule_true + (1 - human_true) * (1 - rule_true)
    if expected >= 1.0:
        return 1.0 if agree >= 1.0 else 0.0
    return (agree - expected) / (1.0 - expected)


def make_template(rows, sample: int, seed: int) -> list:
    """Blank labelling sheets from collected evidence, keeping the response text."""
    pool = [r for r in rows if str(r.get("answer", "")).strip()]
    if not pool:
        raise SystemExit("no records carry an 'answer'; nothing to label")
    if sample and sample < len(pool):
        pool = random.Random(seed).sample(pool, sample)
    return [
        {
            "id": r.get("id", ""),
            "category": r.get("category", ""),
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "rule_refused": classify_refusal(r.get("answer", "")).refused,
            "human_refused": None,
            "note": "",
        }
        for r in pool
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", help="labelling sheet to check")
    parser.add_argument("--make-template", metavar="EVIDENCE",
                        help="write a blank labelling sheet from this evidence file")
    parser.add_argument("--out", help="where to write the sheet (with --make-template)")
    parser.add_argument("--sample", type=int, default=100, help="rows to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--min-agreement", type=float, default=None,
                        help="exit non-zero below this raw agreement")
    parser.add_argument("--max-shown", type=int, default=10,
                        help="how many disagreements to print")
    args = parser.parse_args(argv)

    if args.make_template:
        if not args.out:
            raise SystemExit("--make-template needs --out")
        rows = make_template(load_jsonl(Path(args.make_template)), args.sample, args.seed)
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        refused = sum(1 for r in rows if r["rule_refused"])
        print(f"wrote {len(rows)} rows to {out}")
        print(f"  rule already calls {refused} of them refusals ({refused/len(rows):.1%})")
        print("  fill 'human_refused' with true or false, then re-run with --data")
        return 0

    if not args.data:
        raise SystemExit("pass --data to check a sheet, or --make-template to make one")

    rows = load_jsonl(Path(args.data))
    labelled, unlabelled = [], 0
    for row in rows:
        human = row.get("human_refused")
        if human is None:
            unlabelled += 1
            continue
        verdict = classify_refusal(row.get("answer", ""))
        labelled.append((bool(human), verdict.refused, verdict.certain, row))

    if not labelled:
        raise SystemExit(
            f"none of the {len(rows)} rows have 'human_refused' set. "
            "Label some first -- the rule cannot be checked against itself."
        )

    pairs = [(h, r) for h, r, _, _ in labelled]
    agreement = sum(1 for a, b in pairs if a == b) / len(pairs)
    kappa = cohen_kappa(pairs)
    tp = sum(1 for h, r, _, _ in labelled if h and r)
    tn = sum(1 for h, r, _, _ in labelled if not h and not r)
    fp = sum(1 for h, r, _, _ in labelled if not h and r)
    fn = sum(1 for h, r, _, _ in labelled if h and not r)
    disagree = [(h, r, c, row) for h, r, c, row in labelled if h != r]

    print("=" * 78)
    print("拒绝判定规则 vs 人工标注")
    print("=" * 78)
    print(f"  已标注            : {len(labelled)}   (未标注 {unlabelled} 行已排除)")
    print(f"  一致率            : {agreement:.3%}")
    print(f"  Cohen's kappa     : {kappa:.4f}")
    print(f"  人工认为拒绝      : {sum(1 for h, _ in pairs if h)}")
    print(f"  规则认为拒绝      : {sum(1 for _, r in pairs if r)}")
    print()
    print("  混淆（正类 = 拒绝）:")
    print(f"    人工拒绝 / 规则拒绝  TP = {tp}")
    print(f"    人工拒绝 / 规则放行  FN = {fn}   <- 有害回答被当成合规")
    print(f"    人工合规 / 规则拒绝  FP = {fp}   <- 正常回答被当成拒绝")
    print(f"    人工合规 / 规则放行  TN = {tn}")
    print()

    if disagree:
        print(f"  不一致 {len(disagree)} 条，前 {min(args.max_shown, len(disagree))} 条：")
        for human, rule, certain, row in disagree[: args.max_shown]:
            text = " ".join(str(row.get("answer", "")).split())[:150]
            print(f"    人工={'拒绝' if human else '合规'} 规则={'拒绝' if rule else '合规'}"
                  f" 规则确定={certain}  {row.get('id', '')}")
            print(f"      {text}")
        print()
    else:
        print("  没有不一致的条目。")
        print()

    # The rule's own confidence flag, against the cases where it was actually wrong.
    wrong_uncertain = sum(1 for _, _, c, _ in disagree if not c)
    if disagree:
        print(f"  不一致条目里规则自报不确定的: {wrong_uncertain}/{len(disagree)}")
        print("  （这个比例说明 certain=False 这个标记是否有用）")
        print()

    if args.min_agreement is not None and agreement < args.min_agreement:
        print(f"  一致率 {agreement:.3%} 低于要求 {args.min_agreement:.3%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
