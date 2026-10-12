#!/usr/bin/env python
"""Attach labels to evidence that was collected where the answers were not allowed.

A collection run can be split across two machines: one is allowed to run the model but
not to see the benchmark's answers, the other holds the answers and never touches a
GPU. The collector on the first machine generates the answer and every signal, writes
``label_from_response`` in place of a label, and stops there. This is the second half:
it joins the two files by item id and applies the benchmark's own rule.

The join is checked rather than trusted. Items are matched on ``id`` alone, because the
two sides legitimately disagree about the image path -- the collecting machine rewrote
it to its own layout -- and the question text is compared as well, so a file that has
drifted out of step with its answers is reported instead of silently mislabelled.

    python scripts/apply_labels.py \\
        --evidence cell_LLaVA-1.5-13B_okvqa.jsonl \\
        --gold okvqa_input.jsonl \\
        --out cell_LLaVA-1.5-13B_okvqa_labelled.jsonl \\
        --rule vqa

``--rule vqa`` is the VQA agreement rule: matching answers over three, half credit to
pass. ``--rule yes_no`` parses a yes/no response and compares it with the reference.
Both are the same functions the collector uses, imported rather than re-implemented, so
a locally labelled cell and a remotely labelled one cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_evidence import vqa_label, yes_no_label  # noqa: E402


def load_jsonl(path: Path) -> list:
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} holds no records")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--evidence", required=True, help="rows carrying answers, no labels")
    parser.add_argument("--gold", required=True, help="rows carrying gold_answers / label")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rule", choices=("vqa", "yes_no"), default="vqa")
    args = parser.parse_args(argv)

    evidence = load_jsonl(Path(args.evidence).expanduser())
    gold_rows = load_jsonl(Path(args.gold).expanduser())

    gold = {}
    for row in gold_rows:
        key = row.get("id")
        if key is None:
            raise SystemExit(f"--gold row without an 'id': {str(row)[:120]}")
        if key in gold:
            raise SystemExit(f"--gold has two rows with id {key!r}; the join is ambiguous")
        gold[key] = row

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    labelled = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in evidence:
            key = row.get("id")
            reference = gold.get(key)
            if reference is None:
                counts["no matching gold row"] += 1
                continue
            if str(reference.get("question", "")) != str(row.get("question", "")):
                counts["question differs from the gold file"] += 1
                continue
            answer = str(row.get("answer", "") or "")
            if not answer.strip():
                counts["empty answer"] += 1
                continue

            if args.rule == "vqa":
                answers = reference.get("gold_answers") or reference.get("answers") or []
                if not answers:
                    counts["gold row has no accepted answers"] += 1
                    continue
                row["label"] = vqa_label(answer, answers)
                row["label_rule"] = "vqa_agreement_div3"
            else:
                answers = reference.get("gold_answers") or reference.get("answers") or []
                if not answers:
                    counts["gold row has no reference answer"] += 1
                    continue
                label, parsed = yes_no_label(
                    answer, answers[0], parser=reference.get("answer_parser", "legacy"))
                row["label"] = label
                row["label_parsed"] = parsed
                row["label_rule"] = "yes_no_" + str(reference.get("answer_parser", "legacy"))
                if not parsed:
                    counts["response said neither yes nor no (labelled wrong)"] += 1

            row.pop("label_from_response", None)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            labelled += 1
            counts["positive" if row["label"] else "negative"] += 1

    print(f"[labels] {labelled}/{len(evidence)} rows labelled by the {args.rule} rule -> {out_path}")
    if labelled:
        positives = counts["positive"]
        print(f"[labels] positive rate {100 * positives / labelled:.1f}% "
              f"({positives} risky, {counts['negative']} safe)")
    for reason, count in counts.most_common():
        if reason in ("positive", "negative"):
            continue
        print(f"[labels] dropped {count}: {reason}")
    if labelled < len(evidence):
        print("[labels] the dropped rows are absent from the output rather than guessed at; "
              "report how many were dropped beside the accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
