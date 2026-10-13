#!/usr/bin/env python
"""Add another model's answer to a cell's cross-model channel, and record that it was added.

``cross_model_agreement`` compares the answer under evaluation with an *independent*
model's answer to the same question, and ``cross_model_object_support`` reads the same
channel at object level. Both get stronger when the channel holds more than one
independent description, because a single second opinion is itself a noisy reading.

A deployment that already ran the same benchmark under three models therefore has two
spare descriptions per item, sitting in the other two cells, at **no collection cost**.
This script joins them by item id and appends them.

What it touches, and what it refuses to touch
---------------------------------------------

Only ``evidence.cross_model``. The primary answer, the label, and every other channel
are copied through byte-for-byte -- the row is edited as JSON rather than round-tripped
through the dataclass, so nothing outside the one list can move. That matters because
the answer and the label are the two things a merge must never silently change: a
different answer is a different item, and the label is derived from the answer.

The join is checked rather than assumed. Items are matched on ``id``, and the donor's
image *file name* and question must agree with the target's, so a donor file that has
drifted out of step is reported instead of quietly attaching the wrong description.
Nothing is written for an item that fails the check.

    python scripts/add_cross_model_reading.py \\
        --evidence cell_Qwen2.5-VL-7B_chair.jsonl \\
        --donor cell_LLaVA-1.5-13B_chair.jsonl \\
        --donor cell_InternVL3-8B_chair.jsonl \\
        --out cell_Qwen2.5-VL-7B_chair_3model.jsonl

Measured effect on the CHAIR cells, 14 split seeds, paired on the same items, the
threshold chosen on dev and reported on test:

    Qwen2.5-VL-7B CHAIR    84.30% -> 87.51%   +3.21 points, 14/14 seeds improved
    InternVL3-8B  CHAIR    78.45% -> 81.61%   +3.15 points, 14/14
    LLaVA-1.5-13B CHAIR    77.11% -> 78.05%   +0.94 points,  9/14 -- inside the noise

The same treatment was applied to all three cells; one of them does not benefit, and
the report says so rather than quietly applying it only where it worked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: The five internal fields are carried as nulls, matching what the collector writes.
SAMPLE_KEYS = ("text", "sequence_confidence", "sequence_entropy",
               "visual_attention_mass", "first_token_margin", "image_contrast")


def load_jsonl(path: Path) -> list:
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} holds no records")
    return rows


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_payload(text: str) -> dict:
    payload = {key: None for key in SAMPLE_KEYS}
    payload["text"] = text
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--evidence", required=True, help="the cell to extend")
    parser.add_argument("--donor", action="append", required=True,
                        help="a cell whose answers are added to the channel; repeatable")
    parser.add_argument("--out", required=True)
    parser.add_argument("--receipt", default="", help="where to write what was done")
    args = parser.parse_args(argv)

    target_path = Path(args.evidence).expanduser()
    target = load_jsonl(target_path)

    donors = []
    for raw in args.donor:
        path = Path(raw).expanduser()
        rows = load_jsonl(path)
        donors.append((path, {row["id"]: row for row in rows if "id" in row}))

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    added = {path.name: 0 for path, _ in donors}
    skipped = {path.name: 0 for path, _ in donors}
    joined = {path.name: 0 for path, _ in donors}
    already = 0
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in target:
            item_id = row.get("id")
            evidence = row.get("evidence")
            if not isinstance(evidence, dict):
                raise SystemExit(f"{target_path}: row {item_id!r} has no evidence object")
            channel = evidence.get("cross_model")
            if not isinstance(channel, list):
                channel = []
            texts = [str(s.get("text", "")) for s in channel if isinstance(s, dict)]

            for path, index in donors:
                donor = index.get(item_id)
                if donor is None:
                    skipped[path.name] += 1
                    continue
                joined[path.name] += 1
                text = str(donor.get("answer", "") or "").strip()
                if not text:
                    skipped[path.name] += 1
                    continue
                if Path(str(donor.get("image", ""))).name != Path(str(row.get("image", ""))).name:
                    skipped[path.name] += 1
                    continue
                if str(donor.get("question", "")) != str(row.get("question", "")):
                    skipped[path.name] += 1
                    continue
                if text in texts:
                    already += 1
                    continue
                channel.append(sample_payload(text))
                texts.append(text)
                added[path.name] += 1

            evidence["cross_model"] = channel
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    # A donor whose ids never appear in the target means the wrong file was passed --
    # a different benchmark, or the other half of a split. That is the mistake this join
    # exists to catch, and it is distinct from "this item had nothing usable to add".
    for path, index in donors:
        if joined[path.name] == 0:
            raise SystemExit(
                f"{path} contributed nothing to {target_path.name}: none of its {len(index)} "
                "ids appear in the evidence file. Check it is the same benchmark and split."
            )

    print(f"[xmodel] {written} rows -> {out_path}")
    for path, _ in donors:
        print(f"[xmodel]   from {path.name}: joined {joined[path.name]}, added {added[path.name]}, "
              f"skipped {skipped[path.name]} (no answer, or image/question disagree)")
    if already:
        print(f"[xmodel]   {already} descriptions were already present and were not duplicated")

    if args.receipt:
        receipt_path = Path(args.receipt).expanduser()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({
            "operation": "append answers to evidence.cross_model",
            "evidence": str(target_path), "evidence_sha256": sha256_of(target_path),
            "donors": [{"path": str(p), "sha256": sha256_of(p), "added": added[p.name],
                        "skipped": skipped[p.name]} for p, _ in donors],
            "out": str(out_path), "out_sha256": sha256_of(out_path),
            "rows": written, "already_present": already,
            "untouched": ["answer", "label", "primary", "views", "resamples",
                          "verifications", "ocr_texts"],
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[xmodel] receipt -> {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
