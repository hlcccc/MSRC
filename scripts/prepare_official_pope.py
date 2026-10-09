#!/usr/bin/env python
"""Prepare all 9000 original COCO POPE questions, without GPU inference."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("output already exists; select a new filename")
    rows = []
    for split in ("random", "popular", "adversarial"):
        refs = [json.loads(line) for line in (args.references / f"coco_pope_{split}.json").read_text().splitlines()]
        if len(refs) != 3000 or len({row["image"] for row in refs}) != 500:
            raise SystemExit("reference does not match original 500-image / 3000-question setting")
        for ref in refs:
            image = args.image_dir / f"pope_{split}_{ref['image']}"
            if args.verify_images and not image.is_file():
                raise SystemExit(f"missing image: {image}")
            rows.append({"id": f"pope_official:{split}:{ref['question_id']}",
                         "source_image_id": Path(ref["image"]).stem,
                         "official_question_id": ref["question_id"],
                         "question": ref["text"], "image": str(image),
                         "dataset": "POPE", "split": split,
                         "gold_answers": [ref["label"]], "answer_format": "yes_no",
                         "answer_parser": "pope_official"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"prepared {len(rows)} questions; inference has NOT been run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
