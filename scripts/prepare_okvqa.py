#!/usr/bin/env python
"""Build the MSRC input for OKVQA from the official v1.1 release.

OKVQA is the one benchmark in reach that has the shape this method is good at: a short
free-form answer to a question about a real photograph, with ten human answers to
compare it against. That is TextVQA's shape, and TextVQA is where the method reaches
88.0% -- against 77.2% on CHAIR and 67.8% on HallusionBench, both of which cap out well
below the 85% line. Adding it is how the acceptance claim gets a third dataset per
model without dropping the two datasets where the method does badly.

Where the labels come from
--------------------------

The ten answers per question are the official annotations, and they are used exactly
the way :mod:`scripts.collect_evidence` uses them: the VQA agreement rule, matches over
three. Nothing here re-derives or filters them.

``--public-out`` writes the same items **without any answers**, for a machine that is
allowed to run the model but not to see the labels. That file carries
``label_from_response`` so the collector generates the answer and leaves the label
empty; :mod:`scripts.apply_labels` fills it in afterwards, on the machine that holds
the annotations. This is the same split of duties the CHAIR cells already use.

Image identity is carried as ``image_id`` (the COCO image id), which is what
:func:`evaluation.groups.image_group_id` groups on, so two questions about one
photograph can never land on opposite sides of the split.

    python scripts/prepare_okvqa.py \\
        --questions OpenEnded_mscoco_val2014_questions.json \\
        --annotations mscoco_val2014_annotations.json \\
        --images-dir /mnt/data/HLC/.../images/okvqa \\
        --images-dir /mnt/data/HLC/chair_val2014_images \\
        --out okvqa_input.jsonl --public-out okvqa_public.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: The official file name for a COCO 2014 image, which is also what the mirror uses.
IMAGE_TEMPLATE = "COCO_val2014_{:012d}.jpg"


def coco_name(image_id: int) -> str:
    return IMAGE_TEMPLATE.format(int(image_id))


def load_annotations(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {int(a["question_id"]): a for a in payload["annotations"]}


def answers_of(annotation) -> list:
    """The ten official answers, in the release's own order."""
    return [str(a["answer"]) for a in annotation.get("answers", [])]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images-dir", action="append", required=True,
                        help="a directory of COCO_val2014_<12 digits>.jpg; repeatable, "
                             "searched in order")
    parser.add_argument("--out", required=True, help="labelled input for a local run")
    parser.add_argument("--public-out", default="",
                        help="answer-free input for a machine that must not see labels")
    parser.add_argument("--limit", type=int, default=0, help="0 keeps every question")
    args = parser.parse_args(argv)

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8-sig"))["questions"]
    annotations = load_annotations(Path(args.annotations))
    directories = [Path(d).expanduser() for d in args.images_dir]
    for directory in directories:
        if not directory.is_dir():
            raise SystemExit(f"--images-dir is not a directory: {directory}")

    if args.limit:
        # Sorted by question id so the subset is a documented, reproducible slice
        # rather than whatever order the file happened to be in.
        questions = sorted(questions, key=lambda q: int(q["question_id"]))[: args.limit]

    labelled, public, skipped = [], [], Counter()
    for question in questions:
        question_id = int(question["question_id"])
        image_id = int(question["image_id"])
        name = coco_name(image_id)
        image = next((d / name for d in directories if (d / name).is_file()), None)
        if image is None:
            skipped["image missing"] += 1
            continue
        annotation = annotations.get(question_id)
        if annotation is None:
            skipped["annotation missing"] += 1
            continue
        gold = answers_of(annotation)
        if not gold:
            skipped["no answers"] += 1
            continue

        common = {
            "id": f"okvqa:validation:{question_id}",
            "image": str(image),
            "image_id": image_id,
            "question": str(question["question"]),
            "answer_format": "open",
            "dataset": "okvqa",
            "split": "validation",
        }
        labelled.append({**common, "gold_answers": gold})
        if args.public_out:
            # No gold_answers and no label: the collector generates the answer and
            # says so, and the label is applied later where the answers live.
            public.append({**common, "label_from_response": "okvqa"})

    for path, rows in ((Path(args.out), labelled),
                       (Path(args.public_out), public) if args.public_out else (None, [])):
        if path is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[okvqa] {len(labelled)} items with the official ten answers -> {args.out}")
    if args.public_out:
        print(f"[okvqa] {len(public)} answer-free items -> {args.public_out}")
    for reason, count in skipped.most_common():
        print(f"[okvqa] skipped {count}: {reason}")
    if labelled:
        sizes = Counter(len(r["gold_answers"]) for r in labelled)
        print(f"[okvqa] answers per item: {dict(sorted(sizes.items()))}")
        print(f"[okvqa] distinct images: {len({r['image_id'] for r in labelled})}"
              f"  ({len(labelled)} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
