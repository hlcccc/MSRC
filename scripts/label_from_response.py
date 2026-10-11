#!/usr/bin/env python3
"""Fill in the label for benchmarks whose rule reads the response text.

Three kinds of benchmark appear in this repository, and they differ in *when*
the label can exist:

1. **Reference-answer benchmarks.** The label compares the response against
   accepted answers (POPE, TextVQA, HallusionBench), so
   :mod:`scripts.collect_evidence` can decide it in the same pass that generates
   the response.
2. **Published-run benchmarks.** The response already exists and arrives with
   the record, label included.
3. **Response-text benchmarks.** The label is computed *from the response* by an
   external scorer. CHAIR is this kind: its official rule parses the caption,
   keeps the object mentions the COCO vocabulary recognises, and counts the ones
   COCO does not annotate for that image. No response, no label.

This script is the pass that closes (3). It reads an evidence file collected
with ``"label_from_response": true``, scores every response with the rule named
by ``--rule``, and writes the labels back out. It refuses to run on a file whose
records already carry labels, so a label can never be silently overwritten.

The CHAIRs implementation is a port of the official vocabulary and rules
(``github.com/LisaAnne/Hallucination``): the synonym groups, the multi-word and
animal/vehicle collapsing, the COCO instance dump plus the reference captions as
ground truth, and ``CHAIRs = hallucinated captions / captions``.

Two external files are needed and neither travels with this repository:

* ``--synonyms``  the official ``synonyms.txt`` from the CHAIR release
* ``--annotation-dir``  a directory holding ``instances_val2014.json`` and
  ``captions_val2014.json``

``inflect`` is required for the singularisation the official vocabulary assumes
(``toilets`` -> ``toilet``). Without it the rule silently stops matching plurals
and every count moves, so it is a hard requirement rather than an optional nicety.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
#: A COCO val2014 file name, which is where the image id has to come from.
IMAGE_ID_RE = re.compile(r"(\d{12})\.(?:jpg|png)$", re.I)

#: Multi-word objects, and what the official vocabulary collapses them to.
DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light",
    "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
    "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
    "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear",
    "hair drier", "potted plant", "bow tie", "laptop computer",
    "stove top oven", "home plate", "train track",
]
ANIMALS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
           "zebra", "giraffe", "animal", "cub"]
VEHICLES = ["jet", "train"]


def _require_inflect():
    try:
        import inflect  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "labelling from the response text needs the singulariser the official "
            "CHAIR vocabulary was built with: pip install inflect"
        ) from exc
    import inflect

    return inflect.engine()


class ChairVocabulary:
    """The official object vocabulary and the parsing that goes with it."""

    def __init__(self, synonym_path) -> None:
        self.inflector = _require_inflect()
        groups = [
            line.strip().split(", ")
            for line in pathlib.Path(synonym_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.objects = {word for group in groups for word in group}
        self.canonical = {word: group[0] for group in groups for word in group}
        self.double = {word: word for word in DOUBLE_WORDS}
        for animal in ANIMALS:
            self.double[f"baby {animal}"] = animal
            self.double[f"adult {animal}"] = animal
        for vehicle in VEHICLES:
            self.double[f"passenger {vehicle}"] = vehicle
        self.double["bow tie"] = "tie"
        self.double["toilet seat"] = "toilet"
        self.double["wine glas"] = "wine glass"

    def singular(self, word: str) -> str:
        singular = self.inflector.singular_noun(word)
        return str(singular) if singular else word

    def caption_objects(self, caption: str) -> Tuple[List[str], List[Tuple[str, str, int]]]:
        """Return every token, then the ``(surface, canonical, index)`` mentions."""
        raw = [self.singular(word) for word in TOKEN_RE.findall(str(caption).lower())]
        combined: List[str] = []
        indices: List[int] = []
        index = 0
        while index < len(raw):
            indices.append(index)
            pair = " ".join(raw[index:index + 2])
            if pair in self.double:
                combined.append(self.double[pair])
                index += 2
            else:
                combined.append(raw[index])
                index += 1
        if "toilet" in combined and "seat" in combined:
            keep = [(word, idx) for word, idx in zip(combined, indices) if word != "seat"]
            combined = [word for word, _ in keep]
            indices = [idx for _, idx in keep]
        selected = [
            (word, self.canonical[word], idx)
            for word, idx in zip(combined, indices)
            if word in self.objects
        ]
        return combined, selected


def build_ground_truth(instances, captions, image_ids: Iterable[int],
                       vocabulary: ChairVocabulary) -> Dict[int, set]:
    """COCO instances plus the reference captions, restricted to ``image_ids``."""
    wanted = set(map(int, image_ids))
    id_to_name = {int(row["id"]): row["name"] for row in instances["categories"]}
    ground_truth: Dict[int, set] = {image_id: set() for image_id in wanted}
    for annotation in instances["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in ground_truth:
            ground_truth[image_id].add(vocabulary.canonical[id_to_name[int(annotation["category_id"])]])
    for annotation in captions["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in ground_truth:
            _, objects = vocabulary.caption_objects(annotation["caption"])
            ground_truth[image_id].update(canonical for _, canonical, _ in objects)
    return ground_truth


def chair_scores(captions_by_image: Sequence[Tuple[int, str]],
                 annotation_dir, synonyms) -> List[Dict[str, object]]:
    """Score ``(image_id, caption)`` pairs with the official CHAIRs rule."""
    annotation_dir = pathlib.Path(annotation_dir)
    instances_path = annotation_dir / "instances_val2014.json"
    captions_path = annotation_dir / "captions_val2014.json"
    for path in (instances_path, captions_path):
        if not path.is_file():
            raise SystemExit(
                f"the official CHAIR rule needs {path.name} for ground truth; "
                f"{path} does not exist. --annotation-dir must hold both."
            )
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    vocabulary = ChairVocabulary(synonyms)
    ground_truth = build_ground_truth(
        instances, captions, [image_id for image_id, _ in captions_by_image], vocabulary
    )
    out: List[Dict[str, object]] = []
    for image_id, caption in captions_by_image:
        words, objects = vocabulary.caption_objects(caption)
        hallucinated = [
            {"surface": surface, "canonical": canonical, "token_index": token_index}
            for surface, canonical, token_index in objects
            if canonical not in ground_truth[int(image_id)]
        ]
        out.append({
            "image_id": int(image_id),
            "tokenized_words": words,
            "num_object_mentions": len(objects),
            "num_hallucinated_mentions": len(hallucinated),
            "hallucinated_objects": hallucinated,
            "chair_s": int(bool(hallucinated)),
            "chair_i": (len(hallucinated) / len(objects)) if objects else 0.0,
        })
    return out


def image_id_of(record: Dict[str, object]) -> int:
    """The COCO image id, taken from the image path.

    Not from the record id: CHAIR's own ids read ``CHAIR:<n>:response:<k>`` and
    ``n`` is that dataset's internal index, not a COCO id. Reading it as one
    scores every caption against the wrong image and reports a hallucination rate
    of 0.996, which is how this was found.
    """
    match = IMAGE_ID_RE.search(str(record.get("image") or ""))
    if not match:
        raise SystemExit(
            f"cannot read a COCO image id from {record.get('image')!r}; the CHAIR "
            "rule needs the 12-digit id that COCO file names carry"
        )
    return int(match.group(1))


def label_records(records, rule: str, annotation_dir, synonyms) -> List[Dict[str, object]]:
    if rule != "chair-s":
        raise SystemExit(f"unknown --rule {rule!r}; the only rule implemented is 'chair-s'")
    missing = [r.get("id") for r in records if not str(r.get("answer") or "").strip()]
    if missing:
        raise SystemExit(
            f"{len(missing)} records have no answer to label (first: {missing[0]!r}). "
            "This pass reads the response text; collect the responses first."
        )
    already = [r.get("id") for r in records if r.get("label") is not None]
    if already:
        raise SystemExit(
            f"{len(already)} records already carry a label (first: {already[0]!r}); "
            "refusing to overwrite labels with a different rule"
        )
    pairs = [(image_id_of(r), str(r["answer"])) for r in records]
    scores = chair_scores(pairs, annotation_dir, synonyms)
    rows = []
    for record, score in zip(records, scores):
        row = dict(record)
        row["label"] = int(score["chair_s"])
        row["label_source"] = "CHAIRs-native-any-hallucinated-object"
        row["label_rule"] = "chair-s: any object mentioned that COCO does not annotate"
        row["chair_s"] = score["chair_s"]
        row["chair_i"] = round(float(score["chair_i"]), 6)
        row["num_object_mentions"] = score["num_object_mentions"]
        row["num_hallucinated_mentions"] = score["num_hallucinated_mentions"]
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="evidence JSONL with responses but no labels")
    parser.add_argument("--out", required=True, help="where the labelled JSONL goes")
    parser.add_argument("--rule", default="chair-s", choices=("chair-s",))
    parser.add_argument("--annotation-dir", required=True,
                        help="directory holding instances_val2014.json and captions_val2014.json")
    parser.add_argument("--synonyms", required=True, help="the official CHAIR synonyms.txt")
    args = parser.parse_args()

    source = pathlib.Path(args.data)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"{source} holds no records")
    labelled = label_records(records, args.rule, args.annotation_dir, args.synonyms)

    destination = pathlib.Path(args.out)
    with destination.open("w", encoding="utf-8") as handle:
        for row in labelled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    positives = sum(int(r["label"]) for r in labelled)
    mentions = sum(int(r["num_object_mentions"]) for r in labelled)
    hallucinated = sum(int(r["num_hallucinated_mentions"]) for r in labelled)
    print(f"[label] rule={args.rule}  records={len(labelled)}  -> {destination}")
    print(f"[label] labels: {{0: {len(labelled) - positives}, 1: {positives}}}")
    print(f"[label] CHAIRs = {positives / len(labelled):.4f}  "
          f"CHAIRi = {(hallucinated / mentions) if mentions else 0.0:.6f}")
    if len({int(r['label']) for r in labelled}) < 2:
        print("[label] !! every response got the same label; nothing can be ranked on this file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
