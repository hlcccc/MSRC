"""Labelling a benchmark whose rule reads the response text.

CHAIR is the case: its official rule parses the caption and counts the objects it
names that COCO does not annotate for that image, so the label cannot exist until
the caption does. These tests pin the parts that would otherwise fail quietly --
the image id, the vocabulary, and the refusal to overwrite a label -- using a
vocabulary and annotations written here rather than the real 200 MB dump, so the
suite stays fast and does not need the external files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import label_from_response as lfr  # noqa: E402

pytest.importorskip("inflect", reason="the official CHAIR vocabulary singularises with inflect")


SYNONYMS = """\
person, people, human
toilet, toilets, can, commode
cat, cats, kitten
dog, dogs, puppy
car, cars, automobile
bicycle, bicycles, bike
sign, stop sign, street sign
"""


def write_world(tmp_path: Path):
    """A miniature COCO: two images, one object each, one reference caption."""
    synonyms = tmp_path / "synonyms.txt"
    synonyms.write_text(SYNONYMS, encoding="utf-8")
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "instances_val2014.json").write_text(json.dumps({
        "categories": [{"id": 1, "name": "toilet"}, {"id": 2, "name": "cat"}],
        "annotations": [
            {"image_id": 42, "category_id": 1},
            {"image_id": 43, "category_id": 2},
        ],
    }), encoding="utf-8")
    (annotations / "captions_val2014.json").write_text(json.dumps({
        "annotations": [
            {"image_id": 42, "caption": "a white toilet in a bathroom"},
            {"image_id": 43, "caption": "a cat on a mat"},
        ],
    }), encoding="utf-8")
    return synonyms, annotations


def test_the_image_id_comes_from_the_path_not_the_record_id():
    """The bug that scored every caption against the wrong image.

    CHAIR's own ids read ``CHAIR:<n>:response:<k>`` and ``n`` is that dataset's
    internal index. Reading it as a COCO id reported CHAIRs = 0.996 on 1,200 real
    captions, where the true value was 0.44.
    """
    record = {"id": "CHAIR:7:response:0", "image": "/x/COCO_val2014_000000294832.jpg"}
    assert lfr.image_id_of(record) == 294832

    with pytest.raises(SystemExit, match="cannot read a COCO image id"):
        lfr.image_id_of({"id": "CHAIR:7:response:0", "image": "/x/plain.jpg"})


def test_a_mentioned_object_coco_does_not_annotate_is_a_hallucination(tmp_path):
    synonyms, annotations = write_world(tmp_path)
    pairs = [
        (42, "A white toilet in a bathroom."),          # toilet is annotated -> clean
        (42, "A white toilet next to a cat."),          # cat is not there -> hallucinated
        (43, "A cat on a mat."),                        # clean
        (43, "A cat and a bicycle on a mat."),          # bicycle is not there
    ]
    scores = lfr.chair_scores(pairs, annotations, synonyms)
    assert [s["chair_s"] for s in scores] == [0, 1, 0, 1]
    assert scores[1]["hallucinated_objects"][0]["canonical"] == "cat"
    assert scores[3]["hallucinated_objects"][0]["canonical"] == "bicycle"


def test_plurals_are_singularised_before_the_lookup(tmp_path):
    """Without ``inflect`` the vocabulary stops matching plurals and counts move."""
    synonyms, annotations = write_world(tmp_path)
    scores = lfr.chair_scores([(43, "Two cats and three bicycles.")], annotations, synonyms)
    assert scores[0]["num_object_mentions"] == 2, scores[0]["tokenized_words"]


def test_a_multi_word_object_is_collapsed_before_matching(tmp_path):
    """``stop sign`` is one object, not two; the vocabulary says so in ``double``."""
    synonyms, annotations = write_world(tmp_path)
    scores = lfr.chair_scores([(43, "A bicycle near a stop sign.")], annotations, synonyms)
    surfaces = [d["surface"] for d in scores[0]["hallucinated_objects"]]
    assert surfaces == ["bicycle", "stop sign"], surfaces
    # "sign" is the first word of its synonym group, so that is the canonical form.
    assert [d["canonical"] for d in scores[0]["hallucinated_objects"]] == ["bicycle", "sign"]
    assert scores[0]["num_object_mentions"] == 2


def test_labels_are_written_and_an_existing_label_is_never_overwritten(tmp_path):
    synonyms, annotations = write_world(tmp_path)
    records = [
        {"id": "a", "image": "/x/COCO_val2014_000000000042.jpg", "answer": "A toilet."},
        {"id": "b", "image": "/x/COCO_val2014_000000000042.jpg", "answer": "A dog."},
    ]
    labelled = lfr.label_records(records, "chair-s", annotations, synonyms)
    assert [r["label"] for r in labelled] == [0, 1]
    assert all(r["label_source"] == "CHAIRs-native-any-hallucinated-object" for r in labelled)

    records[0]["label"] = 1
    with pytest.raises(SystemExit, match="already carry a label"):
        lfr.label_records(records, "chair-s", annotations, synonyms)


def test_a_response_is_required_because_the_rule_reads_it(tmp_path):
    synonyms, annotations = write_world(tmp_path)
    with pytest.raises(SystemExit, match="no answer to label"):
        lfr.label_records(
            [{"id": "a", "image": "/x/COCO_val2014_000000000042.jpg", "answer": ""}],
            "chair-s", annotations, synonyms,
        )


def test_missing_annotations_is_reported_as_a_missing_file(tmp_path):
    synonyms, _ = write_world(tmp_path)
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    with pytest.raises(SystemExit, match="does not exist"):
        lfr.chair_scores([(42, "A toilet.")], empty, synonyms)
