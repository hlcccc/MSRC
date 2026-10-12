"""OKVQA preparation and deferred labelling.

Two things here would fail quietly and corrupt a headline number rather than raise.

The label join (``scripts/apply_labels.py``) matches evidence collected on one machine
against answers held on another. If it matched on the wrong key, or trusted a file that
had drifted out of step, every item would still get a label and the accuracy would still
print -- just for the wrong items. So the join is tested for the drift it is meant to
catch, not only for the happy path.

The preparation (``scripts/prepare_okvqa.py``) decides which questions are usable and
carries the COCO image id that :func:`evaluation.groups.image_group_id` groups on. A
wrong image identity there would put two questions about one photograph on opposite
sides of the dev/test split, and nothing downstream would notice.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


okvqa = _load("prepare_okvqa")
labels = _load("apply_labels")

from evaluation.groups import image_group_id  # noqa: E402


def write_jsonl(path: Path, rows) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# -- OKVQA preparation -----------------------------------------------------

def test_coco_name_is_the_official_twelve_digit_form():
    assert okvqa.coco_name(42) == "COCO_val2014_000000000042.jpg"
    assert okvqa.coco_name(297147) == "COCO_val2014_000000297147.jpg"


def test_answers_are_read_in_the_release_order():
    annotation = {"answers": [{"answer": "cat"}, {"answer": "dog"}, {"answer": "cat"}]}
    assert okvqa.answers_of(annotation) == ["cat", "dog", "cat"]


def test_annotations_are_keyed_by_question_id(tmp_path):
    path = tmp_path / "ann.json"
    path.write_text(json.dumps({"annotations": [
        {"question_id": 7, "answers": [{"answer": "a"}]},
        {"question_id": 9, "answers": [{"answer": "b"}]},
    ]}), encoding="utf-8")
    assert set(okvqa.load_annotations(path)) == {7, 9}


def test_image_identity_is_the_coco_id_so_one_photo_never_splits(tmp_path):
    """Two questions about one photograph must share a group."""
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"questions": [
        {"question_id": 1, "image_id": 555, "question": "first?"},
        {"question_id": 2, "image_id": 555, "question": "second?"},
    ]}), encoding="utf-8")
    annotations = tmp_path / "a.json"
    annotations.write_text(json.dumps({"annotations": [
        {"question_id": 1, "answers": [{"answer": "x"}] * 10},
        {"question_id": 2, "answers": [{"answer": "y"}] * 10},
    ]}), encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    (images / okvqa.coco_name(555)).write_bytes(b"jpeg")

    out = tmp_path / "in.jsonl"
    public = tmp_path / "pub.jsonl"
    okvqa.main(["--questions", str(questions), "--annotations", str(annotations),
                "--images-dir", str(images), "--out", str(out),
                "--public-out", str(public)])

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["image_id"] == 555 for row in rows)
    assert len({image_group_id(row) for row in rows}) == 1
    assert all(len(row["gold_answers"]) == 10 for row in rows)


def test_the_public_file_carries_no_answers_and_no_label(tmp_path):
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"questions": [
        {"question_id": 1, "image_id": 5, "question": "what?"}]}), encoding="utf-8")
    annotations = tmp_path / "a.json"
    annotations.write_text(json.dumps({"annotations": [
        {"question_id": 1, "answers": [{"answer": "secret"}] * 10}]}), encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    (images / okvqa.coco_name(5)).write_bytes(b"jpeg")

    public = tmp_path / "pub.jsonl"
    okvqa.main(["--questions", str(questions), "--annotations", str(annotations),
                "--images-dir", str(images), "--out", str(tmp_path / "in.jsonl"),
                "--public-out", str(public)])

    text = public.read_text(encoding="utf-8")
    assert "secret" not in text
    row = json.loads(text.splitlines()[0])
    assert "gold_answers" not in row and "label" not in row
    assert row["label_from_response"] == "okvqa"


def test_a_question_whose_image_is_absent_is_skipped_not_invented(tmp_path):
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"questions": [
        {"question_id": 1, "image_id": 5, "question": "here?"},
        {"question_id": 2, "image_id": 6, "question": "gone?"}]}), encoding="utf-8")
    annotations = tmp_path / "a.json"
    annotations.write_text(json.dumps({"annotations": [
        {"question_id": 1, "answers": [{"answer": "x"}] * 10},
        {"question_id": 2, "answers": [{"answer": "y"}] * 10}]}), encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    (images / okvqa.coco_name(5)).write_bytes(b"jpeg")

    out = tmp_path / "in.jsonl"
    okvqa.main(["--questions", str(questions), "--annotations", str(annotations),
                "--images-dir", str(images), "--out", str(out)])

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["okvqa:validation:1"]


def test_limit_takes_the_lowest_question_ids_so_a_subset_is_reproducible(tmp_path):
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"questions": [
        {"question_id": 9, "image_id": 5, "question": "nine?"},
        {"question_id": 3, "image_id": 6, "question": "three?"}]}), encoding="utf-8")
    annotations = tmp_path / "a.json"
    annotations.write_text(json.dumps({"annotations": [
        {"question_id": 9, "answers": [{"answer": "x"}] * 10},
        {"question_id": 3, "answers": [{"answer": "y"}] * 10}]}), encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    for image_id in (5, 6):
        (images / okvqa.coco_name(image_id)).write_bytes(b"jpeg")

    out = tmp_path / "in.jsonl"
    okvqa.main(["--questions", str(questions), "--annotations", str(annotations),
                "--images-dir", str(images), "--out", str(out), "--limit", "1"])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["okvqa:validation:3"]


# -- deferred labelling ----------------------------------------------------

def _vqa_row(item_id, question, answer, gold):
    return ({"id": item_id, "question": question, "answer": answer},
            {"id": item_id, "question": question, "gold_answers": gold})


def test_the_vqa_rule_is_the_one_the_collector_uses(tmp_path):
    evidence, gold = _vqa_row("okvqa:1", "what?", "cat", ["cat"] * 10)
    ev = write_jsonl(tmp_path / "ev.jsonl", [evidence])
    go = write_jsonl(tmp_path / "gold.jsonl", [gold])
    out = tmp_path / "out.jsonl"
    labels.main(["--evidence", str(ev), "--gold", str(go), "--out", str(out), "--rule", "vqa"])

    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["label"] == 0
    assert row["label_rule"] == "vqa_agreement_div3"
    assert "label_from_response" not in row


def test_a_wrong_answer_is_labelled_one(tmp_path):
    evidence, gold = _vqa_row("okvqa:1", "what?", "dog", ["cat"] * 10)
    ev = write_jsonl(tmp_path / "ev.jsonl", [evidence])
    go = write_jsonl(tmp_path / "gold.jsonl", [gold])
    out = tmp_path / "out.jsonl"
    labels.main(["--evidence", str(ev), "--gold", str(go), "--out", str(out), "--rule", "vqa"])
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["label"] == 1


def test_a_drifted_question_is_dropped_rather_than_mislabelled(tmp_path):
    """Same id, different question: the files are out of step, so nothing is written."""
    evidence, _ = _vqa_row("okvqa:1", "what colour?", "cat", None)
    _, gold = _vqa_row("okvqa:1", "what animal?", None, ["cat"] * 10)
    ev = write_jsonl(tmp_path / "ev.jsonl", [evidence])
    go = write_jsonl(tmp_path / "gold.jsonl", [gold])
    out = tmp_path / "out.jsonl"
    labels.main(["--evidence", str(ev), "--gold", str(go), "--out", str(out), "--rule", "vqa"])
    assert out.read_text(encoding="utf-8").strip() == ""


def test_evidence_without_a_matching_gold_row_is_dropped(tmp_path):
    evidence, _ = _vqa_row("okvqa:2", "what?", "cat", None)
    _, gold = _vqa_row("okvqa:1", "what?", None, ["cat"] * 10)
    ev = write_jsonl(tmp_path / "ev.jsonl", [evidence])
    go = write_jsonl(tmp_path / "gold.jsonl", [gold])
    out = tmp_path / "out.jsonl"
    labels.main(["--evidence", str(ev), "--gold", str(go), "--out", str(out), "--rule", "vqa"])
    assert out.read_text(encoding="utf-8").strip() == ""


def test_a_duplicated_gold_id_is_refused(tmp_path):
    evidence, gold = _vqa_row("okvqa:1", "what?", "cat", None)
    _, gold_again = _vqa_row("okvqa:1", "what?", None, ["cat"] * 10)
    ev = write_jsonl(tmp_path / "ev.jsonl", [evidence])
    go = write_jsonl(tmp_path / "gold.jsonl", [gold, gold_again])
    with pytest.raises(SystemExit):
        labels.main(["--evidence", str(ev), "--gold", str(go),
                     "--out", str(tmp_path / "out.jsonl"), "--rule", "vqa"])


def test_an_empty_answer_is_dropped_rather_than_called_correct(tmp_path):
    evidence, gold = _vqa_row("okvqa:1", "what?", "   ", ["cat"] * 10)
    ev = write_jsonl(tmp_path / "ev.jsonl", [evidence])
    go = write_jsonl(tmp_path / "gold.jsonl", [gold])
    out = tmp_path / "out.jsonl"
    labels.main(["--evidence", str(ev), "--gold", str(go), "--out", str(out), "--rule", "vqa"])
    assert out.read_text(encoding="utf-8").strip() == ""


def test_yes_no_rule_marks_an_evasive_response_wrong_and_unparsed(tmp_path):
    evidence = {"id": "pope:1", "question": "is there a cat?", "answer": "maybe"}
    gold = {"id": "pope:1", "question": "is there a cat?", "gold_answers": ["yes"]}
    ev = write_jsonl(tmp_path / "ev.jsonl", [evidence])
    go = write_jsonl(tmp_path / "gold.jsonl", [gold])
    out = tmp_path / "out.jsonl"
    labels.main(["--evidence", str(ev), "--gold", str(go),
                 "--out", str(out), "--rule", "yes_no"])
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["label"] == 1 and row["label_parsed"] is False


def test_a_gold_row_without_answers_is_dropped(tmp_path):
    evidence, gold = _vqa_row("okvqa:1", "what?", "cat", [])
    ev = write_jsonl(tmp_path / "ev.jsonl", [evidence])
    go = write_jsonl(tmp_path / "gold.jsonl", [gold])
    out = tmp_path / "out.jsonl"
    labels.main(["--evidence", str(ev), "--gold", str(go), "--out", str(out), "--rule", "vqa"])
    assert out.read_text(encoding="utf-8").strip() == ""
