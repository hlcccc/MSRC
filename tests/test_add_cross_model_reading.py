"""Three CLIP models were run on the same COCO images, so each one's cell already holds
two spare descriptions per item. These tests pin the two things that would make adding
them a silent corruption rather than an improvement: the join, and what the script is
allowed to touch.

The rule the whole file exists to enforce: a merge may add to the cross-model channel
and nothing else. The answer is the item and the label is derived from the answer, so a
merge that moved either would be producing a different cell and calling it the same one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "add_cross_model_reading", ROOT / "scripts" / "add_cross_model_reading.py")
acm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acm)


def write_jsonl(path: Path, rows) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def cell_row(item_id, answer, label=0, image="/data/COCO_val2014_000000000001.jpg",
             question="Describe this image in detail", cross_model=None):
    return {
        "id": item_id, "image": image, "question": question,
        "answer": answer, "label": label,
        "evidence": {
            "question": question, "answer": answer, "image": image,
            "primary": {"text": answer},
            "views": [{"text": "a view"}],
            "resamples": [{"text": "a resample"}],
            "verifications": [{"text": "yes"}],
            "cross_model": list(cross_model or []),
            "ocr_texts": ["text"],
        },
    }


def run(tmp_path, target_rows, donor_rows, name="out.jsonl", receipt=False):
    t = write_jsonl(tmp_path / "target.jsonl", target_rows)
    d = write_jsonl(tmp_path / "donor.jsonl", donor_rows)
    out = tmp_path / name
    argv = ["--evidence", str(t), "--donor", str(d), "--out", str(out)]
    if receipt:
        argv += ["--receipt", str(tmp_path / "receipt.json")]
    acm.main(argv)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    return rows, out


# -- the join --------------------------------------------------------------

def test_a_matching_donor_answer_is_appended(tmp_path):
    rows, _ = run(tmp_path,
                  [cell_row("chair:1", "a cat on a mat")],
                  [cell_row("chair:1", "a cat sitting on a rug")])
    channel = rows[0]["evidence"]["cross_model"]
    assert len(channel) == 1
    assert channel[0]["text"] == "a cat sitting on a rug"


def test_existing_descriptions_are_kept_not_replaced(tmp_path):
    rows, _ = run(tmp_path,
                  [cell_row("chair:1", "a cat", cross_model=[{"text": "first opinion"}])],
                  [cell_row("chair:1", "second opinion")])
    texts = [s["text"] for s in rows[0]["evidence"]["cross_model"]]
    assert texts == ["first opinion", "second opinion"]


def test_an_item_missing_from_the_donor_is_left_alone(tmp_path):
    rows, _ = run(tmp_path,
                  [cell_row("chair:1", "a cat"), cell_row("chair:2", "a dog")],
                  [cell_row("chair:1", "a cat here")])
    assert len(rows[0]["evidence"]["cross_model"]) == 1
    assert rows[1]["evidence"]["cross_model"] == []


def test_a_donor_whose_image_disagrees_is_refused(tmp_path):
    """Same id, different picture: attaching the description would be a fabrication."""
    rows, _ = run(tmp_path,
                  [cell_row("chair:1", "a cat")],
                  [cell_row("chair:1", "a dog", image="/data/COCO_val2014_000000000999.jpg")])
    assert rows[0]["evidence"]["cross_model"] == []


def test_a_donor_whose_question_disagrees_is_refused(tmp_path):
    rows, _ = run(tmp_path,
                  [cell_row("chair:1", "a cat")],
                  [cell_row("chair:1", "a dog", question="What colour is the car?")])
    assert rows[0]["evidence"]["cross_model"] == []


def test_an_empty_donor_answer_is_not_added(tmp_path):
    rows, _ = run(tmp_path,
                  [cell_row("chair:1", "a cat")],
                  [cell_row("chair:1", "   ")])
    assert rows[0]["evidence"]["cross_model"] == []


def test_a_donor_that_shares_no_ids_is_refused_loudly(tmp_path):
    """The whole file being wrong is a different failure from one item missing."""
    with pytest.raises(SystemExit, match="none of its"):
        run(tmp_path, [cell_row("chair:1", "a cat")], [cell_row("other:1", "a dog")])


def test_running_twice_does_not_duplicate_the_description(tmp_path):
    t = write_jsonl(tmp_path / "target.jsonl", [cell_row("chair:1", "a cat")])
    d = write_jsonl(tmp_path / "donor.jsonl", [cell_row("chair:1", "a dog")])
    first = tmp_path / "first.jsonl"
    acm.main(["--evidence", str(t), "--donor", str(d), "--out", str(first)])
    second = tmp_path / "second.jsonl"
    acm.main(["--evidence", str(first), "--donor", str(d), "--out", str(second)])
    rows = [json.loads(l) for l in second.read_text(encoding="utf-8").splitlines()]
    assert [s["text"] for s in rows[0]["evidence"]["cross_model"]] == ["a dog"]


def test_two_donors_both_contribute(tmp_path):
    t = write_jsonl(tmp_path / "target.jsonl", [cell_row("chair:1", "a cat")])
    d1 = write_jsonl(tmp_path / "d1.jsonl", [cell_row("chair:1", "from one")])
    d2 = write_jsonl(tmp_path / "d2.jsonl", [cell_row("chair:1", "from two")])
    out = tmp_path / "out.jsonl"
    acm.main(["--evidence", str(t), "--donor", str(d1), "--donor", str(d2), "--out", str(out)])
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [s["text"] for s in rows[0]["evidence"]["cross_model"]] == ["from one", "from two"]


# -- what must not move ----------------------------------------------------

def test_the_answer_and_label_are_copied_through_unchanged(tmp_path):
    rows, _ = run(tmp_path,
                  [cell_row("chair:1", "the original caption", label=1)],
                  [cell_row("chair:1", "another model's caption", label=0)])
    assert rows[0]["answer"] == "the original caption"
    assert rows[0]["label"] == 1


def test_every_other_channel_survives_untouched(tmp_path):
    original = cell_row("chair:1", "a cat")
    before = json.loads(json.dumps(original["evidence"]))
    rows, _ = run(tmp_path, [original], [cell_row("chair:1", "a dog")])
    after = rows[0]["evidence"]
    for key in ("primary", "views", "resamples", "verifications", "ocr_texts"):
        assert after[key] == before[key], key
    assert after["answer"] == before["answer"]
    assert after["question"] == before["question"]


def test_the_added_sample_fills_the_same_six_fields_a_collector_writes(tmp_path):
    """A half-populated sample would read as an internal signal of zero rather than absent."""
    rows, _ = run(tmp_path,
                  [cell_row("chair:1", "a cat")],
                  [cell_row("chair:1", "a dog")])
    added = rows[0]["evidence"]["cross_model"][0]
    assert set(added) == set(acm.SAMPLE_KEYS)
    assert added["text"] == "a dog"
    assert all(added[k] is None for k in acm.SAMPLE_KEYS if k != "text")


# -- the receipt -----------------------------------------------------------

def test_the_receipt_records_what_was_added_and_from_where(tmp_path):
    rows, out = run(tmp_path,
                    [cell_row("chair:1", "a cat")],
                    [cell_row("chair:1", "a dog")], receipt=True)
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["rows"] == 1
    assert receipt["donors"][0]["added"] == 1
    assert receipt["donors"][0]["sha256"]
    assert receipt["out_sha256"] == acm.sha256_of(out)
    assert "answer" in receipt["untouched"] and "label" in receipt["untouched"]
