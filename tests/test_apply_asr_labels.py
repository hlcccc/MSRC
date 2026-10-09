"""Turning judge verdicts into labels, including the items the judge could not rule on.

This is the step that lets the official MM-SafetyBench criterion reach an
indicator. Two things about it are easy to get wrong in ways nothing downstream
would notice:

  * an unparseable verdict must be DROPPED, not read as compliance. The safety
    label's positive class is "the model did not refuse", so defaulting an unknown
    verdict to either side invents an observation.
  * a verdict whose id is not in the evidence must be ignored, not appended. An
    appended row would enter the evaluation with no evidence attached to it.

Both are tested here against synthetic verdicts, so the real judge does not have to
have run for the labelling step to be verified.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "apply_asr_labels_under_test", ROOT / "scripts" / "apply_asr_labels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


labels = _load()


def _evidence(n=12):
    return [
        {
            "id": "mmsafety:Health_Consultation:TYPO:%d" % i,
            "category": "Health_Consultation",
            "question": "q%d" % i,
            "answer": "a%d" % i,
            "label": i % 2,                      # the keyword-proxy label
            "evidence": {"risk_family": "safety"},
        }
        for i in range(n)
    ]


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def _run(tmp_path, evidence, verdicts, extra=()):
    src = tmp_path / "evidence.jsonl"
    asr = tmp_path / "asr.jsonl"
    out = tmp_path / "labelled.jsonl"
    _write(src, evidence)
    _write(asr, verdicts)
    argv = sys.argv
    sys.argv = ["apply_asr_labels.py", "--evidence", str(src), "--asr", str(asr),
                "--out", str(out), *extra]
    try:
        rc = labels.main()
    finally:
        sys.argv = argv
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows, out


def test_an_unparseable_verdict_drops_the_item_rather_than_guessing(tmp_path):
    evidence = _evidence(6)
    verdicts = [
        {"id": r["id"], "category": r["category"],
         "judge_output": "I cannot determine" if i == 2 else "safe",
         "is_safe": None if i == 2 else "safe"}
        for i, r in enumerate(evidence)
    ]
    rows, _ = _run(tmp_path, evidence, verdicts)
    assert len(rows) == 5, "the undecided item must not be counted either way"
    assert evidence[2]["id"] not in {r["id"] for r in rows}


def test_a_verdict_for_an_id_that_is_not_in_the_evidence_is_ignored(tmp_path):
    evidence = _evidence(4)
    verdicts = [{"id": r["id"], "category": r["category"],
                 "judge_output": "safe", "is_safe": "safe"} for r in evidence]
    verdicts.append({"id": "mmsafety:NotAThing:TYPO:0", "category": "X",
                     "judge_output": "unsafe", "is_safe": "unsafe"})
    rows, _ = _run(tmp_path, evidence, verdicts)
    assert len(rows) == 4
    assert all(r["evidence"] for r in rows), "a row with no evidence must never appear"


def test_the_unsafe_verdict_is_label_one(tmp_path):
    evidence = _evidence(4)
    verdicts = [
        {"id": r["id"], "category": r["category"],
         "judge_output": "unsafe" if i < 2 else "safe",
         "is_safe": "unsafe" if i < 2 else "safe"}
        for i, r in enumerate(evidence)
    ]
    rows, _ = _run(tmp_path, evidence, verdicts)
    assert [r["label"] for r in rows] == [1, 1, 0, 0]


def test_the_two_label_rules_are_both_kept_so_they_can_be_compared(tmp_path):
    """The proxy label is moved, not overwritten.

    The whole point of judging is to compare the official criterion against the
    keyword proxy; discarding the proxy would make that comparison impossible
    after the fact.
    """
    evidence = _evidence(4)
    verdicts = [
        {"id": r["id"], "category": r["category"],
         "judge_output": "unsafe", "is_safe": "unsafe"} for r in evidence
    ]
    rows, _ = _run(tmp_path, evidence, verdicts)
    for row, src in zip(rows, evidence):
        assert row["label_keyword_proxy"] == src["label"]
        assert row["label_source"] == "mmsafety_asr_judge"
        assert row["judge_output"] == "unsafe"
        assert row["label"] == 1


def test_a_summary_records_how_many_were_dropped(tmp_path):
    evidence = _evidence(6)
    verdicts = [
        {"id": r["id"], "category": r["category"],
         "judge_output": None if i == 1 else "safe",
         "is_safe": None if i == 1 else "safe"}
        for i, r in enumerate(evidence)
    ]
    summary = tmp_path / "summary.json"
    rows, _ = _run(tmp_path, evidence, verdicts, extra=("--summary", str(summary)))
    got = json.loads(summary.read_text(encoding="utf-8"))
    assert got["n_evidence"] == 6
    assert got["n_kept"] == 5
    assert got["n_dropped_unparsed"] == 1
    assert got["judge_unsafe_rate"] == pytest.approx(0.0)
