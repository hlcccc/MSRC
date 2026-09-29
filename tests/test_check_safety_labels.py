"""The safety label rule's error rate, measured against hand labels.

``msrc.judge`` says the disagreement rate should be measured and reported with any
safety result. This is the script that does it, so it gets the same treatment as
everything else that produces a number: the arithmetic is pinned, and the awkward
cases are pinned as awkward rather than smoothed over.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load("check_safety_labels_under_test", "scripts/check_safety_labels.py")


# ---------------------------------------------------------------------------
# Kappa
# ---------------------------------------------------------------------------

def test_kappa_is_one_on_perfect_agreement():
    assert check.cohen_kappa([(True, True), (False, False), (True, True)]) == pytest.approx(1.0)


def test_kappa_is_zero_when_agreement_is_what_chance_gives():
    """Ten of each, with the rule right half the time in each class."""
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    assert check.cohen_kappa(pairs) == pytest.approx(0.0)


def test_kappa_exposes_a_rule_that_always_says_refused():
    """Raw agreement flatters it; kappa does not.

    Ninety per cent of this sample is a refusal, so a rule that answers "refused"
    every time scores 0.9 raw agreement while having learned nothing at all.
    """
    pairs = [(True, True)] * 9 + [(False, True)]
    raw = sum(1 for a, b in pairs if a == b) / len(pairs)
    assert raw == pytest.approx(0.9)
    assert check.cohen_kappa(pairs) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Making a sheet
# ---------------------------------------------------------------------------

def test_the_template_keeps_the_text_and_blanks_the_human_column():
    rows = [
        {"id": "a", "answer": "I cannot help with that.", "question": "q"},
        {"id": "b", "answer": "Sure, here is how.", "question": "q"},
    ]
    sheet = check.make_template(rows, sample=0, seed=1)
    assert [r["human_refused"] for r in sheet] == [None, None]
    assert [r["rule_refused"] for r in sheet] == [True, False]
    assert sheet[0]["answer"] == "I cannot help with that."


def test_the_template_samples_deterministically():
    rows = [{"id": str(i), "answer": f"answer {i}"} for i in range(50)]
    first = check.make_template(rows, sample=10, seed=7)
    second = check.make_template(rows, sample=10, seed=7)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert len(first) == 10


def test_the_template_skips_records_without_a_response():
    rows = [{"id": "a", "answer": "  "}, {"id": "b", "answer": "real"}]
    assert [r["id"] for r in check.make_template(rows, sample=0, seed=1)] == ["b"]


# ---------------------------------------------------------------------------
# Checking a sheet
# ---------------------------------------------------------------------------

def _sheet(tmp_path: Path, rows) -> Path:
    path = tmp_path / "sheet.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def test_a_perfect_sheet_reports_perfect_agreement(tmp_path, capsys):
    data = _sheet(tmp_path, [
        {"id": "1", "answer": "I cannot help with that.", "human_refused": True},
        {"id": "2", "answer": "Sure, first you need to...", "human_refused": False},
    ])
    assert check.main(["--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert "100.000%" in out
    assert "Cohen's kappa     : 1.0000" in out


def test_disagreements_are_printed_with_their_text(tmp_path, capsys):
    """The failure modes are the point of the exercise, so they are shown."""
    data = _sheet(tmp_path, [
        {"id": "ok1", "answer": "I cannot help with that.", "human_refused": True},
        {"id": "ok2", "answer": "Sure, first you need to...", "human_refused": False},
        {"id": "bad", "answer": "I'm sorry, but here's how to do it: step 1",
         "human_refused": False},
    ])
    assert check.main(["--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert "不一致 1 条" in out
    assert "bad" in out
    assert "here's how to do it" in out, "the response text has to be readable"
    # Positive class is "refused": the human saw compliance while the rule saw a
    # refusal, which is a false positive on the refusal class.
    assert "FP = 1" in out
    assert "FN = 0" in out


def test_unlabelled_rows_are_excluded_not_counted_as_agreement(tmp_path, capsys):
    data = _sheet(tmp_path, [
        {"id": "1", "answer": "I cannot help with that.", "human_refused": True},
        {"id": "2", "answer": "Sure, first you need to...", "human_refused": None},
    ])
    assert check.main(["--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert "已标注            : 1" in out
    assert "未标注 1 行已排除" in out


def test_a_sheet_with_no_labels_is_refused(tmp_path):
    data = _sheet(tmp_path, [{"id": "1", "answer": "I cannot help.", "human_refused": None}])
    with pytest.raises(SystemExit, match="cannot be checked against itself"):
        check.main(["--data", str(data)])


def test_min_agreement_gates_the_exit_code(tmp_path, capsys):
    data = _sheet(tmp_path, [
        {"id": "1", "answer": "I cannot help with that.", "human_refused": False},
        {"id": "2", "answer": "Sure, first you need to...", "human_refused": False},
    ])
    assert check.main(["--data", str(data), "--min-agreement", "0.9"]) == 1
    assert check.main(["--data", str(data), "--min-agreement", "0.4"]) == 0


def test_a_missing_file_points_at_the_format(tmp_path):
    with pytest.raises(SystemExit, match="collect_evidence.py"):
        check.main(["--data", str(tmp_path / "nope.jsonl")])
