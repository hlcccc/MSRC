"""The generation-budget check, which exists because it caught a real defect.

`cell_Qwen2.5-VL-7B_chair.jsonl` held 1,200 rows -- the right number -- of which 460 had
been cut off at 64 tokens by an early run and 740 generated at 128 by the run that
resumed it. Resuming is keyed on the item, so the early answers survived. Nothing in the
row count, the label distribution or the channel availability said anything was wrong;
the token counts did, because a response cut off by its budget has *exactly* that many
tokens. Those 460 items hallucinated at 9.1% against 19.9% for the rest, so the cell was
easier than either budget alone would have made it.

These tests pin the detection, not the incident.
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
    "audit_generation_budget", ROOT / "scripts" / "audit_generation_budget.py")
agb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agb)


# -- the detection ---------------------------------------------------------

def test_a_single_clamp_is_reported_as_single():
    verdict, clamps = agb.classify([128] * 80 + [40, 55, 72, 90])
    assert verdict == "single"
    assert clamps == {128: 80}


def test_two_clamps_are_reported_as_mixed():
    """The shape the incident actually had: 460 at 64 and 740 at 128."""
    lengths = [64] * 460 + [128] * 740
    verdict, clamps = agb.classify(lengths)
    assert verdict == "mixed"
    assert set(clamps) == {64, 128}


def test_a_few_rows_on_a_second_clamp_is_not_mixed():
    """Short-answer cells legitimately brush against 64 without being two experiments."""
    verdict, clamps = agb.classify([64] * 5 + [20] * 995)
    assert verdict == "none"


def test_a_cell_below_the_material_share_on_both_clamps_is_not_mixed():
    # The filler sits away from every candidate budget, so only the two small piles are
    # candidates and neither is material.
    lengths = [64] * 40 + [128] * 40 + [20] * 460 + [90] * 460
    verdict, _ = agb.classify(lengths)
    assert verdict == "none"


def test_lengths_near_a_budget_count_as_that_budget():
    """A stop token emitted one step early is still that budget."""
    verdict, clamps = agb.classify([63] * 50 + [127] * 50)
    assert verdict == "mixed"
    assert set(clamps) == {64, 128}


def test_an_empty_cell_does_not_crash():
    assert agb.classify([]) == ("none", {})


def test_the_whole_incident_is_reproduced_from_the_token_counts_alone():
    """460 at 64 and 740 at 128 -- the numbers read off the real file."""
    verdict, clamps = agb.classify([64] * 460 + [128] * 740)
    share64 = clamps[64] / (clamps[64] + clamps[128])
    assert verdict == "mixed"
    assert share64 == pytest.approx(460 / 1200)


# -- the driver ------------------------------------------------------------

def test_a_cell_whose_file_is_missing_is_a_problem(tmp_path, capsys):
    assert agb.main(["--cell", f"M|D={tmp_path / 'nope.jsonl'}",
                     "--tokenizer", str(tmp_path / "tok")]) == 1
    assert "文件不存在" in capsys.readouterr().out


def test_a_cell_with_no_answers_is_a_problem(tmp_path, capsys):
    path = tmp_path / "empty.jsonl"
    path.write_text(json.dumps({"id": "a", "answer": "   "}) + "\n", encoding="utf-8")
    assert agb.main(["--cell", f"M|D={path}", "--tokenizer", str(tmp_path / "tok")]) == 1


def test_a_cell_spec_is_parsed_into_three_parts(tmp_path):
    model, dataset, path = agb.parse_cell(f"M|D={tmp_path / 'x.jsonl'}")
    assert (model, dataset) == ("M", "D")


def test_the_repository_has_no_unexplained_second_budget():
    """Guards the check itself: the budgets searched are the ones collectors pass."""
    assert 64 in agb.CANDIDATE_BUDGETS, "the collector's default"
    assert 128 in agb.CANDIDATE_BUDGETS, "what the CHAIR runs pass explicitly"
