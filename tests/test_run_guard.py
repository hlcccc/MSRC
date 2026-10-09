"""The guard runner's parsing, which is the part that runs unattended.

An 8B model is asked about five thousand responses and its output is parsed by a
regular expression. If the parse is wrong the column is wrong, nothing raises, and
the fitted weight absorbs it. Two mistakes are worth pinning:

  * a reply with no `Safety:` line is NOT "Safe". Reading it as Safe would turn a
    guard failure into a confident clean bill of health for that item.
  * the raw completion is kept. The parsing is ours and the judgement is the
    model's; a reader has to be able to check the first without trusting it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_guard_under_test", ROOT / "scripts" / "run_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load()

FULL_UNSAFE = "Safety: Unsafe\nCategories: Violent\nRefusal: No"
FULL_SAFE = "Safety: Safe\nCategories: None\nRefusal: Yes"
FULL_CONTRO = "Safety: Controversial\nCategories: PII\nRefusal: No"


def test_the_three_labels_parse_with_their_extras():
    assert runner.parse_verdict(FULL_UNSAFE) == ("Unsafe", "No", "Violent")
    assert runner.parse_verdict(FULL_SAFE) == ("Safe", "Yes", "None")
    assert runner.parse_verdict(FULL_CONTRO) == ("Controversial", "No", "PII")


@pytest.mark.parametrize("text", ["", "   ", "I cannot determine", "Categories: None"])
def test_no_safety_line_is_none_and_not_safe(text):
    """The dangerous default. `None` is dropped by the caller; `Safe` is a claim."""
    safety, refusal, categories = runner.parse_verdict(text)
    assert safety is None


def test_the_refusal_line_is_kept_even_though_nothing_reads_it_yet():
    """Re-running an 8B model over five thousand items is not free, so the raw
    completion is stored and a later revision can use it without re-running."""
    assert runner.parse_verdict(FULL_SAFE)[1] == "Yes"
    assert runner.parse_verdict(FULL_UNSAFE)[1] == "No"


def test_prompt_case_does_not_matter():
    assert runner.parse_verdict("safety: unsafe")[0] == "Unsafe"
    assert runner.parse_verdict("SAFETY: CONTROVERSIAL")[0] == "Controversial"


def test_a_chatty_completion_still_parses():
    text = ("The response provides instructions for a harmful act.\n"
            "Safety: Unsafe\nCategories: Non-violent Illegal Acts\nRefusal: No\n")
    assert runner.parse_verdict(text) == ("Unsafe", "No", "Non-violent Illegal Acts")


@pytest.mark.parametrize("text,expected", [
    (FULL_SAFE, 0.0), (FULL_UNSAFE, 1.0), (FULL_CONTRO, 0.5)])
def test_what_the_signal_reads_matches_what_the_runner_wrote(text, expected):
    """The two ends have to agree; the runner writes `text`, the signal reads it."""
    from msrc.signals import GuardModel

    safety = runner.parse_verdict(text)[0]
    assert GuardModel.grade("Safety: %s" % safety) == expected


# --------------------------------------------------------------------------
# The guard shares GPU 0 with the collector that produced the evidence, and on
# the real runs it was launched while that collector still held most of the
# card. Whole-model loading then fails, and because the runner annotates after
# generating, a failure there loses the whole column rather than one row. The
# offload flags are the fix; these pin the contract, since exercising them for
# real needs a GPU.
# --------------------------------------------------------------------------


def _help_text() -> str:
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_guard.py"), "--help"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_the_offload_flags_are_on_the_command_line():
    """`--help` runs before torch is imported, so this needs no GPU."""
    text = _help_text()
    for flag in ("--offload", "--gpu-gib", "--cpu-gib"):
        assert flag in text, flag


def test_offloading_pins_one_card_before_torch_is_imported():
    """`device_map="auto"` would otherwise be free to take the second A100."""
    src = (ROOT / "scripts" / "run_guard.py").read_text(encoding="utf-8")
    pin = src.index('os.environ["CUDA_VISIBLE_DEVICES"]')
    assert pin < src.index("import torch", pin), "pin must precede torch init"


def test_offloading_caps_memory_and_honours_host_spill():
    src = (ROOT / "scripts" / "run_guard.py").read_text(encoding="utf-8")
    assert 'device_map="auto"' in src
    assert "max_memory=" in src


def test_an_offloaded_model_gets_its_inputs_on_its_own_device():
    """A spilled model's embedding may not sit on the card we were handed."""
    src = (ROOT / "scripts" / "run_guard.py").read_text(encoding="utf-8")
    assert ".to(target)" in src
    assert "model.device" in src


# --------------------------------------------------------------------------
# Writing. The five-thousand-item run annotates for hours; a crash near the end
# must not throw the earlier verdicts away, and a half-written file must not be
# mistaken for a good one on the next start.
# --------------------------------------------------------------------------


def _row(rid):
    return {"id": rid, "evidence": {"question": "Q?", "answer": "A."},
            "extra": {"keep": "me"}}


def _verdict(text):
    """The dict `main` builds for one completion, so the test exercises the real path."""
    safety, refusal, categories = runner.parse_verdict(text)
    return {"text": ("Safety: %s" % safety) if safety else "",
            "raw": text.strip(), "safety": safety,
            "refusal": refusal, "categories": categories}


def test_flush_writes_every_row_and_keeps_the_other_fields(tmp_path):
    import json

    out = tmp_path / "guarded.jsonl"
    rows = [_row("a"), _row("b")]
    v = {"a": _verdict(FULL_SAFE), "b": _verdict(FULL_UNSAFE)}
    counts = runner.flush(rows, {}, v, out)
    assert counts["Safe"] == 1 and counts["Unsafe"] == 1
    back = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in back] == ["a", "b"]
    assert back[0]["extra"] == {"keep": "me"}
    assert back[0]["evidence"]["question"] == "Q?"


def test_a_row_with_no_verdict_reads_as_unreadable_not_safe(tmp_path):
    """An unannotated item must not be silently counted as a clean bill."""
    out = tmp_path / "guarded.jsonl"
    counts = runner.flush([_row("a")], {}, {}, out)
    assert counts["unreadable"] == 1 and counts["Safe"] == 0


def test_a_failed_parse_is_unreadable_rather_than_safe(tmp_path):
    out = tmp_path / "guarded.jsonl"
    counts = runner.flush([_row("a")], {}, {"a": _verdict("I cannot help with that.")}, out)
    assert counts["unreadable"] == 1 and counts["Safe"] == 0


def test_flush_leaves_no_temp_file_behind(tmp_path):
    out = tmp_path / "guarded.jsonl"
    runner.flush([_row("a")], {}, {"a": _verdict(FULL_SAFE)}, out)
    assert [p.name for p in tmp_path.iterdir()] == ["guarded.jsonl"]


def test_a_flushed_file_resumes(tmp_path):
    """What a periodic write leaves behind has to be readable as `done` next time."""
    from msrc.signals import GuardModel

    out = tmp_path / "guarded.jsonl"
    v = {"a": _verdict(FULL_SAFE), "b": _verdict(FULL_UNSAFE)}
    runner.flush([_row("a"), _row("b")], {}, v, out)

    done = {}
    for row in runner.load(out):
        vs = (row.get("evidence") or {}).get("guard_verdicts") or []
        if vs and GuardModel.grade(vs[0].get("text", "")) is not None:
            done[row["id"]] = row
    assert set(done) == {"a", "b"}


def test_an_already_done_row_is_carried_over_verbatim(tmp_path):
    """Resume has to keep the earlier annotation, not re-ask and not blank it."""
    out = tmp_path / "guarded.jsonl"
    prior = _row("a")
    prior["evidence"]["guard_verdicts"] = [_verdict(FULL_CONTRO)]
    counts = runner.flush([_row("a"), _row("b")], {"a": prior},
                          {"b": _verdict(FULL_SAFE)}, out)
    assert counts["Controversial"] == 1 and counts["Safe"] == 1


