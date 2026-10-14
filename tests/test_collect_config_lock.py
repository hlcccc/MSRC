# -*- coding: utf-8 -*-
"""Resuming a collection must not mix generation protocols in one file.

The resume key identifies the *item*, so before this check a changed
``--max-new-tokens`` (or model path, or ``k``) kept the rows already collected and
appended new ones produced under the new setting. One cell ended up holding 460
answers capped at 64 tokens and 740 at 128, and the only way to notice was to
tokenise every answer and look at where the lengths piled up.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_evidence_configlock_under_test",
        ROOT / "scripts" / "collect_evidence.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_collector()


def _args(**over):
    base = dict(provider="llava", model_path="/models/llava", max_new_tokens=128,
                k=3, seed=20260920, dtype="", want_attention=False,
                want_image_contrast=False, max_pixels=0, second_provider=None,
                second_model_path="", no_ocr=False, answer_parser="legacy",
                answer_format="")
    base.update(over)
    return Namespace(**base)


def _evidence_row():
    return {"id": "x", "image": "a.jpg", "question": "q", "answer": "a",
            "evidence": {"question": "q"}}


# --------------------------------------------------------------- fingerprint
def test_fingerprint_carries_the_settings_that_change_a_reading():
    fp = collector.collection_fingerprint(_args())
    for key in ("model_path", "max_new_tokens", "k", "want_attention"):
        assert key in fp, key
    assert fp["max_new_tokens"] == 128


def test_fingerprint_ignores_how_the_run_is_executed():
    """Paths and batch sizes change the run, not what a reading means."""
    fp = collector.collection_fingerprint(_args())
    for key in ("out", "data", "batch_size", "workers", "gpu_memory_fraction"):
        assert key not in fp, key


def test_changing_a_generation_setting_changes_the_fingerprint():
    a = collector.collection_fingerprint(_args())
    b = collector.collection_fingerprint(_args(max_new_tokens=64))
    assert a != b


# ------------------------------------------------------------ resume manifest
def test_a_matching_configuration_resumes(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps(_evidence_row()) + "\n", encoding="utf-8")
    fp = collector.collection_fingerprint(_args())
    collector.check_resume_manifest(out, fp)          # first run writes the manifest
    collector.check_resume_manifest(out, fp)          # second run agrees with it


def test_a_changed_configuration_is_refused(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps(_evidence_row()) + "\n", encoding="utf-8")
    collector.check_resume_manifest(out, collector.collection_fingerprint(_args()))

    with pytest.raises(SystemExit) as exc:
        collector.check_resume_manifest(
            out, collector.collection_fingerprint(_args(max_new_tokens=64)))
    message = str(exc.value)
    assert "different configuration" in message
    assert "max_new_tokens" in message, "the message must name the setting that moved"


def test_an_output_without_a_manifest_warns_but_proceeds(tmp_path, capsys):
    """Every file collected before this check exists is in that state."""
    out = tmp_path / "legacy.jsonl"
    out.write_text(json.dumps(_evidence_row()) + "\n", encoding="utf-8")

    collector.check_resume_manifest(out, collector.collection_fingerprint(_args()))

    assert "cannot be verified" in capsys.readouterr().out
    assert (tmp_path / "legacy.jsonl.manifest.json").is_file()


def test_a_fresh_output_does_not_warn(tmp_path, capsys):
    out = tmp_path / "new.jsonl"
    collector.check_resume_manifest(out, collector.collection_fingerprint(_args()))
    assert capsys.readouterr().out == ""


def test_the_manifest_records_the_fingerprint(tmp_path):
    out = tmp_path / "out.jsonl"
    fp = collector.collection_fingerprint(_args(max_new_tokens=96))
    collector.check_resume_manifest(out, fp)
    recorded = json.loads((tmp_path / "out.jsonl.manifest.json").read_text("utf-8"))
    assert recorded["fingerprint"] == fp
