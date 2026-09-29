"""The command line: the declared entry point, and the errors it must not make.

`pyproject.toml` declares `msrc = "msrc.cli:main"`. If that module is missing or
its commands raise, an install produces a broken command -- which is the defect
class this test file exists to catch, having already appeared once in a sibling
project where the documented command did not exist after the documented install.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msrc import MSRCConfig, MSRCPipeline, Sample, StubProvider  # noqa: E402
from msrc.cli import VERSION, build_parser, main  # noqa: E402
from msrc.provider import gather_evidence  # noqa: E402


# ---------------------------------------------------------------------------
# The declared entry point really exists
# ---------------------------------------------------------------------------

def test_pyproject_entry_point_resolves_to_a_callable():
    """`msrc = "msrc.cli:main"` must actually import and be callable.

    Read as text rather than parsed with tomllib: tomllib is 3.11+, and this
    package declares 3.9. Parsing the file here would have made the test itself
    the thing that fails on the oldest supported interpreter.
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "[project.scripts]" in text
    assert 'msrc = "msrc.cli:main"' in text

    target = "msrc.cli:main"
    module_name, _, attr = target.partition(":")
    module = __import__(module_name, fromlist=[attr])
    assert callable(getattr(module, attr))


def test_every_declared_subcommand_has_a_handler():
    """A subcommand in the parser with no handler fails only when someone runs it."""
    parser = build_parser()
    choices = set(parser._subparsers._group_actions[0].choices)
    assert {"signals", "demo", "explain", "fit", "score", "evaluate", "version"} <= choices


def test_version_prints(capsys):
    assert main(["version"]) == 0
    assert VERSION in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def test_signals_reports_the_single_model_count(capsys):
    assert main(["signals"]) == 0
    out = capsys.readouterr().out
    assert "defined : 9" in out
    assert "apply   : 8" in out
    assert "run     : 7" in out
    assert "3 internal + 4 external" in out
    assert "needs a second model" in out


def test_signals_reports_eight_with_a_second_model(capsys):
    assert main(["signals", "--second-model"]) == 0
    out = capsys.readouterr().out
    assert "run     : 8" in out
    assert "needs a second model" not in out


def test_safety_family_says_it_has_no_taxonomy(capsys):
    """The safety family must not imply a policy it does not have."""
    assert main(["signals", "--family", "safety"]) == 0
    out = capsys.readouterr().out
    assert "no policy taxonomy" in out
    assert "policy_probe" in out


# ---------------------------------------------------------------------------
# Errors: no tracebacks, and the message says what to do
# ---------------------------------------------------------------------------

def test_a_missing_file_is_reported_not_raised(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["explain", "--data", str(tmp_path / "nope.jsonl")])
    message = str(excinfo.value)
    assert "not found" in message
    assert "docs/evaluation.md" in message, "the message should point at the format"


def test_malformed_jsonl_names_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["explain", "--data", str(path)])
    assert ":2" in str(excinfo.value)


def test_a_utf8_bom_is_tolerated(tmp_path):
    """PowerShell's Out-File and Excel both prepend one."""
    path = tmp_path / "bom.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n", encoding="utf-8-sig")
    assert path.read_bytes()[:3] == b"\xef\xbb\xbf"
    # _load_jsonl is what every command uses; reaching the evidence check means
    # the BOM was read correctly.
    with pytest.raises(SystemExit, match="no evidence"):
        main(["explain", "--data", str(path)])


def test_records_without_evidence_are_refused(tmp_path):
    path = tmp_path / "noev.jsonl"
    path.write_text(
        json.dumps({"id": "x", "question": "q", "answer": "a", "label": 0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        main(["explain", "--data", str(path)])
    assert "collect_evidence" in str(excinfo.value)


def test_a_single_class_dataset_is_refused(tmp_path):
    """A calibrator fitted on one class is a constant; say so instead of fitting."""
    rows = []
    provider = StubProvider(ocr_text="Flickr")
    for i in range(6):
        evidence, _ = gather_evidence(provider, "q", "Flickr", f"/i{i}.jpg", k=2)
        rows.append({"id": f"i{i}", "image": f"/i{i}.jpg", "label": 0,
                     "evidence": evidence.to_dict()})
    path = tmp_path / "oneclass.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["explain", "--data", str(path)])
    assert "one class" in str(excinfo.value)


def test_scoring_an_unfitted_scorer_is_refused(tmp_path):
    """Returning an arbitrary number from an unfitted scorer is the failure this
    check exists to prevent."""
    scorer = tmp_path / "empty.json"
    scorer.write_text(json.dumps({"format": "msrc-scorer", "config": {}, "calibrator": None}),
                      encoding="utf-8")
    evidence = tmp_path / "ev.json"
    evidence.write_text(json.dumps({"primary": {"text": "a"}}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["score", "--scorer", str(scorer), "--evidence", str(evidence)])
    assert "no fitted coefficients" in str(excinfo.value)


def test_a_missing_scorer_is_reported(tmp_path):
    evidence = tmp_path / "ev.json"
    evidence.write_text(json.dumps({"primary": {"text": "a"}}), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["score", "--scorer", str(tmp_path / "nope.json"), "--evidence", str(evidence)])
    assert "scorer not found" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Fit and score round trip
# ---------------------------------------------------------------------------

def _write_evidence(path: Path, n: int = 40) -> Path:
    provider = StubProvider(ocr_text="Flickr")
    rows = []
    for i in range(n):
        failed = i % 2
        answer = "Flickr" + (" WRONG" if failed else "")
        primary = Sample(text=answer,
                         sequence_confidence=0.34 if failed else 0.88,
                         sequence_entropy=2.8 if failed else 0.5,
                         visual_attention_mass=0.12 if failed else 0.62)
        evidence, _ = gather_evidence(provider, "what is the website?", answer,
                                      f"/img{i % 20}.jpg", k=3, primary_sample=primary)
        rows.append({"id": f"i{i}", "question": "q", "answer": answer,
                     "image": f"/img{i % 20}.jpg", "label": failed,
                     "evidence": evidence.to_dict()})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_fit_writes_a_scorer_and_records_provenance(tmp_path, capsys):
    data = _write_evidence(tmp_path / "ev.jsonl")
    out = tmp_path / "scorer.json"
    assert main(["fit", "--data", str(data), "--out", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == "msrc-scorer"
    assert payload["provenance"]["n_signals"] == 8
    assert "scorer written" in capsys.readouterr().out


def test_score_uses_the_fitted_scorer(tmp_path, capsys):
    data = _write_evidence(tmp_path / "ev.jsonl")
    scorer = tmp_path / "scorer.json"
    main(["fit", "--data", str(data), "--out", str(scorer)])
    capsys.readouterr()  # drop the fit output, so what follows is only the score

    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    evidence = tmp_path / "one.json"
    evidence.write_text(json.dumps({"evidence": rows[0]["evidence"]}), encoding="utf-8")

    assert main(["score", "--scorer", str(scorer), "--evidence", str(evidence), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["calibrated_confidence"] == pytest.approx(1.0 - result["risk_score"], abs=1e-6)
    assert len(result["signals"]["signals"]) == 8
