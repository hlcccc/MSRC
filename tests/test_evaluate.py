"""The evaluation report, which has to describe the right risk family.

The script used to print one fixed closing paragraph, so pointing it at safety
evidence produced a report that said, in as many words, that the number was not a
safety judgement. The family now comes from the evidence, and a file that mixes
two families is refused rather than averaged: a factual label and a safety label
are not the same quantity, and 1 does not even mean the same kind of thing in the
two cases.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msrc.provider import StubProvider, gather_evidence  # noqa: E402
from msrc.types import RISK_FACTUAL, RISK_SAFETY, Sample  # noqa: E402


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = _load("evaluate_under_test", "scripts/evaluate.py")


def _evidence_rows(family: str, n: int = 40, *, with_verdicts: bool = False):
    """Alternating safe/risky items, with internals that track the label."""
    provider = StubProvider(ocr_text="Flickr")
    rows = []
    for i in range(n):
        risky = i % 2
        primary_text = "Flickr" + (" WRONG" if risky else "")
        primary = Sample(
            text=primary_text,
            sequence_confidence=0.34 if risky else 0.88,
            sequence_entropy=2.8 if risky else 0.5,
            visual_attention_mass=0.12 if risky else 0.62,
        )
        evidence, _ = gather_evidence(
            provider, "q", primary_text, f"/img{i % 20}.jpg",
            risk_family=family, k=3, primary_sample=primary,
        )
        row = {
            "id": f"i{i}", "question": "q", "answer": primary_text,
            "image": f"/img{i % 20}.jpg", "label": risky,
            "evidence": evidence.to_dict(),
        }
        if with_verdicts:
            row["refusal"] = {
                "refused": not risky, "certain": i % 10 != 0, "groups": [],
                "compliance_hits": 0,
            }
        rows.append(row)
    return rows


def _write(path: Path, rows) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def test_the_factual_family_is_reported_as_factual(tmp_path, capsys):
    data = _write(tmp_path / "ev.jsonl", _evidence_rows(RISK_FACTUAL))
    assert evaluate.main(["--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert f"risk family      : {RISK_FACTUAL}" in out
    assert "内容安全" in out, "the factual report says what it is not"


def test_the_safety_family_is_reported_as_safety(tmp_path, capsys):
    data = _write(
        tmp_path / "ev.jsonl", _evidence_rows(RISK_SAFETY, with_verdicts=True)
    )
    assert evaluate.main(["--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert f"risk family      : {RISK_SAFETY}" in out
    assert "不安全响应" in out
    assert "拒绝判定不确定" in out, "the label rule's uncertainty is part of the result"
    assert "事实性风险（回答与图像矛盾）" not in out, "must not describe the other family"


def test_the_safety_report_computes_the_label_uncertainty_rate(tmp_path, capsys):
    rows = _evidence_rows(RISK_SAFETY, n=40, with_verdicts=True)
    data = _write(tmp_path / "ev.jsonl", rows)
    assert evaluate.main(["--data", str(data), "--json"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])

    expected = sum(1 for r in rows if not r["refusal"]["certain"]) / len(rows)
    assert payload["label_uncertain_rate"] == pytest.approx(expected, abs=1e-9)
    assert payload["risk_family"] == RISK_SAFETY


def test_mixed_families_are_refused(tmp_path, capsys):
    rows = _evidence_rows(RISK_FACTUAL, n=20) + _evidence_rows(RISK_SAFETY, n=20)
    data = _write(tmp_path / "ev.jsonl", rows)
    with pytest.raises(SystemExit, match="mixes risk families"):
        evaluate.main(["--data", str(data)])


def test_an_unknown_family_is_refused(tmp_path):
    rows = _evidence_rows(RISK_FACTUAL, n=20)
    for row in rows:
        row["evidence"]["risk_family"] = "made-up"
    data = _write(tmp_path / "ev.jsonl", rows)
    with pytest.raises(SystemExit, match="unknown risk family"):
        evaluate.main(["--data", str(data)])


def test_the_report_always_prints_auroc_beside_ece(tmp_path, capsys):
    """ECE alone is gameable: a constant scores 0 while ranking nothing."""
    data = _write(tmp_path / "ev.jsonl", _evidence_rows(RISK_FACTUAL))
    assert evaluate.main(["--data", str(data), "--json"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    for key in ("auroc", "brier", "ece_after", "ece_before", "n_bins", "threshold_metrics"):
        assert key in payload, key
    assert "AUROC" in out


def test_the_baseline_is_described_as_what_it_actually_is(tmp_path, capsys):
    """The first real run called an internal signal "the strongest external one".

    The baseline is deliberately the strongest signal of any kind, so it is often
    an internal one; the label has to follow the signal rather than the prose that
    was written when the baseline happened to be external.
    """
    data = _write(tmp_path / "ev.jsonl", _evidence_rows(RISK_FACTUAL))
    assert evaluate.main(["--data", str(data), "--json"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])

    name, kind = payload["baseline_signal"], payload["baseline_kind"]
    assert kind in ("internal", "external")
    assert f"（{kind}，" in out, "the printed baseline names its kind"
    assert f"{name!r}（{kind}" in out
    internal = {"sequence_confidence", "output_entropy", "visual_attention"}
    assert kind == ("internal" if name in internal else "external")


def test_a_file_without_evidence_says_so(tmp_path):
    data = _write(tmp_path / "ev.jsonl", [{"id": "i0", "label": 0}])
    with pytest.raises(SystemExit, match="no records with evidence"):
        evaluate.main(["--data", str(data)])
