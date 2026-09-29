"""The HTTP service: what it refuses to do, and what it answers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fastapi = pytest.importorskip("fastapi", reason="the service needs the service extra")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from msrc.conformal import VALIDATED  # noqa: E402
from msrc.pipeline import MSRCConfig, MSRCPipeline  # noqa: E402
from msrc.provider import gather_evidence  # noqa: E402
from msrc.service import create_app  # noqa: E402
from msrc.types import Sample  # noqa: E402


def _evidence(failed: int = 0):
    from msrc import StubProvider

    provider = StubProvider(ocr_text="Flickr")
    answer = "Flickr" + (" WRONG" if failed else "")
    primary = Sample(text=answer,
                     sequence_confidence=0.34 if failed else 0.88,
                     sequence_entropy=2.8 if failed else 0.5,
                     visual_attention_mass=0.12 if failed else 0.62)
    evidence, _ = gather_evidence(provider, "what is the website?", answer,
                                  "/img.jpg", k=3, primary_sample=primary)
    return evidence.to_dict()


def _fitted_pipeline() -> MSRCPipeline:
    import numpy as np

    pipeline = MSRCPipeline(config=MSRCConfig(k=3))
    rows = [_evidence(i % 2) for i in range(40)]
    from msrc.signals import Evidence, build_report

    names: list = []
    matrix, labels = [], []
    for row, label in zip(rows, [i % 2 for i in range(40)]):
        report = build_report(Evidence.from_dict(row))
        if not names:
            names = report.names()
        matrix.append(report.vector())
        labels.append(label)
    pipeline.fit_from_signals(np.asarray(matrix, float), np.asarray(labels, int), names)
    return pipeline


@pytest.fixture()
def client():
    return TestClient(create_app(_fitted_pipeline()))


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------

def test_an_unfitted_pipeline_is_refused_at_construction():
    """A 200 with an uncalibrated prior is the failure this framework exists to
    prevent, and an endpoint is where it would do the most damage."""
    with pytest.raises(ValueError, match="no fitted coefficients"):
        create_app(MSRCPipeline(config=MSRCConfig()))


def test_no_pipeline_at_all_is_refused():
    with pytest.raises(ValueError, match="needs a fitted pipeline"):
        create_app(None)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_reports_the_scorer_and_the_security_boundary(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["scorer_fitted"] is True
    assert body["n_signals"] == 8
    assert body["conformal_validated"] == VALIDATED
    assert "gateway" in body["auth"], (
        "the service has no auth; the health payload should say so rather than "
        "leaving it to a docstring"
    )


def test_health_carries_provenance(client):
    """A caller should be able to see what the served scorer was fitted on."""
    body = client.get("/health").json()
    assert "provenance" in body
    assert body["provenance"]["n_signals"] == 8


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_risk_returns_the_platform_contract(client):
    response = client.post("/v1/msrc/risk", json={"evidence": _evidence(0)})
    assert response.status_code == 200
    body = response.json()
    for key in ("risk_score", "calibrated_confidence", "is_high_risk", "threshold",
                "risk_family", "signals", "missing_signals", "warnings", "version"):
        assert key in body, key
    assert 0.0 <= body["risk_score"] <= 1.0


def test_confidence_is_the_complement_of_the_score(client):
    body = client.post("/v1/msrc/risk", json={"evidence": _evidence(0)}).json()
    assert body["calibrated_confidence"] == pytest.approx(1.0 - body["risk_score"], abs=1e-6)


def test_a_wrong_answer_scores_higher_than_a_right_one(client):
    good = client.post("/v1/msrc/risk", json={"evidence": _evidence(0)}).json()
    bad = client.post("/v1/msrc/risk", json={"evidence": _evidence(1)}).json()
    assert good["risk_score"] < bad["risk_score"]


def test_the_response_names_the_missing_signals(client):
    """A caller must be able to see which channels their deployment cannot feed."""
    body = client.post("/v1/msrc/risk", json={"evidence": _evidence(0)}).json()
    if body["n_signals_missing"]:
        assert body["missing_signals"], "a count without names is not actionable"


def test_an_empty_evidence_object_is_a_400(client):
    response = client.post("/v1/msrc/risk", json={"evidence": {}})
    assert response.status_code == 400


def test_unreadable_evidence_is_a_400_not_a_500(client):
    """A malformed payload is the caller's problem and the message should say so.

    The deserialiser validates at the boundary, so a field of the wrong type
    arrives as a ValueError naming the field rather than as an AttributeError
    from whichever helper touches it first.
    """
    response = client.post("/v1/msrc/risk", json={"evidence": {"views": "not a list"}})
    assert response.status_code == 400
    assert "views" in response.json()["detail"]
    assert "list" in response.json()["detail"]


def test_evidence_that_is_not_an_object_is_a_400(client):
    response = client.post("/v1/msrc/risk", json={"evidence": {"primary": "a string"}})
    assert response.status_code == 400
    assert "primary" in response.json()["detail"] or "object" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def test_signals_lists_what_applies_to_the_served_family(client):
    body = client.get("/v1/msrc/signals").json()
    names = {s["name"] for s in body}
    assert "grounding_check" in names, "the factual family's own signal"
    assert "policy_probe" not in names, "which belongs to the other family"
    assert all(s["kind"] in ("internal", "external") for s in body)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_select_runs_the_procedure(client):
    response = client.post("/v1/msrc/select", json={
        "calibration_null_scores": [0.05 * (i + 1) for i in range(20)],
        "test_scores": [0.95, 0.9, 0.05],
        "test_labels": [1, 1, 0],
        "alpha": 0.3,
        "procedure": "BH",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["validated"] is VALIDATED
    assert body["num_items"] == 3
    assert "accepted_mask" in body


def test_select_refuses_an_unknown_procedure(client):
    response = client.post("/v1/msrc/select", json={
        "calibration_null_scores": [0.1, 0.2],
        "test_scores": [0.3],
        "procedure": "holm",
    })
    assert response.status_code == 400
    assert "procedure" in response.json()["detail"]


def test_select_refuses_an_empty_calibration_set(client):
    response = client.post("/v1/msrc/select", json={
        "calibration_null_scores": [], "test_scores": [0.5],
    })
    assert response.status_code == 400


def test_select_refuses_an_out_of_range_alpha(client):
    response = client.post("/v1/msrc/select", json={
        "calibration_null_scores": [0.1, 0.2], "test_scores": [0.3], "alpha": 2.0,
    })
    assert response.status_code == 400
