"""Tests for the first-step decision margin.

The reading exists because the mean token probability is diluted by boilerplate:
on a yes/no benchmark one hard decision is averaged together with twenty easy
tokens, so the mean stays high on exactly the items the model guessed at. What has
to hold here is that the margin reads only the first step and the runner-up, and
that its direction is the one the framework expects.
"""

from __future__ import annotations

import pytest

from msrc.signals import ALL_SIGNALS, DecisionMargin, Evidence, build_report
from msrc.types import SIGNAL_INTERNAL, Sample


def ev(margin, confidence=None, text="Yes"):
    return Evidence(
        question="q",
        answer=text,
        primary=Sample(text=text, sequence_confidence=confidence,
                       first_token_margin=margin),
    )


def test_it_is_an_internal_signal():
    assert DecisionMargin().kind == SIGNAL_INTERNAL
    assert DecisionMargin().name == "decision_margin"


def test_a_confident_commitment_is_low_risk():
    v = DecisionMargin().compute(ev(0.98))
    assert v.available
    assert v.risk == pytest.approx(0.02)
    assert v.raw == pytest.approx(0.98)


def test_a_torn_commitment_is_high_risk():
    v = DecisionMargin().compute(ev(0.02))
    assert v.available
    assert v.risk == pytest.approx(0.98)


def test_risk_falls_as_the_margin_grows():
    risks = [DecisionMargin().compute(ev(m)).risk for m in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert risks == sorted(risks, reverse=True)


def test_missing_scores_are_unavailable_not_zero():
    """A placeholder would make the column constant and the calibrator meaningless."""
    v = DecisionMargin().compute(ev(None, confidence=0.9))
    assert not v.available
    assert v.risk == 0.0
    assert "per-step scores" in v.detail


def test_it_reads_something_the_mean_confidence_does_not():
    """Same mean confidence, opposite margins: the two readings must disagree."""
    certain = DecisionMargin().compute(ev(0.95, confidence=0.90)).risk
    torn = DecisionMargin().compute(ev(0.05, confidence=0.90)).risk
    assert certain < 0.1 < 0.9 < torn


def test_out_of_range_input_is_clamped():
    assert DecisionMargin().compute(ev(1.7)).risk == pytest.approx(0.0)
    assert DecisionMargin().compute(ev(-0.4)).risk == pytest.approx(1.0)


def test_the_registry_includes_it_before_visual_attention():
    names = [s.name for s in ALL_SIGNALS]
    assert names.index("decision_margin") < names.index("visual_attention")


def test_a_round_trip_through_json_keeps_the_margin():
    """Evidence is cached as JSONL, so the field has to survive serialisation."""
    original = Evidence(
        question="q", answer="Yes",
        primary=Sample(text="Yes", sequence_confidence=0.4, sequence_entropy=1.2,
                       visual_attention_mass=0.8, first_token_margin=0.13),
    )
    back = Evidence.from_dict(original.to_dict())
    assert back.primary.first_token_margin == pytest.approx(0.13)
    assert back.primary.sequence_confidence == pytest.approx(0.4)


def test_older_evidence_without_the_field_still_loads():
    """Every cached file written before this signal existed lacks the key."""
    payload = {
        "question": "q", "answer": "Yes",
        "primary": {"text": "Yes", "sequence_confidence": 0.4},
    }
    ev_back = Evidence.from_dict(payload)
    assert ev_back.primary.first_token_margin is None
    report = build_report(ev_back)
    assert "decision_margin" in report.missing()
