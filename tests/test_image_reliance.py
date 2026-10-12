"""Tests for the image-reliance signal.

The reading answers one question: would this answer have been the same if the model
had never seen the picture? Low divergence means yes, and that is the hallucination
this exists to catch -- the model answering from its prior about what such a picture
usually contains.

Two things have to hold: the direction, and that a provider which cannot run the
ablated pass reports the reading missing rather than substituting a number.
"""

from __future__ import annotations

import pytest

from msrc.signals import ALL_SIGNALS, Evidence, ImageReliance, build_report
from msrc.types import SIGNAL_INTERNAL, Sample


def ev(contrast, text="Yes"):
    return Evidence(question="q", answer=text,
                    primary=Sample(text=text, image_contrast=contrast))


def test_it_is_an_internal_signal():
    assert ImageReliance().kind == SIGNAL_INTERNAL
    assert ImageReliance().name == "image_reliance"


def test_an_answer_that_did_not_need_the_image_is_high_risk():
    v = ImageReliance().compute(ev(0.0))
    assert v.available
    assert v.risk == pytest.approx(1.0)


def test_an_answer_that_depended_on_the_image_is_low_risk():
    v = ImageReliance().compute(ev(0.9))
    assert v.available
    assert v.risk == pytest.approx(0.1)


def test_risk_falls_as_the_image_matters_more():
    risks = [ImageReliance().compute(ev(c)).risk for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert risks == sorted(risks, reverse=True)


def test_missing_reading_is_unavailable_not_zero():
    """A placeholder would make the column constant and the calibrator meaningless."""
    v = ImageReliance().compute(ev(None))
    assert not v.available
    assert v.risk == 0.0
    assert "ablated" in v.detail


def test_out_of_range_input_is_clamped():
    assert ImageReliance().compute(ev(1.4)).risk == pytest.approx(0.0)
    assert ImageReliance().compute(ev(-0.3)).risk == pytest.approx(1.0)


def test_the_registry_includes_it_before_visual_attention():
    names = [s.name for s in ALL_SIGNALS]
    assert names.index("image_reliance") == names.index("visual_attention") - 1


def test_a_round_trip_through_json_keeps_the_contrast():
    """Evidence is cached as JSONL, so the field has to survive serialisation."""
    original = Evidence(question="q", answer="Yes",
                        primary=Sample(text="Yes", image_contrast=0.42))
    back = Evidence.from_dict(original.to_dict())
    assert back.primary.image_contrast == pytest.approx(0.42)


def test_evidence_collected_before_the_field_existed_still_loads():
    """Every cached file predates it, and must report the signal as missing."""
    payload = {"question": "q", "answer": "Yes", "primary": {"text": "Yes"}}
    back = Evidence.from_dict(payload)
    assert back.primary.image_contrast is None
    assert "image_reliance" in build_report(back).missing()


def test_it_reads_something_the_mean_confidence_does_not():
    """Two answers equally confident, one of which never looked at the picture."""
    looked = Evidence(question="q", answer="Yes",
                      primary=Sample(text="Yes", sequence_confidence=0.9,
                                     image_contrast=0.8))
    guessed = Evidence(question="q", answer="Yes",
                       primary=Sample(text="Yes", sequence_confidence=0.9,
                                      image_contrast=0.02))
    a = {s.name: s for s in build_report(looked).signals}
    b = {s.name: s for s in build_report(guessed).signals}
    assert a["sequence_confidence"].risk == pytest.approx(b["sequence_confidence"].risk)
    assert b["image_reliance"].risk > 0.9 > a["image_reliance"].risk
