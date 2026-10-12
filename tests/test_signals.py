"""Signal layer: orientation, missing-not-zero, and the helper maths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msrc.signals import (  # noqa: E402
    ALL_SIGNALS,
    CrossModelAgreement,
    CrossModelObjectSupport,
    Evidence,
    GroundingCheck,
    ObjectSupport,
    OutputEntropy,
    ResampleConsistency,
    SelfConsistency,
    SequenceConfidence,
    TypedVerification,
    VisualAttention,
    agreement,
    build_report,
    cluster_entropy,
    normalize_text,
)
from msrc.types import (  # noqa: E402
    SIGNAL_EXTERNAL,
    SIGNAL_INTERNAL,
    Sample,
    SignalValue,
)


# ---------------------------------------------------------------------------
# The declared inventory. Numbers here are the project's "≥N signals" claim.
# ---------------------------------------------------------------------------

def test_the_signal_inventory_is_what_the_readme_claims():
    external = [s for s in ALL_SIGNALS if s.kind == SIGNAL_EXTERNAL]
    internal = [s for s in ALL_SIGNALS if s.kind == SIGNAL_INTERNAL]
    assert len(external) == 7, [s.name for s in external]
    assert len(internal) == 5, [s.name for s in internal]


def test_the_declared_inventory_is_exactly_the_twelve_signals():
    """Named, and in the order that fixes the fitted matrix's columns.

    The order is part of the contract: a fitted scorer's coefficients are applied
    positionally, so a reordering is only visible as a wrong number.
    """
    assert [s.name for s in ALL_SIGNALS] == [
        "self_consistency", "resample_consistency", "typed_verification",
        "cross_model_agreement", "grounding_check", "object_support",
        "cross_model_object_support",
        "sequence_confidence", "output_entropy", "decision_margin",
        "image_reliance", "visual_attention",
    ]


def test_a_second_model_buys_exactly_two_signals():
    """A second VLM is a real deployment cost; what it buys must be countable.

    It buys two, and they read the same channel at different resolutions:
    ``cross_model_agreement`` compares the whole caption, ``cross_model_object_support``
    asks whether the other model also failed to name each object this answer names.

    Built with full external evidence and no internals. The answer names an object
    so that ``object_support`` can read the re-lookings, which keeps the unavailable
    set to the second-model signals plus the three internal ones -- nothing else.
    """
    external_only = dict(
        question="q", answer="a car",
        views=[Sample(text="a car")],
        resamples=[Sample(text="a car"), Sample(text="a car")],
        verifications=[Sample(text="yes")],
        ocr_texts=["a car"],
    )
    evidence = Evidence(primary=Sample(text="a car"), **external_only)
    report = build_report(evidence)
    assert sorted(report.missing()) == [
        "cross_model_agreement", "cross_model_object_support", "decision_margin",
        "image_reliance", "output_entropy", "sequence_confidence", "visual_attention",
    ]
    assert len(report.available()) == 5, "five external signals are live here"

    # Now give the internals, which is the only thing a single model at full
    # access adds -- the second-model signals stay unavailable.
    with_internals = Evidence(
        primary=Sample(text="a car", sequence_confidence=0.9, sequence_entropy=0.3,
                       visual_attention_mass=0.5, first_token_margin=0.8,
                       image_contrast=0.4),
        **external_only,
    )
    live = build_report(with_internals)
    # Five external that need no second model, plus the five internal: ten.
    assert len(live.available()) == 10, [s.name for s in live.available()]
    assert live.missing() == ["cross_model_agreement", "cross_model_object_support"]
    assert sum(1 for s in live.available() if s.kind == SIGNAL_INTERNAL) == 5
    assert any(s.name == "object_support" for s in live.available())


def test_signal_names_are_unique_and_populated():
    """A blank or duplicated name would silently collide in the fitted matrix."""
    names = [s.name for s in ALL_SIGNALS]
    assert all(names), names
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a,b",
    [("Dakota Digital", "dakota digital"), ("the red car", "Red Car!"),
     ("two people", "2 people"), ("  Flickr  ", "flickr")],
)
def test_normalisation_makes_trivial_variants_agree(a, b):
    assert agreement(a, b) == pytest.approx(1.0)


def test_partial_containment_is_high_but_not_perfect():
    """'Dakota' and 'Dakota Digital' are the same answer at different lengths.

    Scoring them as a full disagreement would inject noise into every consistency
    signal; scoring them as identical would hide a real difference.
    """
    score = agreement("Dakota", "Dakota Digital")
    assert 0.8 < score < 1.0


def test_unrelated_answers_do_not_agree():
    assert agreement("Flickr", "Nikon Coolpix") == pytest.approx(0.0)


def test_entropy_is_zero_when_the_model_repeats_itself():
    assert cluster_entropy(["Flickr"] * 5) == pytest.approx(0.0)


def test_entropy_is_one_when_every_sample_differs():
    assert cluster_entropy(["a", "b", "c", "d"]) == pytest.approx(1.0)


def test_entropy_is_normalised_so_k_does_not_change_the_scale():
    """K is a deployment dial; the signal must not drift with it."""
    three = cluster_entropy(["a", "b", "c"])
    five = cluster_entropy(["a", "b", "c", "d", "e"])
    assert three == pytest.approx(1.0)
    assert five == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Missing is not zero
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "signal,field",
    [
        (SequenceConfidence(), "sequence_confidence"),
        (OutputEntropy(), "sequence_entropy"),
        (VisualAttention(), "visual_attention_mass"),
    ],
)
def test_internal_signals_report_unavailable_without_internals(signal, field):
    """A text-only platform is supported, not silently degraded."""
    value = signal.compute(Evidence(question="q", answer="a", primary=Sample(text="a")))
    assert value.available is False
    assert field.split("_")[1] in value.detail or "internal signal unavailable" in value.detail
    assert value.kind == SIGNAL_INTERNAL


def test_cross_model_signal_is_unavailable_without_a_second_model():
    value = CrossModelAgreement().compute(Evidence(question="q", answer="a"))
    assert value.available is False


# ---------------------------------------------------------------------------
# cross_model_object_support
# ---------------------------------------------------------------------------

def test_an_object_the_other_model_also_never_names_is_full_risk():
    """The whole point: a second, independent look that misses the same object."""
    evidence = Evidence(
        question="describe", answer="a cat and a surfboard",
        primary=Sample(text="a cat and a surfboard"),
        cross_model=[Sample(text="a cat on a rug")],
    )
    value = CrossModelObjectSupport().compute(evidence)
    assert value.available is True
    assert value.risk == pytest.approx(1.0)
    assert value.raw == pytest.approx(0.0)
    assert "surfboard" in value.detail


def test_an_object_the_other_model_names_too_is_no_risk():
    evidence = Evidence(
        question="describe", answer="a cat and a surfboard",
        primary=Sample(text="a cat and a surfboard"),
        cross_model=[Sample(text="a cat with a surfboard on a beach")],
    )
    value = CrossModelObjectSupport().compute(evidence)
    assert value.available is True
    assert value.risk == pytest.approx(0.0)


def test_the_weakest_object_decides_even_when_the_others_are_supported():
    """A caption is only as trustworthy as its least supported claim."""
    evidence = Evidence(
        question="describe", answer="a cat and a surfboard",
        primary=Sample(text="a cat and a surfboard"),
        cross_model=[Sample(text="a cat on a rug"), Sample(text="a cat and a surfboard")],
    )
    value = CrossModelObjectSupport().compute(evidence)
    assert value.risk == pytest.approx(0.5)
    assert "surfboard" in value.detail


def test_one_reading_from_another_model_is_enough():
    """MIN_AUXILIARY is a floor on repeating oneself, not on being told by a peer.

    A single other-model description is the usual deployment, and refusing it would
    be refusing the only cross-model object-level reading there is.
    """
    evidence = Evidence(
        question="describe", answer="a cat",
        primary=Sample(text="a cat"),
        cross_model=[Sample(text="a cat")],
    )
    assert CrossModelObjectSupport().compute(evidence).available is True


def test_cross_model_object_support_is_unavailable_without_a_second_model():
    value = CrossModelObjectSupport().compute(Evidence(question="q", answer="a cat"))
    assert value.available is False
    assert "second model" in value.detail


def test_cross_model_object_support_is_unavailable_when_no_object_is_named():
    evidence = Evidence(
        question="is there a cat?", answer="Yes",
        primary=Sample(text="Yes"), cross_model=[Sample(text="Yes")],
    )
    value = CrossModelObjectSupport().compute(evidence)
    assert value.available is False
    assert "lexicon" in value.detail


def test_a_blank_other_model_description_does_not_count_as_a_reading():
    evidence = Evidence(
        question="describe", answer="a cat", primary=Sample(text="a cat"),
        cross_model=[Sample(text="   ")],
    )
    assert CrossModelObjectSupport().compute(evidence).available is False


def test_the_two_second_model_signals_read_the_same_channel_differently():
    """One compares captions, the other compares the objects inside them.

    A caption that agrees about the toilet and the shower while inventing a mirror is
    the case this signal exists for: whole-caption agreement is high and the object
    reading still fires.
    """
    evidence = Evidence(
        question="describe",
        answer="a cat, a dog and a surfboard",
        primary=Sample(text="a cat, a dog and a surfboard"),
        cross_model=[Sample(text="a cat, a dog and a boat")],
    )
    assert CrossModelAgreement().compute(evidence).risk < 1.0
    assert CrossModelObjectSupport().compute(evidence).risk == pytest.approx(1.0)


def test_the_two_object_signals_disagree_when_the_model_repeats_itself():
    """The failure mode object_support alone cannot see: a consistent mistake."""
    evidence = Evidence(
        question="describe", answer="a cat and a surfboard",
        primary=Sample(text="a cat and a surfboard"),
        views=[Sample(text="a cat and a surfboard")] * 2,
        resamples=[Sample(text="a cat and a surfboard")],
        cross_model=[Sample(text="a cat and a boat")],
    )
    assert ObjectSupport().compute(evidence).risk == pytest.approx(0.0)
    assert CrossModelObjectSupport().compute(evidence).risk == pytest.approx(1.0)


def test_unavailable_signals_enter_the_vector_as_zero_but_are_named():
    """External evidence present, internals and the second model absent.

    The ones that cannot run must say so by name. The alternative -- filling them
    with a neutral value -- produces a constant column and a calibrator that looks
    healthy while encoding nothing.

    ``object_support`` is among the missing here on purpose: the answer is the word
    "Flickr", which names no object in the lexicon. That is the designed outcome, and
    it takes ``cross_model_object_support`` with it -- both read the same lexicon.
    """
    evidence = Evidence(
        question="q", answer="Flickr", primary=Sample(text="Flickr"),
        views=[Sample(text="Flickr")],
        resamples=[Sample(text="Flickr"), Sample(text="Nikon")],
        verifications=[Sample(text="yes")],
        ocr_texts=["Flickr"],
    )
    report = build_report(evidence)
    assert sorted(report.missing()) == [
        "cross_model_agreement", "cross_model_object_support", "decision_margin",
        "image_reliance", "object_support", "output_entropy", "sequence_confidence",
        "visual_attention",
    ], report.missing()
    assert len(report.available()) == 4
    assert len(report.vector()) == len(report.signals) == 12


def test_every_signal_missing_when_there_is_no_evidence_at_all():
    report = build_report(Evidence(question="q", answer="a", primary=Sample(text="a")))
    assert len(report.missing()) == len(report.signals) == 12
    assert report.available() == []


# ---------------------------------------------------------------------------
# Orientation: higher must always mean riskier
# ---------------------------------------------------------------------------

def test_self_consistency_risk_rises_when_the_model_drifts():
    steady = Evidence(
        question="q", answer="Flickr",
        views=[Sample(text="Flickr"), Sample(text="Flickr"), Sample(text="Flickr")],
    )
    drifting = Evidence(
        question="q", answer="Flickr",
        views=[Sample(text="Pinterest"), Sample(text="Tumblr"), Sample(text="Nikon")],
    )
    signal = SelfConsistency()
    assert signal.compute(steady).risk == pytest.approx(0.0)
    assert signal.compute(drifting).risk > 0.8


def test_typed_verification_risk_rises_when_probes_say_no():
    signal = TypedVerification()
    supported = Evidence(question="q", answer="a",
                         verifications=[Sample(text="Yes"), Sample(text="yes")])
    denied = Evidence(question="q", answer="a",
                      verifications=[Sample(text="No"), Sample(text="unsupported")])
    assert signal.compute(supported).risk == pytest.approx(0.0)
    assert signal.compute(denied).risk == pytest.approx(1.0)


def test_grounding_risk_falls_when_the_answer_is_in_the_image_text():
    signal = GroundingCheck()
    found = Evidence(question="q", answer="Flickr", ocr_texts=["Flickr; open 24 days"])
    absent = Evidence(question="q", answer="Nikon", ocr_texts=["Flickr; open 24 days"])
    assert signal.compute(found).risk == pytest.approx(0.0)
    assert signal.compute(absent).risk == pytest.approx(1.0)


def test_sequence_confidence_risk_is_the_complement():
    signal = SequenceConfidence()
    sure = Evidence(question="q", answer="a", primary=Sample(text="a", sequence_confidence=0.95))
    unsure = Evidence(question="q", answer="a", primary=Sample(text="a", sequence_confidence=0.30))
    assert signal.compute(sure).risk == pytest.approx(0.05)
    assert signal.compute(unsure).risk == pytest.approx(0.70)


def test_output_entropy_and_confidence_are_separate_signals():
    """The reason both are kept: they can disagree on the same generation."""
    evidence = Evidence(
        question="q", answer="a",
        primary=Sample(text="a", sequence_confidence=0.99, sequence_entropy=3.5),
    )
    assert SequenceConfidence().compute(evidence).risk == pytest.approx(0.01)
    assert OutputEntropy().compute(evidence).risk > 0.8


def test_visual_attention_risk_rises_when_the_model_ignores_the_image():
    signal = VisualAttention()
    looked = Evidence(question="q", answer="a",
                      primary=Sample(text="a", visual_attention_mass=0.7))
    ignored = Evidence(question="q", answer="a",
                       primary=Sample(text="a", visual_attention_mass=0.02))
    assert signal.compute(looked).risk == pytest.approx(0.3)
    assert signal.compute(ignored).risk == pytest.approx(0.98)


def test_resampling_risk_rises_when_samples_disagree():
    signal = ResampleConsistency()
    agreed = Evidence(question="q", answer="a",
                      resamples=[Sample(text="Flickr")] * 4)
    disagreed = Evidence(question="q", answer="a",
                         resamples=[Sample(text=t) for t in ("a", "b", "c", "d")])
    assert signal.compute(agreed).risk == pytest.approx(0.0)
    assert signal.compute(disagreed).risk == pytest.approx(1.0)


def test_resample_needs_at_least_two_samples():
    value = ResampleConsistency().compute(
        Evidence(question="q", answer="a", resamples=[Sample(text="a")])
    )
    assert value.available is False


# ---------------------------------------------------------------------------
# The invariant itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signal", ALL_SIGNALS, ids=lambda s: s.name)
def test_every_signal_returns_a_risk_in_the_unit_interval(signal):
    """Every signal, on every branch, must satisfy the contract."""
    rich = Evidence(
        question="q", answer="Flickr",
        primary=Sample(text="Flickr", sequence_confidence=0.9, sequence_entropy=0.3,
                       visual_attention_mass=0.5),
        views=[Sample(text="Flickr")],
        resamples=[Sample(text="Flickr"), Sample(text="Nikon")],
        verifications=[Sample(text="yes")],
        cross_model=[Sample(text="Flickr")],
        ocr_texts=["Flickr"],
    )
    for evidence in (rich, Evidence(question="q", answer="a")):
        value = signal.compute(evidence)
        assert isinstance(value, SignalValue)
        assert 0.0 <= value.risk <= 1.0, (signal.name, value.risk)
        assert value.kind in (SIGNAL_EXTERNAL, SIGNAL_INTERNAL)


def test_signal_value_rejects_out_of_range_risk():
    with pytest.raises(ValueError):
        SignalValue(name="x", kind=SIGNAL_EXTERNAL, risk=1.5)
    with pytest.raises(ValueError):
        SignalValue(name="x", kind="not-a-kind", risk=0.5)
