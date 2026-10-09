"""Signal layer: orientation, missing-not-zero, and the helper maths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msrc.signals import (  # noqa: E402
    ALL_SIGNALS,
    CrossModelAgreement,
    Evidence,
    GroundingCheck,
    OutputEntropy,
    PolicyProbe,
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
    RISK_FACTUAL,
    RISK_SAFETY,
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
    assert len(internal) == 3, [s.name for s in internal]


#: How many signals apply to each family, and how many of those run with one model.
#: The two families are no longer symmetric: `guard_model` is a safety judgement on
#: the response and means nothing for a hallucination, so it applies to safety only.
#: Stating the numbers here rather than asserting "8 everywhere" is the point --
#: an inventory that silently drifts is exactly what this file exists to catch.
FAMILY_COUNTS = {
    RISK_FACTUAL: {"applied": 8, "single_model": 7},
    RISK_SAFETY: {"applied": 9, "single_model": 8},
}


@pytest.mark.parametrize("family", sorted(FAMILY_COUNTS))
def test_each_family_gets_its_declared_signal_count(family):
    """The distinction that matters for the readme's claim is what *runs*:
    one of the applied signals needs a second model attached."""
    want = FAMILY_COUNTS[family]
    applied = [s for s in ALL_SIGNALS if s.applies_to(family)]
    assert len(applied) == want["applied"], [s.name for s in applied]
    assert sum(1 for s in applied if s.kind == SIGNAL_INTERNAL) == 3

    single_model = [s for s in applied if s.name != "cross_model_agreement"]
    assert len(single_model) == want["single_model"]
    assert sum(1 for s in single_model if s.kind == SIGNAL_INTERNAL) == 3
    assert sum(1 for s in single_model if s.kind == SIGNAL_EXTERNAL) == want["single_model"] - 3


def test_the_guard_signal_is_a_safety_judgement_and_applies_only_there():
    """A response being harmful says nothing about whether it matches the image."""
    guard = next(s for s in ALL_SIGNALS if s.name == "guard_model")
    assert guard.applies_to(RISK_SAFETY)
    assert not guard.applies_to(RISK_FACTUAL)


def test_the_two_guard_channels_are_separate_signals():
    """`policy_probe` (the model judging itself) and `guard_model` (a model trained
    for the judgement) are different measurements. Merging them would stop the
    calibrator weighting them differently, which is the whole reason to have both."""
    names = [s.name for s in ALL_SIGNALS]
    assert "policy_probe" in names
    assert "guard_model" in names
    assert names.index("policy_probe") != names.index("guard_model")


def test_only_one_signal_needs_a_second_model():
    """A second VLM is a real deployment cost; it must buy exactly one signal.

    Built with full external evidence and no internals, so the unavailable set is
    the second-model signal plus the three internal ones -- nothing else.
    """
    evidence = Evidence(
        question="q", answer="Flickr", primary=Sample(text="Flickr"),
        views=[Sample(text="Flickr")],
        resamples=[Sample(text="Flickr"), Sample(text="Flickr")],
        verifications=[Sample(text="yes")],
        ocr_texts=["Flickr"],
    )
    report = build_report(evidence)
    assert sorted(report.missing()) == [
        "cross_model_agreement", "output_entropy", "sequence_confidence", "visual_attention",
    ]
    assert len(report.available()) == 4, "four external signals are live here"

    # Now give the internals, which is the only thing a single model at full
    # access adds -- the second-model signal stays unavailable.
    with_internals = Evidence(
        question="q", answer="Flickr",
        primary=Sample(text="Flickr", sequence_confidence=0.9, sequence_entropy=0.3,
                       visual_attention_mass=0.5),
        views=[Sample(text="Flickr")],
        resamples=[Sample(text="Flickr"), Sample(text="Flickr")],
        verifications=[Sample(text="yes")],
        ocr_texts=["Flickr"],
    )
    live = build_report(with_internals)
    # Four external that need no second model, plus the three internal: seven.
    assert len(live.available()) == 7, [s.name for s in live.available()]
    assert live.missing() == ["cross_model_agreement"]
    assert sum(1 for s in live.available() if s.kind == SIGNAL_INTERNAL) == 3


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


def test_policy_probe_only_applies_to_the_safety_family():
    probe = PolicyProbe()
    assert probe.applies_to(RISK_SAFETY)
    assert not probe.applies_to(RISK_FACTUAL)


def test_grounding_only_applies_to_the_factual_family():
    check = GroundingCheck()
    assert check.applies_to(RISK_FACTUAL)
    assert not check.applies_to(RISK_SAFETY)


def test_unavailable_signals_enter_the_vector_as_zero_but_are_named():
    """External evidence present, internals and the second model absent.

    The four that cannot run must say so by name. The alternative -- filling them
    with a neutral value -- produces a constant column and a calibrator that looks
    healthy while encoding nothing.
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
        "cross_model_agreement", "output_entropy", "sequence_confidence", "visual_attention",
    ], report.missing()
    assert len(report.available()) == 4
    assert len(report.vector()) == len(report.signals) == 8


def test_every_signal_missing_when_there_is_no_evidence_at_all():
    report = build_report(Evidence(question="q", answer="a", primary=Sample(text="a")))
    assert len(report.missing()) == len(report.signals) == 8
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


def test_policy_probe_counts_violations():
    signal = PolicyProbe()
    clean = Evidence(question="q", answer="a", policy_verdicts=[Sample(text="safe")])
    dirty = Evidence(question="q", answer="a",
                     policy_verdicts=[Sample(text="unsafe: violence"), Sample(text="unsafe: hate")])
    assert signal.compute(clean).risk == pytest.approx(0.0)
    assert signal.compute(dirty).risk == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The invariant itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signal", ALL_SIGNALS, ids=lambda s: s.name)
def test_every_signal_returns_a_risk_in_the_unit_interval(signal):
    """Every signal, on every branch, must satisfy the contract."""
    rich = Evidence(
        question="q", answer="Flickr", risk_family=RISK_SAFETY,
        primary=Sample(text="Flickr", sequence_confidence=0.9, sequence_entropy=0.3,
                       visual_attention_mass=0.5),
        views=[Sample(text="Flickr")],
        resamples=[Sample(text="Flickr"), Sample(text="Nikon")],
        verifications=[Sample(text="yes")],
        cross_model=[Sample(text="Flickr")],
        policy_verdicts=[Sample(text="safe")],
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
