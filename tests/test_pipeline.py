"""Decision layer and end-to-end pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msrc import MSRCConfig, MSRCPipeline, Sample, StubProvider  # noqa: E402
from msrc.model import FusionHead, RiskCalibrator, RidgeLogistic  # noqa: E402
from msrc.types import RISK_FACTUAL, RISK_SAFETY  # noqa: E402


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

def test_calibrator_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(0, 1, 400), rng.normal(0, 1, 400)])
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    cal = RiskCalibrator(l2=0.05).fit(X, y)
    p = cal.predict_proba(X)
    assert p[y == 1].mean() > p[y == 0].mean()


def test_calibrator_refuses_a_single_class():
    """One class means a constant, and a constant cannot rank anything."""
    with pytest.raises(ValueError, match="one class"):
        RiskCalibrator().fit([[0.1], [0.2], [0.3]], [1, 1, 1])


def test_calibrator_handles_a_constant_column_without_dividing_by_zero():
    X = np.column_stack([np.ones(50), np.linspace(0, 1, 50)])
    y = (X[:, 1] > 0.5).astype(int)
    cal = RiskCalibrator().fit(X, y)
    assert np.all(np.isfinite(cal.coef_))
    # The constant column carries nothing, so its weight is shrunk away.
    assert abs(float(cal.coef_[1])) < 1e-6


def test_predict_before_fit_is_an_error_not_a_guess():
    with pytest.raises(RuntimeError):
        RiskCalibrator().predict_proba([[0.5]])


# ---------------------------------------------------------------------------
# The fusion guard. This is the defect the framework exists partly to prevent.
# ---------------------------------------------------------------------------

def test_fusion_head_is_refused_when_it_would_invert_the_ranking():
    """A head that contradicts the score it sharpens is never wanted.

    The extra channel is given real spread so the fit reaches the sign check
    rather than being stopped by the constant-channel check first.
    """
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 1, 200)
    extra = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < base).astype(int)

    class Inverting(RidgeLogistic):
        def fit(self, features, labels):
            self.mean_ = np.zeros(features.shape[1])
            self.std_ = np.ones(features.shape[1])
            self.coef_ = np.array([0.0, -0.5, 0.1])
            self.success_ = True
            return self

    import msrc.model as model_module

    original = model_module.RidgeLogistic
    model_module.RidgeLogistic = Inverting
    try:
        head = FusionHead().fit(base, extra, y, extra_names=["extra"])
    finally:
        model_module.RidgeLogistic = original

    assert head.fitted is False
    assert "negative weight" in head.refused_reason


def test_fusion_head_is_refused_when_the_extra_channel_is_constant():
    base = np.linspace(0, 1, 100)
    y = (base > 0.5).astype(int)
    head = FusionHead().fit(base, np.ones((100, 1)), y, extra_names=["flat"])
    assert head.fitted is False
    assert "constant" in head.refused_reason


def test_fusion_head_fits_when_the_extra_channel_carries_signal():
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 1, 300)
    extra = rng.uniform(0, 1, 300)
    y = ((base + extra) / 2 > 0.5).astype(int)
    head = FusionHead().fit(base, extra, y, extra_names=["extra"])
    assert head.fitted is True
    assert head.refused_reason == ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _records(n=60, with_internals=True):
    out = []
    for i in range(n):
        failed = i % 2
        answer = f"Flickr{' WRONG' if failed else ''}"
        primary = (
            Sample(
                text=answer,
                sequence_confidence=0.34 if failed else 0.88,
                sequence_entropy=2.8 if failed else 0.5,
                visual_attention_mass=0.12 if failed else 0.62,
            )
            if with_internals
            else None
        )
        out.append(
            {
                "question": "what is the website?",
                "answer": answer,
                "image": f"/img{i}.jpg",
                "label": failed,
                "_primary": primary,
            }
        )
    return out


def _fit(pipe, records):
    """Fit through the pipeline's live path, threading the primary sample in."""
    from msrc.provider import gather_evidence

    vectors, labels = [], []
    for rec in records:
        ev, _ = gather_evidence(
            pipe.provider, rec["question"], rec["answer"], rec["image"],
            risk_family=pipe.config.risk_family, k=pipe.config.k,
            prompts=pipe.prompts, primary_sample=rec["_primary"],
        )
        # Through the pipeline so the signal order is fixed on it, exactly as the
        # live fit() path does.
        report = pipe._report_for(ev)
        vectors.append([s.risk for s in report.signals])
        labels.append(rec["label"])
    return pipe.fit_from_signals(np.asarray(vectors, float), np.asarray(labels, int), pipe.signal_names)


def _pipeline(**kwargs):
    return MSRCPipeline(provider=StubProvider(), config=MSRCConfig(k=5, **kwargs))


def test_end_to_end_ranks_wrong_answers_above_right_ones():
    pipe = _pipeline()
    _fit(pipe, _records())

    good = pipe.score("q", "Flickr", "/i.jpg", primary_sample=Sample(
        text="Flickr", sequence_confidence=0.9, sequence_entropy=0.4, visual_attention_mass=0.6))
    bad = pipe.score("q", "Flickr WRONG", "/i.jpg", primary_sample=Sample(
        text="Flickr WRONG", sequence_confidence=0.34, sequence_entropy=2.8,
        visual_attention_mass=0.1))
    assert good.risk_score < bad.risk_score


def test_internal_signals_change_the_score_when_supplied():
    """The internal channels must actually carry weight, not just exist."""
    pipe = _pipeline()
    _fit(pipe, _records(with_internals=True))

    with_internals = pipe.score(
        "q", "Flickr", "/i.jpg",
        primary_sample=Sample(text="Flickr", sequence_confidence=0.9,
                              sequence_entropy=0.4, visual_attention_mass=0.6))
    without = pipe.score("q", "Flickr", "/i.jpg")
    assert with_internals.risk_score != without.risk_score
    assert any("unavailable" in w for w in without.warnings)


def test_scoring_before_fitting_is_an_error():
    """An unfitted pipeline has no calibrated scale; returning a number is worse
    than failing."""
    pipe = _pipeline()
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.score("q", "a", "/i.jpg")


def test_the_signal_set_is_pinned_between_fit_and_score():
    pipe = _pipeline()
    _fit(pipe, _records())
    pipe.signal_names = ["only_one"]
    with pytest.raises(RuntimeError, match="signal set changed"):
        pipe.score("q", "Flickr", "/i.jpg")


def test_the_safety_family_swaps_the_family_specific_signal():
    factual = MSRCPipeline(provider=StubProvider(), config=MSRCConfig(risk_family=RISK_FACTUAL))
    safety = MSRCPipeline(provider=StubProvider(), config=MSRCConfig(risk_family=RISK_SAFETY))
    from msrc.signals import ALL_SIGNALS

    f_names = [s.name for s in ALL_SIGNALS if s.applies_to(RISK_FACTUAL)]
    s_names = [s.name for s in ALL_SIGNALS if s.applies_to(RISK_SAFETY)]
    assert "grounding_check" in f_names and "grounding_check" not in s_names
    assert "policy_probe" in s_names and "policy_probe" not in f_names
    assert factual.config.risk_family != safety.config.risk_family


def test_provenance_records_what_was_fitted():
    pipe = _pipeline()
    _fit(pipe, _records())
    assert pipe.provenance["n_signals"] == 8
    assert pipe.provenance["n_internal_signals"] == 3
    assert pipe.provenance["provider"] == "stub"
    assert pipe.provenance["risk_family"] == RISK_FACTUAL


def test_scorer_round_trips_through_json(tmp_path):
    pipe = _pipeline()
    _fit(pipe, _records())
    path = pipe.save(tmp_path / "scorer.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "msrc-scorer"
    assert payload["provenance"]["n_signals"] == 8

    reloaded = MSRCPipeline.load(path)
    hardcoded = {"question": "q", "answer": "Flickr", "image": "/i.jpg"}

    # Score both against evidence captured once, so the comparison is exact.
    from msrc.provider import gather_evidence

    ev, _ = gather_evidence(StubProvider(), "q", "Flickr", "/i.jpg", k=5)
    a = pipe.score("q", "Flickr", "/i.jpg", evidence=ev.to_dict())
    b = reloaded.score("q", "Flickr", "/i.jpg", evidence=ev.to_dict())
    assert a.risk_score == pytest.approx(b.risk_score, abs=1e-9)
    assert hardcoded  # keep the fixture in view for readers


def test_signal_vector_is_reproducible_for_the_same_evidence():
    pipe = _pipeline()
    _fit(pipe, _records())
    from msrc.provider import gather_evidence

    ev, _ = gather_evidence(StubProvider(), "q", "Flickr", "/i.jpg", k=5)
    first = pipe.score("q", "Flickr", "/i.jpg", evidence=ev.to_dict()).risk_score
    second = pipe.score("q", "Flickr", "/i.jpg", evidence=ev.to_dict()).risk_score
    assert first == second


def test_confidence_is_the_complement_and_cannot_drift():
    pipe = _pipeline()
    _fit(pipe, _records())
    result = pipe.score("q", "Flickr", "/i.jpg")
    assert result.calibrated_confidence == pytest.approx(1.0 - result.risk_score, abs=1e-6)
    payload = result.to_dict()
    assert payload["calibrated_confidence"] == pytest.approx(1.0 - payload["risk_score"], abs=1e-6)


def test_every_result_carries_a_justification_per_signal():
    pipe = _pipeline()
    _fit(pipe, _records())
    result = pipe.score("q", "Flickr", "/i.jpg")
    for signal in result.signals.signals:
        assert signal.name
        assert signal.detail, f"{signal.name} has no detail string"
