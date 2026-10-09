"""One evidence object, two risks.

The claim this file exists to defend is narrow and specific: given a *single*
collection of evidence -- one image, one question, one answer, the follow-up calls
already made -- the system reports a hallucination risk and a content-safety risk,
each from its own calibrated head.

That claim is easy to fake. A dual scorer that silently builds the same matrix twice
would look identical from the outside; so would one whose second head reads the
first head's column set. Both are checked here by construction: the two heads must
disagree about exactly one column, and moving a reading that only one head uses must
move only that head.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msrc.dual import DualRisk, DualRiskScorer, signal_names_for  # noqa: E402
from msrc.signals import Evidence, build_report  # noqa: E402
from msrc.types import RISK_FACTUAL, RISK_SAFETY, Sample  # noqa: E402

WORDS = ["No", "no", "No, there is no car", "Yes", "yes", "None of these"]


def make_evidence(seed=0, **overrides):
    """One item's evidence, with every field the two heads read.

    Both the OCR strings (factual head only) and the policy verdicts (safety head
    only) are present, which is the whole point: one collection serves both.

    Every reading is drawn from `seed`. An earlier version left the primary
    sample's internals fixed, which made every item's feature vector identical and
    collapsed both fits to the base rate -- the tests then compared two constants
    and could not tell a working dual scorer from a broken one.
    """
    rng = np.random.default_rng(seed)
    answer = overrides.pop("answer", WORDS[seed % len(WORDS)])
    ev = Evidence(question="Is there a car in the image?", answer=answer,
                  image="/tmp/x.png")
    ev.primary = Sample(
        text=answer,
        sequence_confidence=float(rng.uniform(0.35, 0.95)),
        sequence_entropy=float(rng.uniform(0.15, 2.2)),
        visual_attention_mass=float(rng.uniform(0.04, 0.45)),
    )
    for name, n in (("views", 4), ("resamples", 3), ("verifications", 3)):
        getattr(ev, name).extend(
            Sample(
                text=WORDS[(seed + i) % len(WORDS)],
                sequence_confidence=float(rng.uniform(0.3, 0.9)),
                sequence_entropy=float(rng.uniform(0.1, 1.5)),
                visual_attention_mass=float(rng.uniform(0.05, 0.4)),
            )
            for i in range(n)
        )
    ev.cross_model = [Sample(text=WORDS[(seed + 2) % len(WORDS)],
                             sequence_confidence=0.88, sequence_entropy=0.27)]
    ev.policy_verdicts = [Sample(text=overrides.pop("verdict", "safe"),
                                 sequence_confidence=0.91, sequence_entropy=0.33)]
    ev.ocr_texts = overrides.pop("ocr", ["STOP", "SALE"])
    for key, value in overrides.items():
        setattr(ev, key, value)
    return ev


def labelled(n, family, seed=1):
    """`n` cached-evidence records with a mostly-consistent label.

    The label is a function of the readings with noise, so both calibrators have
    something to fit and neither is degenerate.

    The column that only one head uses has to vary across these rows, or its weight
    is fitted to zero and a later test that changes it at scoring time sees nothing
    move -- not because the heads are entangled but because that column was constant
    during training.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if family == RISK_FACTUAL:
            # three genuinely different grounding readings: nothing read off the
            # image, unrelated text, and the answer verbatim. The answer has to be
            # more than a word or the containment test fires on any OCR string that
            # happens to contain it as a substring -- "no" is inside "nothing".
            ocr = ([], ["village", "fountain", "path"],
                   ["no", "there", "is", "no", "car", "in", "the", "image"])[i % 3]
            ev = make_evidence(seed=seed * 100 + i,
                               answer="No, there is no car in the image", ocr=ocr)
        else:
            verdict = ["safe", "unsafe: illegal activity",
                       "unsafe: violence"][i % 3]
            ev = make_evidence(seed=seed * 100 + i, verdict=verdict)
        x = np.asarray(build_report(
            dataclasses.replace(ev, risk_family=family)).vector())
        p = 1.0 / (1.0 + np.exp(-(3.0 * (x.mean() - 0.4))))
        rows.append({"evidence": ev.to_dict(), "label": int(rng.random() < 0.3 + 0.5 * p)})
    return rows


# ---------------------------------------------------------------------------
# The two heads are two heads
# ---------------------------------------------------------------------------

def test_the_two_heads_differ_in_exactly_one_column():
    f = signal_names_for(RISK_FACTUAL)
    s = signal_names_for(RISK_SAFETY)
    assert len(f) == len(s) == 8
    assert set(f) - set(s) == {"grounding_check"}
    assert set(s) - set(f) == {"policy_probe"}
    assert [n for n in f if n != "grounding_check"] == [n for n in s if n != "policy_probe"]


@pytest.mark.parametrize("family,wanted", [(RISK_FACTUAL, 8), (RISK_SAFETY, 8)])
def test_each_head_builds_its_own_eight_wide_matrix(family, wanted):
    rows = labelled(12, family)
    X = DualRiskScorer.matrix(rows, family)
    assert X.shape == (12, wanted)
    assert np.isfinite(X).all()


def test_an_unknown_family_is_refused():
    with pytest.raises(ValueError, match="unknown risk family"):
        signal_names_for("vibes")


# ---------------------------------------------------------------------------
# One evidence object serves both
# ---------------------------------------------------------------------------

def test_one_evidence_object_yields_both_risks():
    scorer = DualRiskScorer().fit(labelled(60, RISK_FACTUAL, seed=2),
                                  labelled(60, RISK_SAFETY, seed=3))
    ev = make_evidence(seed=7)
    out = scorer.score(ev)

    assert isinstance(out, DualRisk)
    assert 0.0 <= out.hallucination_risk <= 1.0
    assert 0.0 <= out.safety_risk <= 1.0
    assert out.hallucination_available and out.safety_available
    assert out.to_dict()["warning"] == (out.hallucination_warning or out.safety_warning)


def test_the_only_difference_between_the_heads_is_the_swapped_column():
    """Change the OCR text and only the hallucination score may move.

    `grounding_check` is the factual head's alone. If the safety score moved too,
    the two heads would be reading the same matrix and the whole dual construction
    would be decoration.
    """
    scorer = DualRiskScorer().fit(labelled(60, RISK_FACTUAL, seed=2),
                                  labelled(60, RISK_SAFETY, seed=3))
    answer = "No, there is no car in the image"
    a = scorer.score(make_evidence(seed=7, answer=answer,
                                   ocr=["village", "fountain", "path"]))
    b = scorer.score(make_evidence(seed=7, answer=answer,
                                   ocr=["no", "there", "is", "no", "car", "in",
                                        "the", "image"]))
    assert a.hallucination_risk != pytest.approx(b.hallucination_risk)
    assert a.safety_risk == pytest.approx(b.safety_risk)


def test_change_the_policy_verdict_and_only_the_safety_score_may_move():
    scorer = DualRiskScorer().fit(labelled(60, RISK_FACTUAL, seed=2),
                                  labelled(60, RISK_SAFETY, seed=3))
    a = scorer.score(make_evidence(seed=7, answer="No, there is no car in the image",
                                   verdict="safe"))
    b = scorer.score(make_evidence(seed=7, answer="No, there is no car in the image",
                                   verdict="unsafe: illegal activity"))
    assert a.safety_risk != pytest.approx(b.safety_risk)
    assert a.hallucination_risk == pytest.approx(b.hallucination_risk)


# ---------------------------------------------------------------------------
# Honest reporting
# ---------------------------------------------------------------------------

def test_a_head_that_was_never_fitted_says_so_instead_of_reporting_zero():
    """0.0 reads as "we looked and it is fine". An unfitted head must not."""
    scorer = DualRiskScorer().fit(labelled(40, RISK_FACTUAL, seed=5))
    out = scorer.score(make_evidence())
    assert out.hallucination_available is True
    assert out.safety_available is False
    assert np.isnan(out.safety_risk)
    assert out.safety_warning is False
    assert out.to_dict()["safety_available"] is False


def test_missing_signals_are_named_per_head():
    ev = make_evidence()
    ev.ocr_texts = []                      # grounding_check produces nothing
    ev.cross_model = []                    # so does cross_model_agreement
    scorer = DualRiskScorer().fit(labelled(60, RISK_FACTUAL, seed=2),
                                  labelled(60, RISK_SAFETY, seed=3))
    out = scorer.score(ev)
    assert "grounding_check" in out.hallucination_missing
    assert "cross_model_agreement" in out.hallucination_missing
    assert "grounding_check" not in out.safety_missing


def test_each_head_gets_its_own_threshold_from_its_own_development_rows():
    f_rows, s_rows = labelled(80, RISK_FACTUAL, seed=2), labelled(80, RISK_SAFETY, seed=3)
    scorer = DualRiskScorer().fit(f_rows, s_rows).choose_thresholds(f_rows, s_rows)
    for rows, family, thr in ((f_rows, RISK_FACTUAL, scorer.factual_threshold),
                              (s_rows, RISK_SAFETY, scorer.safety_threshold)):
        assert thr in set(np.round(np.linspace(0.05, 0.95, 91), 4).tolist())
        scores = scorer._scores(rows, family)
        y = np.asarray([r["label"] for r in rows])
        grid = np.round(np.linspace(0.05, 0.95, 91), 4)
        best = max(float(((scores >= t).astype(int) == y).mean()) for t in grid)
        got = float(((scores >= thr).astype(int) == y).mean())
        assert got == pytest.approx(best)


# ---------------------------------------------------------------------------
# Refusals and persistence
# ---------------------------------------------------------------------------

def test_fitting_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to fit"):
        DualRiskScorer().fit()


def test_a_single_class_set_is_refused():
    rows = labelled(20, RISK_FACTUAL, seed=2)
    for r in rows:
        r["label"] = 1
    with pytest.raises(ValueError, match="single class"):
        DualRiskScorer().fit(rows)


def test_a_scorer_round_trips_through_a_file(tmp_path):
    f_rows, s_rows = labelled(50, RISK_FACTUAL, seed=2), labelled(50, RISK_SAFETY, seed=3)
    scorer = DualRiskScorer().fit(f_rows, s_rows).choose_thresholds(f_rows, s_rows)
    path = scorer.save(tmp_path / "dual.json")

    back = DualRiskScorer.load(path)
    ev = make_evidence(seed=11)
    a, b = scorer.score(ev), back.score(ev)
    assert a.hallucination_risk == pytest.approx(b.hallucination_risk)
    assert a.safety_risk == pytest.approx(b.safety_risk)
    assert back.factual_threshold == pytest.approx(scorer.factual_threshold)
    assert back.safety_threshold == pytest.approx(scorer.safety_threshold)
    assert back.weights()["safety"][0]["signal"]
    assert back.provenance["factual"]["n"] == 50


def test_a_file_that_is_not_a_dual_scorer_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"format": "msrc-scorer"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an MSRC dual scorer"):
        DualRiskScorer.load(path)
