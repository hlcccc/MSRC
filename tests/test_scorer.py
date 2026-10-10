"""Cached evidence in, a calibrated hallucination risk out.

The claim this file exists to defend is narrow and specific: given cached evidence
rows -- the format the collector writes -- a scorer fits one calibrator on one
signal matrix and reports a risk per item, with the columns behind it.

That claim is easy to fake. A scorer that silently builds a different matrix at fit
time than at scoring time would look identical from the outside; so would one whose
score ignores a column entirely. Both are checked here by construction: the matrix
shape and column order are pinned, and moving the one column the calibration is
grounded on has to move the score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msrc.scorer import RiskScore, RiskScorer  # noqa: E402
from msrc.signals import Evidence, SIGNAL_NAMES, build_report  # noqa: E402
from msrc.types import Sample  # noqa: E402

WORDS = ["No", "no", "No, there is no car", "Yes", "yes", "None of these"]


def make_evidence(seed=0, **overrides):
    """One item's evidence, with every field the signals read.

    Every reading is drawn from `seed`. An earlier version left the primary
    sample's internals fixed, which made every item's feature vector identical and
    collapsed the fit to the base rate -- the tests then compared two constants and
    could not tell a working scorer from a broken one.
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
    ev.ocr_texts = overrides.pop("ocr", ["STOP", "SALE"])
    for key, value in overrides.items():
        setattr(ev, key, value)
    return ev


def labelled(n, seed=1):
    """`n` cached-evidence records with a mostly-consistent label.

    The label is a function of the readings with noise, so the calibrator has
    something to fit and is not degenerate.

    The grounding column has to vary across these rows, or its weight is fitted to
    zero and a later test that changes it at scoring time sees nothing move -- not
    because the score ignores it but because that column was constant during
    training.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # three genuinely different grounding readings: nothing read off the
        # image, unrelated text, and the answer verbatim. The answer has to be
        # more than a word or the containment test fires on any OCR string that
        # happens to contain it as a substring -- "no" is inside "nothing".
        ocr = ([], ["village", "fountain", "path"],
               ["no", "there", "is", "no", "car", "in", "the", "image"])[i % 3]
        ev = make_evidence(seed=seed * 100 + i,
                           answer="No, there is no car in the image", ocr=ocr)
        x = np.asarray(build_report(ev).vector())
        p = 1.0 / (1.0 + np.exp(-(3.0 * (x.mean() - 0.4))))
        rows.append({"evidence": ev.to_dict(), "label": int(rng.random() < 0.3 + 0.5 * p)})
    return rows


# ---------------------------------------------------------------------------
# The matrix is the declared one
# ---------------------------------------------------------------------------

def test_the_matrix_carries_the_declared_columns_in_order():
    rows = labelled(12)
    X = RiskScorer.matrix(rows)
    assert X.shape == (12, len(SIGNAL_NAMES))
    assert np.isfinite(X).all()
    assert RiskScorer().signal_names == list(SIGNAL_NAMES)


# ---------------------------------------------------------------------------
# One evidence object, one risk
# ---------------------------------------------------------------------------

def test_one_evidence_object_yields_a_risk():
    scorer = RiskScorer().fit(labelled(60, seed=2))
    out = scorer.score(make_evidence(seed=7))

    assert isinstance(out, RiskScore)
    assert 0.0 <= out.risk <= 1.0
    assert out.available is True
    assert out.to_dict()["warning"] is out.warning


def test_changing_the_grounding_reading_moves_the_score():
    """`grounding_check` is where the OCR text enters. If the score did not move,
    the fit would be reading a different matrix than the one scored."""
    scorer = RiskScorer().fit(labelled(60, seed=2))
    answer = "No, there is no car in the image"
    a = scorer.score(make_evidence(seed=7, answer=answer,
                                   ocr=["village", "fountain", "path"]))
    b = scorer.score(make_evidence(seed=7, answer=answer,
                                   ocr=["no", "there", "is", "no", "car", "in",
                                        "the", "image"]))
    assert a.risk != pytest.approx(b.risk)


# ---------------------------------------------------------------------------
# Honest reporting
# ---------------------------------------------------------------------------

def test_an_unfitted_scorer_says_so_instead_of_reporting_zero():
    """0.0 reads as "we looked and it is fine". An unfitted scorer must not."""
    scorer = RiskScorer()
    out = scorer.score(make_evidence())
    assert out.available is False
    assert np.isnan(out.risk)
    assert out.warning is False
    assert out.to_dict()["available"] is False


def test_missing_signals_are_named():
    ev = make_evidence()
    ev.ocr_texts = []                      # grounding_check produces nothing
    ev.cross_model = []                    # so does cross_model_agreement
    scorer = RiskScorer().fit(labelled(60, seed=2))
    out = scorer.score(ev)
    assert "grounding_check" in out.missing
    assert "cross_model_agreement" in out.missing


def test_the_threshold_comes_from_the_development_rows():
    rows = labelled(80, seed=2)
    scorer = RiskScorer().fit(rows).choose_threshold(rows)
    grid = np.round(np.linspace(0.05, 0.95, 91), 4)
    assert scorer.threshold in set(grid.tolist())

    scores = scorer._scores(rows)
    y = np.asarray([r["label"] for r in rows])
    best = max(float(((scores >= t).astype(int) == y).mean()) for t in grid)
    got = float(((scores >= scorer.threshold).astype(int) == y).mean())
    assert got == pytest.approx(best)


def test_a_tie_goes_to_the_higher_threshold():
    """Warning less is the conservative side, and the tie rule is what encodes it."""
    rows = labelled(80, seed=2)
    scorer = RiskScorer().fit(rows).choose_threshold(rows)
    grid = np.round(np.linspace(0.05, 0.95, 91), 4)
    scores = scorer._scores(rows)
    y = np.asarray([r["label"] for r in rows])
    accs = np.array([float(((scores >= t).astype(int) == y).mean()) for t in grid])
    tied = grid[accs == accs.max()]
    assert scorer.threshold == pytest.approx(float(tied.max()))


# ---------------------------------------------------------------------------
# Refusals and persistence
# ---------------------------------------------------------------------------

def test_fitting_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to fit"):
        RiskScorer().fit([])


def test_a_single_class_set_is_refused():
    rows = labelled(20, seed=2)
    for r in rows:
        r["label"] = 1
    with pytest.raises(ValueError, match="single class"):
        RiskScorer().fit(rows)


def test_a_scorer_round_trips_through_a_file(tmp_path):
    rows = labelled(50, seed=2)
    scorer = RiskScorer().fit(rows).choose_threshold(rows)
    path = scorer.save(tmp_path / "risk.json")

    back = RiskScorer.load(path)
    ev = make_evidence(seed=11)
    a, b = scorer.score(ev), back.score(ev)
    assert a.risk == pytest.approx(b.risk)
    assert back.threshold == pytest.approx(scorer.threshold)
    assert back.weights()[0]["signal"]
    assert back.provenance["fit"]["n"] == 50


def test_a_file_that_is_not_a_risk_scorer_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"format": "msrc-scorer"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an MSRC risk scorer"):
        RiskScorer.load(path)
