"""The NumPy-only claim, and the trap that kept it red.

The workflow's NumPy-only job fitted a pipeline on invented column names and
then scored it. The pipeline refuses that -- correctly -- but only at scoring
time, so the job failed on every commit from the first one onward. A check that
has never been green stops being read as a signal, which is why the failure
outlived ten pushes.

Two things are asserted here so it cannot come back:

* ``signal_names_for`` really is the list a report carries for a family. If the
  two ever drift, fitting from a prebuilt matrix breaks again.
* the NumPy-only example runs, since that is the artefact the job invokes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msrc import (  # noqa: E402
    MSRCConfig,
    MSRCPipeline,
    StubProvider,
    gather_evidence,
    signal_names_for,
)
from msrc.signals import build_report  # noqa: E402
from msrc.types import RISK_FACTUAL, RISK_FAMILIES, RISK_SAFETY  # noqa: E402


@pytest.mark.parametrize("family", RISK_FAMILIES)
def test_signal_names_for_matches_what_a_report_carries(family):
    """The helper must agree with the pipeline, not just with the signal list."""
    evidence, _ = gather_evidence(
        StubProvider(ocr_text="Flickr"),
        "q",
        "Flickr",
        "/i.jpg",
        risk_family=family,
        k=2,
    )
    assert build_report(evidence).names() == signal_names_for(family)


def test_each_family_names_eight_signals_applying_seven_with_one_model():
    """The README's inventory, asserted against the code rather than the prose."""
    for family in RISK_FAMILIES:
        names = signal_names_for(family)
        assert len(names) == 8, (family, names)
        single = [n for n in names if n != "cross_model_agreement"]
        assert len(single) == 7, (family, single)


def test_fitting_on_invented_names_is_refused_at_scoring_time():
    """The exact shape of the old workflow failure, pinned as a known refusal."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    pipeline = MSRCPipeline(provider=StubProvider(), config=MSRCConfig(k=2))
    pipeline.fit_from_signals(X, y, [f"s{i}" for i in range(8)])

    with pytest.raises(RuntimeError, match="signal set changed since fitting"):
        pipeline.score("q", "a", "/i.jpg")


def test_the_example_scoring_path_survives_a_second_call():
    """Fit once, score twice: the second call is where a column mismatch lands."""
    rng = np.random.default_rng(0)
    names = signal_names_for(RISK_FACTUAL)
    X = rng.normal(size=(200, len(names)))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    pipeline = MSRCPipeline(provider=StubProvider(), config=MSRCConfig(k=2))
    pipeline.fit_from_signals(X, y, names)

    for _ in range(2):
        result = pipeline.score("q", "a", "/i.jpg")
        assert 0.0 <= result.risk_score <= 1.0


def test_the_numpy_only_example_runs():
    """The script the workflow invokes, run the way the workflow runs it."""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "numpy_only.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "numpy alone" in completed.stdout


def test_signal_names_for_rejects_an_unknown_family_by_returning_nothing():
    """An unknown family is not an error here -- it simply has no signals, and
    grounding_check/policy_probe are what make the two known families differ."""
    assert signal_names_for("not-a-family") == []
    assert "grounding_check" in signal_names_for(RISK_FACTUAL)
    assert "grounding_check" not in signal_names_for(RISK_SAFETY)
    assert "policy_probe" in signal_names_for(RISK_SAFETY)
