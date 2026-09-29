# coding: utf-8
"""The NumPy-only claim, runnable.

    python examples/numpy_only.py

The decision layer -- evidence -> signals -> calibrated risk -- must work with
NumPy alone: no torch, no fastapi, no model weights, no network. That is what
makes the framework evaluable on a laptop before a GPU is committed to it, so it
is checked rather than assumed.

This lives here rather than inline in the workflow so that it can be run by hand
and covered by the test suite. The workflow still installs NumPy *only* before
calling it; running this script in a full environment proves nothing.

One trap it exists to catch: a pipeline fitted through ``fit_from_signals``
remembers the signal names it was fitted on and refuses to score a different set.
Fitting on invented column names therefore fails at scoring time, not at fit
time, which is the wrong moment to find out. ``signal_names_for`` is the list to
use, and this script is what keeps that honest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from msrc import MSRCConfig, MSRCPipeline, StubProvider, signal_names_for  # noqa: E402
from msrc.model import RiskCalibrator  # noqa: E402
from msrc.types import RISK_FACTUAL  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)

    names = signal_names_for(RISK_FACTUAL)
    assert len(names) == 8, names

    # The calibrator on its own is a plain NumPy logistic fit.
    X = rng.normal(size=(200, len(names)))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    calibrator = RiskCalibrator().fit(X, y)
    probabilities = calibrator.predict_proba(X)
    assert probabilities.shape == (200,)
    assert float(probabilities.min()) >= 0.0 and float(probabilities.max()) <= 1.0

    # And the whole pipeline, end to end, on a stubbed provider.
    pipeline = MSRCPipeline(provider=StubProvider(), config=MSRCConfig(k=2))
    pipeline.fit_from_signals(X, y, names)
    result = pipeline.score("q", "a", "/i.jpg")
    assert 0.0 <= result.risk_score <= 1.0, result.risk_score

    # Scoring is where a wrong column list surfaces, so exercise it twice: the
    # second call is the one that would fail if fit and score disagreed.
    again = pipeline.score("q2", "a2", "/i2.jpg")
    assert 0.0 <= again.risk_score <= 1.0, again.risk_score

    print(f"ok: decision layer works with numpy alone ({len(names)} signals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
