"""HTTP service: the framework behind an endpoint.

The service takes **evidence**, not raw answers, and that is deliberate. The whole
design of this framework turns on where the numbers came from -- the internal
signals exist only if they were captured in the call that produced the answer --
so an endpoint that accepted `{question, answer, image}` and invented the rest
would be offering a score it cannot justify. Evidence is produced by
`scripts/collect_evidence.py` or by the platform's own adapter.

Two behaviours worth stating because the alternative is worse:

* **An unfitted scorer is a 503, not a number.** A prior dressed up as a
  calibrated probability is the failure this framework was built to avoid, and an
  endpoint is where it would do the most damage.
* **There is no authentication, no rate limiting and no request timeout.** This is
  a decision, not an oversight: the service assumes it runs inside the platform's
  network behind a gateway. Do not expose it directly. See docs/evaluation.md.

Models are declared at module scope. Defining them inside `create_app` would make
Pydantic resolve their forward references lazily and every request would fail
validation -- a defect this project's predecessor shipped once.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "the service needs the service extra:\n  pip install -e \".[service]\""
    ) from exc

from msrc.conformal import VALIDATED as CONFORMAL_VALIDATED
from msrc.conformal import VALID_PROCEDURES, selective_report
from msrc.pipeline import MSRCPipeline
from msrc.signals import ALL_SIGNALS

__all__ = ["create_app", "RiskRequest", "SelectRequest"]


# ---------------------------------------------------------------------------
# Request and response models, at module level
# ---------------------------------------------------------------------------


class RiskRequest(BaseModel):
    """One item's evidence, as produced by a collection run."""

    evidence: Dict[str, Any] = Field(
        ...,
        description="The `evidence` object for one item. It carries the readings "
                    "the signals are computed from, including the generation's "
                    "internals when the platform captured them.",
    )


class SignalOut(BaseModel):
    name: str
    kind: str
    risk: float
    available: bool
    detail: str


class RiskResponse(BaseModel):
    risk_score: float
    calibrated_confidence: float
    is_high_risk: bool
    threshold: float
    risk_family: str
    signals: List[SignalOut]
    n_signals_available: int
    n_signals_missing: int
    missing_signals: List[str]
    warnings: List[str]
    version: str


class SelectRequest(BaseModel):
    calibration_null_scores: List[float] = Field(
        ...,
        description="Risk scores of items known to satisfy the null (correct "
                    "answers). They must be exchangeable with the test items; "
                    "nothing here can check that.",
    )
    test_scores: List[float]
    test_labels: Optional[List[int]] = Field(
        None,
        description="Only used to report realized FDP. The guarantee does not "
                    "depend on them, and supplying them does not improve it.",
    )
    alpha: float = 0.10
    procedure: str = "BY"


class SignalInfo(BaseModel):
    name: str
    kind: str
    description: str
    families: List[str]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app(pipeline: Optional[MSRCPipeline] = None) -> FastAPI:
    """Build the app around a fitted pipeline.

    Raises if the pipeline carries no fitted coefficients: serving one would mean
    answering every request with a number that has no calibrated meaning, and a
    200 response is exactly the signal a caller uses to decide it can trust the
    value.
    """
    if pipeline is None:
        raise ValueError(
            "create_app needs a fitted pipeline.\n"
            "  Fit one first: msrc fit --data evidence.jsonl --out scorer.json\n"
            "  Then load it:   MSRCPipeline.load('scorer.json')"
        )
    if not pipeline.fitted_:
        raise ValueError(
            "the pipeline has no fitted coefficients. Serving it would return an "
            "uncalibrated prior with a 200 status; fit it first."
        )

    app = FastAPI(
        title="MSRC",
        version=pipeline.config.version,
        description=(
            "Multi-signal risk calibration. Scores evidence, not raw answers; see "
            "docs/evaluation.md for why. No authentication: run behind a gateway."
        ),
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "scorer_fitted": bool(pipeline.fitted_),
            "risk_family": pipeline.config.risk_family,
            "n_signals": len(pipeline.signal_names),
            "signals": list(pipeline.signal_names),
            "provenance": dict(pipeline.provenance),
            "conformal_validated": CONFORMAL_VALIDATED,
            "auth": "none -- run behind a gateway",
        }

    @app.get("/v1/msrc/signals", response_model=List[SignalInfo])
    def signals() -> List[SignalInfo]:
        family = pipeline.config.risk_family
        return [
            SignalInfo(
                name=s.name,
                kind=s.kind,
                description=s.description,
                families=[f for f in s.families],
            )
            for s in ALL_SIGNALS
            if s.applies_to(family)
        ]

    @app.post("/v1/msrc/risk", response_model=RiskResponse)
    def risk(request: RiskRequest) -> RiskResponse:
        if not request.evidence:
            raise HTTPException(status_code=400, detail="evidence is empty")
        try:
            result = pipeline.score("", "", "", evidence=request.evidence)
        except RuntimeError as exc:
            # A signal-set mismatch is a scorer/deployment problem, not a bad
            # request; saying so avoids sending the caller hunting through theirs.
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=400, detail=f"evidence could not be read: {exc}"
            ) from exc

        report = result.signals
        payload = result.to_dict()
        return RiskResponse(
            risk_score=payload["risk_score"],
            calibrated_confidence=payload["calibrated_confidence"],
            is_high_risk=payload["is_high_risk"],
            threshold=payload["threshold"],
            risk_family=payload["risk_family"],
            signals=[SignalOut(**s) for s in payload["signals"]["signals"]],
            n_signals_available=payload["signals"]["n_available"],
            n_signals_missing=len(report.missing()),
            missing_signals=report.missing(),
            warnings=payload["warnings"],
            version=payload["version"],
        )

    @app.post("/v1/msrc/select")
    def select(request: SelectRequest) -> Dict[str, Any]:
        if request.procedure not in VALID_PROCEDURES:
            raise HTTPException(
                status_code=400,
                detail=f"procedure must be one of {list(VALID_PROCEDURES)}",
            )
        try:
            return selective_report(
                request.calibration_null_scores,
                request.test_scores,
                test_labels=request.test_labels,
                alpha=request.alpha,
                procedure=request.procedure,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
