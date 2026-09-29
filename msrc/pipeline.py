"""End-to-end orchestration: evidence -> signals -> calibrated risk.

The pipeline holds a *frozen* configuration: a signal order, a fitted risk map and
an optional fusion head. ``score`` updates nothing, which is what makes an exported
scorer reproducible.

Provenance is recorded, not assumed
-----------------------------------

The exported file says which signals the scorer was fitted on, how many of them
were actually available, and which provider produced the evidence. A scorer fitted
with an internal channel missing is therefore identifiable afterwards -- the
alternative, which an earlier framework in this line shipped, is a file that looks
complete and quietly encodes constants.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from msrc.model import FusionHead, RiskCalibrator
from msrc.provider import PromptSet, gather_evidence
from msrc.signals import ALL_SIGNALS, Evidence, build_report
from msrc.types import (
    RISK_FACTUAL,
    RISK_FAMILIES,
    SIGNAL_INTERNAL,
    RiskResult,
    SignalReport,
)

__all__ = ["MSRCConfig", "MSRCPipeline", "DEFAULT_THRESHOLD"]

DEFAULT_THRESHOLD = 0.5

#: Signals whose value comes from the model internals. A deployment that cannot
#: supply them is still supported; it is recorded so the fitted scorer can say so.
INTERNAL_SIGNAL_NAMES = tuple(s.name for s in ALL_SIGNALS if s.kind == SIGNAL_INTERNAL)


@dataclass
class MSRCConfig:
    """Frozen, serialisable configuration of a scorer."""

    version: str = "0.1.0"
    risk_family: str = RISK_FACTUAL
    #: Resampling depth for the consistency signal. K=5 is the usual setting;
    #: K=1 disables the signal, which then reports itself unavailable.
    k: int = 5
    l2: float = 0.05
    fusion_l2: float = 0.02
    threshold: float = DEFAULT_THRESHOLD
    #: Whether the fusion head may be fitted over the calibrated score plus the
    #: resampling entropy. Off by default: the head needs enough labelled items to
    #: be worth fitting, and refusing to fit is safer than fitting badly.
    use_fusion: bool = False

    def __post_init__(self) -> None:
        if self.risk_family not in RISK_FAMILIES:
            raise ValueError(f"unknown risk family: {self.risk_family!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MSRCConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (payload or {}).items() if k in known})


class MSRCPipeline:
    """Fit on labelled items, then score new ones."""

    def __init__(
        self,
        provider: Any = None,
        second_provider: Any = None,
        config: Optional[MSRCConfig] = None,
        calibrator: Optional[RiskCalibrator] = None,
        fusion: Optional[FusionHead] = None,
        prompts: Optional[PromptSet] = None,
    ):
        self.config = config or MSRCConfig()
        self.provider = provider
        self.second_provider = second_provider
        self.prompts = prompts or PromptSet()
        self.calibrator = calibrator
        self.fusion = fusion
        self.fitted_ = calibrator is not None and calibrator.coef_ is not None
        self.fit_warnings: List[str] = []
        self.provenance: Dict[str, Any] = {}
        #: Fixed signal order. Frozen at fit time and stored with the scorer so a
        #: later scoring run cannot silently use a different column order.
        self.signal_names: List[str] = []

    # ------------------------------------------------------------------
    # Evidence -> features
    # ------------------------------------------------------------------
    def _report_for(self, evidence: Evidence) -> SignalReport:
        report = build_report(evidence)
        if not self.signal_names:
            self.signal_names = report.names()
        return report

    def fit(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        label_key: str = "label",
    ) -> "MSRCPipeline":
        """Fit the risk map on labelled records.

        Each record needs ``answer`` and ``label``; ``question`` and ``image`` are
        used to gather evidence. Pass ``evidence`` (a dict produced by
        :meth:`Evidence.to_dict`) to reuse readings collected earlier instead of
        calling the model again -- which is how a large evaluation is run.
        """
        vectors: List[List[float]] = []
        labels: List[int] = []
        call_count = 0
        missing_seen: Dict[str, int] = {}

        for record in records:
            payload = record.get("evidence")
            if payload:
                evidence = Evidence.from_dict(payload)
            else:
                if self.provider is None:
                    raise ValueError(
                        "a record has no cached evidence and no provider is attached"
                    )
                evidence, calls = gather_evidence(
                    self.provider,
                    str(record.get("question", "")),
                    str(record.get("answer", "")),
                    str(record.get("image", "")),
                    risk_family=self.config.risk_family,
                    k=self.config.k,
                    prompts=self.prompts,
                    second_provider=self.second_provider,
                )
                call_count += calls

            report = self._report_for(evidence)
            for name in report.missing():
                missing_seen[name] = missing_seen.get(name, 0) + 1

            labels.append(int(record[label_key]))
            # Unavailable signals enter as 0.0 risk. That is a *value*, not a
            # silent default: the column becomes constant, the L2 penalty shrinks
            # its weight toward zero, and the provenance below records that it was
            # never available. The warning is what stops it being invisible.
            vectors.append([s.risk for s in report.signals])

        if not vectors:
            raise ValueError("no usable records")
        X = np.asarray(vectors, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)

        self.calibrator = RiskCalibrator(l2=self.config.l2).fit(X, y)
        self.fitted_ = self.calibrator.coef_ is not None

        constant = [n for i, n in enumerate(self.signal_names)
                    if float(np.ptp(X[:, i])) <= 1e-12]
        if constant:
            self.fit_warnings.append(
                "these signals were constant across the development set and carry no "
                f"weight: {', '.join(constant)}. If a signal is an internal one, the "
                "serving stack most likely does not expose what it needs."
            )
        for name, count in sorted(missing_seen.items()):
            self.fit_warnings.append(
                f"signal {name!r} was unavailable for {count} of {len(records)} items"
            )

        if self.config.use_fusion:
            entropy_col = (
                self.signal_names.index("resample_consistency")
                if "resample_consistency" in self.signal_names
                else None
            )
            if entropy_col is not None:
                base = self.calibrator.predict_proba(X)
                head = FusionHead(l2=self.config.fusion_l2).fit(
                    base, X[:, [entropy_col]], y, extra_names=["resample_consistency"]
                )
                self.fusion = head if head.fitted else None
                if head.refused_reason:
                    self.fit_warnings.append(head.refused_reason)

        self.provenance = {
            "provider": getattr(self.provider, "provider_kind", None),
            "second_provider": getattr(self.second_provider, "provider_kind", None),
            "risk_family": self.config.risk_family,
            "signals": list(self.signal_names),
            "n_signals": len(self.signal_names),
            "n_internal_signals": sum(
                1 for s in ALL_SIGNALS if s.kind == SIGNAL_INTERNAL and s.name in self.signal_names
            ),
            "n_items": int(len(y)),
            "label_rate": float(y.mean()),
            "model_calls": call_count,
            "fit_warnings": list(self.fit_warnings),
        }
        return self

    def fit_from_signals(
        self,
        matrix: Any,
        labels: Any,
        signal_names: Sequence[str],
        *,
        model_calls: int = 0,
    ) -> "MSRCPipeline":
        """Fit on a prebuilt signal matrix (used by the evaluation harness).

        Provenance is recorded here too. It would be easy to treat this path as
        the "fast" one and skip it, but a scorer whose file does not say what it
        was fitted on is exactly the artefact this framework refuses to produce.
        """
        X = np.asarray(matrix, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("matrix must be 2-D with one row per label")

        self.signal_names = list(signal_names)
        self.calibrator = RiskCalibrator(l2=self.config.l2).fit(X, y)
        self.fitted_ = self.calibrator.coef_ is not None

        constant = [n for i, n in enumerate(self.signal_names)
                    if float(np.ptp(X[:, i])) <= 1e-12]
        if constant:
            self.fit_warnings.append(
                "these signals were constant across the development set and carry no "
                f"weight: {', '.join(constant)}. If a signal is an internal one, the "
                "serving stack most likely does not expose what it needs."
            )

        self.provenance = {
            "provider": getattr(self.provider, "provider_kind", None),
            "second_provider": getattr(self.second_provider, "provider_kind", None),
            "risk_family": self.config.risk_family,
            "signals": list(self.signal_names),
            "n_signals": len(self.signal_names),
            "n_internal_signals": sum(
                1 for s in ALL_SIGNALS
                if s.kind == SIGNAL_INTERNAL and s.name in self.signal_names
            ),
            "n_items": int(len(y)),
            "label_rate": float(y.mean()),
            "constant_signals": constant,
            "model_calls": int(model_calls),
            "fit_warnings": list(self.fit_warnings),
        }
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(
        self,
        question: str,
        answer: str,
        image: str = "",
        *,
        evidence: Optional[Dict[str, Any]] = None,
        primary_sample: Any = None,
    ) -> RiskResult:
        """Score one item. Raises if the scorer is not fitted."""
        if not self.fitted_:
            raise RuntimeError(
                "scorer is not fitted; call fit() or fit_from_signals() first. "
                "An unfitted pipeline has no calibrated scale, and returning a "
                "number anyway is how a caller ends up trusting a prior."
            )
        started = time.perf_counter()
        warnings: List[str] = []

        if evidence is not None:
            ev = Evidence.from_dict(evidence)
            calls = 0
        else:
            if self.provider is None:
                raise ValueError("no provider attached and no evidence supplied")
            ev, calls = gather_evidence(
                self.provider,
                question,
                answer,
                image,
                risk_family=self.config.risk_family,
                k=self.config.k,
                prompts=self.prompts,
                second_provider=self.second_provider,
                primary_sample=primary_sample,
            )

        report = self._report_for(ev)

        expected = self.signal_names
        actual = report.names()
        if expected and actual != expected:
            raise RuntimeError(
                "signal set changed since fitting.\n"
                f"  fitted on: {expected}\n"
                f"  now      : {actual}\n"
                "  Re-fit, or pin the signal order. Scoring with a different column "
                "order silently applies the wrong weights."
            )

        vector = np.asarray([report.vector()], dtype=np.float64)
        score = float(self.calibrator.predict_proba(vector)[0])

        if self.config.use_fusion and self.fusion is not None and self.fusion.fitted:
            col = self.signal_names.index("resample_consistency")
            fused = float(self.fusion.predict_proba([score], vector[:, [col]])[0])
            score = fused
            warnings.append("fusion head applied to the calibrated score")

        for name in report.missing():
            warnings.append(f"signal {name!r} was unavailable for this item")

        return RiskResult(
            risk_score=round(score, 6),
            is_high_risk=bool(score >= self.config.threshold),
            threshold=float(self.config.threshold),
            risk_family=self.config.risk_family,
            signals=report,
            warnings=warnings,
            model_calls=int(calls),
            latency_ms=int((time.perf_counter() - started) * 1000),
            version=self.config.version,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        """Serialisable scorer state -- JSON only, no pickle."""
        return {
            "format": "msrc-scorer",
            "format_version": 1,
            "config": self.config.to_dict(),
            "signal_names": list(self.signal_names),
            "provenance": dict(self.provenance),
            "calibrator": None
            if not self.fitted_
            else {
                "kind": "RiskCalibrator",
                "l2": self.calibrator.l2,
                "mean": self.calibrator.mean_.tolist(),
                "std": self.calibrator.std_.tolist(),
                "coef": self.calibrator.coef_.tolist(),
            },
            "fusion": None
            if not (self.fusion and self.fusion.fitted)
            else {
                "l2": self.fusion.inner.l2,
                "mean": self.fusion.inner.mean_.tolist(),
                "std": self.fusion.inner.std_.tolist(),
                "coef": self.fusion.inner.coef_.tolist(),
            },
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MSRCPipeline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "msrc-scorer":
            raise ValueError("not an MSRC scorer file")
        config = MSRCConfig.from_dict(payload.get("config", {}))

        calibrator = None
        blob = payload.get("calibrator")
        if blob:
            calibrator = RiskCalibrator(l2=blob.get("l2", config.l2))
            calibrator.mean_ = np.asarray(blob["mean"], dtype=np.float64)
            calibrator.std_ = np.asarray(blob["std"], dtype=np.float64)
            calibrator.coef_ = np.asarray(blob["coef"], dtype=np.float64)
            calibrator.success_ = True

        pipeline = cls(config=config, calibrator=calibrator)
        pipeline.signal_names = list(payload.get("signal_names", []))
        pipeline.provenance = dict(payload.get("provenance") or {})
        pipeline.fit_warnings = list(pipeline.provenance.get("fit_warnings") or [])

        expected = [s.name for s in ALL_SIGNALS if s.applies_to(config.risk_family)]
        if pipeline.signal_names and set(pipeline.signal_names) != set(expected):
            pipeline.fit_warnings.append(
                "the scorer was fitted on a different signal set than this version "
                f"produces.\n  scorer: {pipeline.signal_names}\n  code  : {expected}"
            )
        return pipeline

    def weights(self) -> List[Dict[str, float]]:
        """Fitted weights, largest first. For audit."""
        if not self.fitted_:
            return []
        return self.calibrator.weights(self.signal_names)
