"""Core typed containers for MSRC.

Everything here is plain dataclasses plus NumPy so the decision layer stays
importable on a CPU-only machine with no model weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Risk families
# ---------------------------------------------------------------------------

#: The two things this framework scores. They are separate because they need
#: different labels and different evidence, but they share the whole decision
#: layer: signals in, calibrated probability out.
RISK_FACTUAL = "factual"      # the answer contradicts the image (hallucination)
RISK_SAFETY = "safety"        # the answer violates a content policy
RISK_FAMILIES = (RISK_FACTUAL, RISK_SAFETY)

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

#: Where a signal comes from. This is not cosmetic: an INTERNAL signal needs the
#: model's own numbers (token logprobs, attention), so a text-only serving stack
#: cannot produce it. Recording the kind lets the pipeline refuse to fit on a
#: channel that a deployment cannot actually feed, instead of silently training
#: on a constant column.
SIGNAL_EXTERNAL = "external"  # derived from the model's text output only
SIGNAL_INTERNAL = "internal"  # requires the model's internals
SIGNAL_KINDS = (SIGNAL_EXTERNAL, SIGNAL_INTERNAL)


@dataclass
class Sample:
    """One generation from the model under evaluation."""

    text: str
    #: Mean token probability of the generated tokens. ``None`` when the serving
    #: stack does not expose logprobs -- which is a supported configuration, not
    #: an error, and is recorded as missing rather than defaulted to a number.
    sequence_confidence: Optional[float] = None
    #: Shannon entropy of the per-step output distributions, averaged. ``None``
    #: for the same reason as above.
    sequence_entropy: Optional[float] = None
    #: Attention mass on image tokens, when the stack exposes attentions.
    visual_attention_mass: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalValue:
    """One signal's reading for one item, plus what it means for risk.

    ``risk`` is oriented so that **higher always means more risk**. Every signal
    implementation is responsible for getting its own direction right; the
    calibrator then only has to weight them, not guess signs. (Getting this wrong
    is the single easiest way to invert a score, so it is a stated invariant
    rather than a convention.)
    """

    name: str
    kind: str            # SIGNAL_EXTERNAL | SIGNAL_INTERNAL
    risk: float          # in [0, 1], higher = riskier
    #: The raw reading before orientation, kept for audit and for reporting.
    raw: float = 0.0
    #: A short human-readable justification, surfaced in the payload.
    detail: str = ""
    #: False when the deployment could not supply what this signal needs.
    available: bool = True

    def __post_init__(self) -> None:
        if self.kind not in SIGNAL_KINDS:
            raise ValueError(f"unknown signal kind: {self.kind!r}")
        if not (0.0 <= self.risk <= 1.0):
            raise ValueError(f"risk must be in [0, 1], got {self.risk!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SignalReport:
    """All signals for one item, in a fixed order."""

    signals: List[SignalValue] = field(default_factory=list)

    def names(self) -> List[str]:
        return [s.name for s in self.signals]

    def available(self) -> List[SignalValue]:
        return [s for s in self.signals if s.available]

    def missing(self) -> List[str]:
        return [s.name for s in self.signals if not s.available]

    def by_name(self, name: str) -> Optional[SignalValue]:
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def vector(self) -> List[float]:
        return [s.risk for s in self.signals]

    def count_by_kind(self) -> Dict[str, int]:
        out = {k: 0 for k in SIGNAL_KINDS}
        for s in self.signals:
            out[s.kind] += 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals": [s.to_dict() for s in self.signals],
            "n_signals": len(self.signals),
            "n_available": len(self.available()),
            "by_kind": self.count_by_kind(),
        }


@dataclass
class RiskResult:
    """What the platform receives."""

    risk_score: float
    is_high_risk: bool
    threshold: float
    risk_family: str
    signals: SignalReport
    warnings: List[str] = field(default_factory=list)
    model_calls: int = 0
    latency_ms: int = 0
    version: str = ""

    @property
    def calibrated_confidence(self) -> float:
        """Complement of ``risk_score``. Derived, so it cannot drift from it."""
        return round(1.0 - self.risk_score, 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "calibrated_confidence": self.calibrated_confidence,
            "is_high_risk": self.is_high_risk,
            "threshold": self.threshold,
            "risk_family": self.risk_family,
            "signals": self.signals.to_dict(),
            "warnings": list(self.warnings),
            "model_calls": self.model_calls,
            "latency_ms": self.latency_ms,
            "version": self.version,
        }


@dataclass
class RiskRequest:
    """One scoring request."""

    question: str
    answer: str
    image: str = ""
    risk_family: str = RISK_FACTUAL
    threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
