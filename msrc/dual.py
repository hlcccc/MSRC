"""One input, two risks: hallucination and content safety, scored together.

Why this is a separate object rather than a flag on :class:`MSRCPipeline`
-----------------------------------------------------------------------
The two risk families do not share a label rule, a signal set or a development
half, so they cannot share a calibrator. What they *do* share is everything
upstream: one image, one question, one answer, one collection pass, one evidence
cache. A deployment asked "is this answer trustworthy" needs both answers from that
one pass, not two runs of the model.

The two families differ in exactly one column -- ``grounding_check`` applies only to
factual risk (does the image contain the words the answer claims) and
``policy_probe`` only to safety (did a guard call the response harmful). Everything
else is the same eight-wide design matrix. So both reports are built from the *same*
:class:`~msrc.signals.Evidence` object by overriding only its ``risk_family``
label, which is what :func:`build_report` gates on. No new evidence, no second
model call, no second collection pass.

What this does not do
---------------------
It does not merge the two scores into one number. They are not commensurable: a
0.9 hallucination risk and a 0.9 safety risk mean different things, were fitted
against different labels, and are compared to thresholds chosen on different
development halves. :class:`DualRisk` therefore reports both, and the caller decides
what to do when either fires -- blocking on safety and flagging on hallucination is
a policy decision, and this module does not make it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from msrc.model import RiskCalibrator
from msrc.signals import ALL_SIGNALS, Evidence, SignalReport, build_report
from msrc.types import RISK_FACTUAL, RISK_FAMILIES, RISK_SAFETY

__all__ = ["DualRisk", "DualRiskScorer", "signal_names_for"]


def signal_names_for(family: str) -> List[str]:
    """The columns a family uses, in the order its calibrator was fitted on."""
    if family not in RISK_FAMILIES:
        raise ValueError(f"unknown risk family: {family!r}")
    return [s.name for s in ALL_SIGNALS if s.applies_to(family)]


@dataclasses.dataclass
class DualRisk:
    """Both risks for one item, with the evidence for each kept separate."""

    hallucination_risk: float
    safety_risk: float
    hallucination_warning: bool
    safety_warning: bool
    #: Which of the two heads could actually be computed. A head whose development
    #: data was never fitted is reported as unavailable rather than as 0.0, because
    #: 0.0 reads as "we looked and it is fine".
    hallucination_available: bool = True
    safety_available: bool = True
    #: Signals that produced no reading, per head. Their columns were imputed.
    hallucination_missing: Sequence[str] = ()
    safety_missing: Sequence[str] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hallucination_risk": self.hallucination_risk,
            "safety_risk": self.safety_risk,
            "hallucination_warning": self.hallucination_warning,
            "safety_warning": self.safety_warning,
            "hallucination_available": self.hallucination_available,
            "safety_available": self.safety_available,
            "hallucination_missing_signals": list(self.hallucination_missing),
            "safety_missing_signals": list(self.safety_missing),
            "warning": self.hallucination_warning or self.safety_warning,
        }

    def __str__(self) -> str:  # pragma: no cover - presentation
        def head(name, p, warn, ok, missing):
            if not ok:
                return "%-14s unavailable (not fitted)" % name
            flag = "WARN" if warn else "ok"
            extra = ("  missing: " + ", ".join(missing)) if missing else ""
            return "%-14s %.4f  %s%s" % (name, p, flag, extra)

        return "\n".join([
            head("hallucination", self.hallucination_risk, self.hallucination_warning,
                 self.hallucination_available, self.hallucination_missing),
            head("safety", self.safety_risk, self.safety_warning,
                 self.safety_available, self.safety_missing),
        ])


class DualRiskScorer:
    """Two fitted heads over one shared evidence object.

    Fit with :meth:`fit`, which takes one labelled set per family because that is
    how the data exists -- POPE and HallusionBench for hallucination, MM-SafetyBench
    and SafeBench for safety. Each head is fitted on its own development rows and
    gets its own threshold, chosen on development only, for the same reason.
    """

    def __init__(
        self,
        factual: Optional[RiskCalibrator] = None,
        safety: Optional[RiskCalibrator] = None,
        factual_threshold: float = 0.5,
        safety_threshold: float = 0.5,
    ):
        self.factual = factual
        self.safety = safety
        self.factual_threshold = float(factual_threshold)
        self.safety_threshold = float(safety_threshold)
        self.factual_names: List[str] = signal_names_for(RISK_FACTUAL)
        self.safety_names: List[str] = signal_names_for(RISK_SAFETY)
        self.provenance: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    @staticmethod
    def matrix(evidence_rows: Sequence[Dict[str, Any]], family: str) -> np.ndarray:
        """The design matrix for one family from cached evidence records.

        ``evidence_rows`` are records carrying an ``evidence`` dict -- the format
        the collector writes. The family is applied by overriding the evidence's own
        ``risk_family``, exactly as at scoring time, so fit and score cannot end up
        on different column sets.
        """
        out = []
        for row in evidence_rows:
            ev = Evidence.from_dict(row["evidence"])
            out.append(build_report(dataclasses.replace(ev, risk_family=family)).vector())
        if not out:
            raise ValueError("no evidence rows to build a matrix from")
        return np.asarray(out, dtype=np.float64)

    def fit(
        self,
        factual_rows: Optional[Sequence[Dict[str, Any]]] = None,
        safety_rows: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        l2: float = 0.05,
        label_key: str = "label",
    ) -> "DualRiskScorer":
        """Fit whichever heads have data. At least one is required."""
        if factual_rows is None and safety_rows is None:
            raise ValueError("nothing to fit: supply factual_rows, safety_rows or both")

        if factual_rows:
            X = self.matrix(factual_rows, RISK_FACTUAL)
            y = np.asarray([int(r[label_key]) for r in factual_rows], dtype=np.int64)
            if len(np.unique(y)) < 2:
                raise ValueError("the factual rows carry a single class; nothing to fit")
            self.factual = RiskCalibrator(l2=l2).fit(X, y)
            if not self.factual.success_:
                raise RuntimeError(
                    "the factual calibrator did not converge; refusing to export it "
                    "(max |gradient| = %.3e)" % self.factual.grad_norm_)
            self.factual_names = signal_names_for(RISK_FACTUAL)
            self.provenance["factual"] = {
                "n": len(y), "positive_rate": float(y.mean()),
                "signal_names": list(self.factual_names), "l2": l2,
            }

        if safety_rows:
            X = self.matrix(safety_rows, RISK_SAFETY)
            y = np.asarray([int(r[label_key]) for r in safety_rows], dtype=np.int64)
            if len(np.unique(y)) < 2:
                raise ValueError("the safety rows carry a single class; nothing to fit")
            self.safety = RiskCalibrator(l2=l2).fit(X, y)
            if not self.safety.success_:
                raise RuntimeError(
                    "the safety calibrator did not converge; refusing to export it "
                    "(max |gradient| = %.3e)" % self.safety.grad_norm_)
            self.safety_names = signal_names_for(RISK_SAFETY)
            self.provenance["safety"] = {
                "n": len(y), "positive_rate": float(y.mean()),
                "signal_names": list(self.safety_names), "l2": l2,
            }
        return self

    def choose_thresholds(self, factual_rows=None, safety_rows=None) -> "DualRiskScorer":
        """Pick each head's threshold on its own development rows, by accuracy."""
        for rows, cal, family, attr in (
            (factual_rows, self.factual, RISK_FACTUAL, "factual_threshold"),
            (safety_rows, self.safety, RISK_SAFETY, "safety_threshold"),
        ):
            if not rows or cal is None:
                continue
            scores = self._scores(rows, family)
            y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
            grid = np.round(np.linspace(0.05, 0.95, 91), 4)
            accs = np.array([float(((scores >= t).astype(int) == y).mean()) for t in grid])
            # ties go to the higher threshold: warning less is the conservative side
            setattr(self, attr, float(grid[len(grid) - 1 - int(accs[::-1].argmax())]))
        return self

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _scores(self, rows: Sequence[Dict[str, Any]], family: str) -> np.ndarray:
        X = self.matrix(rows, family)
        cal = self.factual if family == RISK_FACTUAL else self.safety
        if cal is None:
            raise RuntimeError(f"the {family} head is not fitted")
        return cal.predict_proba(X)

    def score(self, ev: Evidence) -> DualRisk:
        """Both risks for one item, from one evidence object."""
        result: Dict[str, Any] = {}
        for family, cal, prefix in ((RISK_FACTUAL, self.factual, "hallucination"),
                                    (RISK_SAFETY, self.safety, "safety")):
            report = build_report(dataclasses.replace(ev, risk_family=family))
            result[prefix + "_missing"] = tuple(report.missing())
            if cal is None:
                result[prefix + "_risk"] = float("nan")
                result[prefix + "_available"] = False
            else:
                result[prefix + "_risk"] = float(cal.predict_proba([report.vector()])[0])
                result[prefix + "_available"] = True

        return DualRisk(
            hallucination_risk=result["hallucination_risk"],
            safety_risk=result["safety_risk"],
            hallucination_warning=bool(result["hallucination_available"]
                                       and result["hallucination_risk"] >= self.factual_threshold),
            safety_warning=bool(result["safety_available"]
                                and result["safety_risk"] >= self.safety_threshold),
            hallucination_available=result["hallucination_available"],
            safety_available=result["safety_available"],
            hallucination_missing=result["hallucination_missing"],
            safety_missing=result["safety_missing"],
        )

    def score_rows(self, rows: Sequence[Dict[str, Any]]) -> List[DualRisk]:
        return [self.score(Evidence.from_dict(r["evidence"])) for r in rows]

    # ------------------------------------------------------------------
    # audit and persistence
    # ------------------------------------------------------------------
    def weights(self) -> Dict[str, List[Dict[str, float]]]:
        out = {}
        if self.factual is not None:
            out["hallucination"] = self.factual.weights(self.factual_names)
        if self.safety is not None:
            out["safety"] = self.safety.weights(self.safety_names)
        return out

    def state(self) -> Dict[str, Any]:
        def blob(cal, names, threshold):
            if cal is None or cal.coef_ is None:
                return None
            return {
                "kind": "RiskCalibrator", "l2": cal.l2,
                "mean": cal.mean_.tolist(), "std": cal.std_.tolist(),
                "coef": cal.coef_.tolist(), "signal_names": list(names),
                "converged": bool(cal.success_), "grad_norm": float(cal.grad_norm_),
                "threshold": float(threshold),
            }

        return {
            "format": "msrc-dual-scorer",
            "format_version": 1,
            "provenance": dict(self.provenance),
            "hallucination": blob(self.factual, self.factual_names, self.factual_threshold),
            "safety": blob(self.safety, self.safety_names, self.safety_threshold),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DualRiskScorer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "msrc-dual-scorer":
            raise ValueError("not an MSRC dual scorer file")

        def revive(blob):
            if not blob:
                return None, 0.5, []
            cal = RiskCalibrator(l2=blob.get("l2", 0.05))
            cal.mean_ = np.asarray(blob["mean"], dtype=np.float64)
            cal.std_ = np.asarray(blob["std"], dtype=np.float64)
            cal.coef_ = np.asarray(blob["coef"], dtype=np.float64)
            cal.success_ = bool(blob.get("converged", True))
            cal.grad_norm_ = float(blob.get("grad_norm", 0.0))
            return cal, float(blob.get("threshold", 0.5)), list(blob.get("signal_names") or [])

        factual, f_thr, f_names = revive(payload.get("hallucination"))
        safety, s_thr, s_names = revive(payload.get("safety"))
        scorer = cls(factual, safety, f_thr, s_thr)
        scorer.provenance = dict(payload.get("provenance") or {})
        if f_names:
            expected = signal_names_for(RISK_FACTUAL)
            if f_names != expected:
                scorer.provenance.setdefault("warnings", []).append(
                    "hallucination head was fitted on %s, this version produces %s"
                    % (f_names, expected))
            scorer.factual_names = f_names
        if s_names:
            expected = signal_names_for(RISK_SAFETY)
            if s_names != expected:
                scorer.provenance.setdefault("warnings", []).append(
                    "safety head was fitted on %s, this version produces %s"
                    % (s_names, expected))
            scorer.safety_names = s_names
        return scorer
