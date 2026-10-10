"""One input, one risk: a calibrator fitted over cached evidence rows.

Why this is a separate object rather than a flag on :class:`MSRCPipeline`
-----------------------------------------------------------------------
The pipeline scores one item at a time, gathering its evidence on the way. This
object fits a calibrator on evidence that was *already* collected -- the rows
``scripts/collect_evidence.py`` writes -- and then scores those rows, or new
evidence, with no model in the loop. That is what makes an evaluation re-runnable
on a CPU and what lets a reviewer check a score against the readings behind it.

The two must agree on the column order, which is why the matrix builder and the
scorer live together here: fit and score cannot end up on different signal sets.

What a score is
---------------
``risk`` is the calibrated probability that the answer contradicts the image, and
it is reported next to the evidence that produced it rather than merged into one
opaque number. A warning is a threshold on that probability; the threshold is
chosen on development rows by :meth:`RiskScorer.choose_threshold`, because a
threshold picked on the rows being reported is not reproducible.

An unfitted scorer reports itself as unavailable rather than as 0.0. 0.0 reads as
"we looked and it is fine", which is the opposite of "nothing was fitted".
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from msrc.model import RiskCalibrator
from msrc.signals import SIGNAL_NAMES, Evidence, build_report

__all__ = ["RiskScore", "RiskScorer"]


@dataclasses.dataclass
class RiskScore:
    """The calibrated risk for one item, with the evidence behind it."""

    risk: float
    warning: bool
    #: False when no calibrator has been fitted. ``risk`` is then NaN rather than
    #: 0.0, because 0.0 reads as a clean bill of health.
    available: bool = True
    #: Signals that produced no reading for this item. Their columns were imputed.
    missing: Sequence[str] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk": self.risk,
            "warning": self.warning,
            "available": self.available,
            "missing_signals": list(self.missing),
        }

    def __str__(self) -> str:  # pragma: no cover - presentation
        if not self.available:
            return "%-14s unavailable (not fitted)" % "hallucination"
        flag = "WARN" if self.warning else "ok"
        extra = ("  missing: " + ", ".join(self.missing)) if self.missing else ""
        return "%-14s %.4f  %s%s" % ("hallucination", self.risk, flag, extra)


class RiskScorer:
    """One fitted calibrator over cached evidence rows.

    Fit with :meth:`fit`, which takes labelled rows in the format the collector
    writes. Give it a threshold with :meth:`choose_threshold` on development rows
    only, for the same reason the fit itself is on development rows.
    """

    def __init__(
        self,
        calibrator: Optional[RiskCalibrator] = None,
        threshold: float = 0.5,
    ):
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.signal_names: List[str] = list(SIGNAL_NAMES)
        self.provenance: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    @staticmethod
    def matrix(evidence_rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        """The design matrix from cached evidence records.

        ``evidence_rows`` are records carrying an ``evidence`` dict -- the format
        the collector writes. The columns come from :func:`build_report`, exactly
        as at scoring time, so fit and score cannot end up on different column
        sets.
        """
        out = []
        for row in evidence_rows:
            ev = Evidence.from_dict(row["evidence"])
            out.append(build_report(ev).vector())
        if not out:
            raise ValueError("no evidence rows to build a matrix from")
        return np.asarray(out, dtype=np.float64)

    def fit(
        self,
        rows: Sequence[Dict[str, Any]],
        *,
        l2: float = 0.05,
        label_key: str = "label",
    ) -> "RiskScorer":
        """Fit the calibrator on labelled evidence rows."""
        if not rows:
            raise ValueError("nothing to fit: supply evidence rows")

        X = self.matrix(rows)
        y = np.asarray([int(r[label_key]) for r in rows], dtype=np.int64)
        if len(np.unique(y)) < 2:
            raise ValueError("the rows carry a single class; nothing to fit")
        self.calibrator = RiskCalibrator(l2=l2).fit(X, y)
        if not self.calibrator.success_:
            raise RuntimeError(
                "the calibrator did not converge; refusing to export it "
                "(max |gradient| = %.3e)" % self.calibrator.grad_norm_)
        self.signal_names = list(SIGNAL_NAMES)
        self.provenance["fit"] = {
            "n": len(y), "positive_rate": float(y.mean()),
            "signal_names": list(self.signal_names), "l2": l2,
        }
        return self

    def choose_threshold(self, rows: Sequence[Dict[str, Any]]) -> "RiskScorer":
        """Pick the threshold on these development rows, by accuracy.

        The grid is 0.05 to 0.95 in steps of 0.01, and ties go to the higher
        threshold: warning less is the conservative side.
        """
        if not rows or self.calibrator is None:
            return self
        scores = self._scores(rows)
        y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
        grid = np.round(np.linspace(0.05, 0.95, 91), 4)
        accs = np.array([float(((scores >= t).astype(int) == y).mean()) for t in grid])
        # ties go to the higher threshold: warning less is the conservative side
        self.threshold = float(grid[len(grid) - 1 - int(accs[::-1].argmax())])
        return self

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _scores(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("the scorer is not fitted")
        return self.calibrator.predict_proba(self.matrix(rows))

    def score(self, ev: Evidence) -> RiskScore:
        """The calibrated risk for one item, from one evidence object."""
        report = build_report(ev)
        missing = tuple(report.missing())
        if self.calibrator is None:
            return RiskScore(risk=float("nan"), warning=False,
                             available=False, missing=missing)
        risk = float(self.calibrator.predict_proba([report.vector()])[0])
        return RiskScore(
            risk=risk,
            warning=bool(risk >= self.threshold),
            available=True,
            missing=missing,
        )

    def score_rows(self, rows: Sequence[Dict[str, Any]]) -> List[RiskScore]:
        return [self.score(Evidence.from_dict(r["evidence"])) for r in rows]

    # ------------------------------------------------------------------
    # audit and persistence
    # ------------------------------------------------------------------
    def weights(self) -> List[Dict[str, float]]:
        if self.calibrator is None:
            return []
        return self.calibrator.weights(self.signal_names)

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
            "format": "msrc-risk-scorer",
            "format_version": 1,
            "provenance": dict(self.provenance),
            "hallucination": blob(self.calibrator, self.signal_names, self.threshold),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RiskScorer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "msrc-risk-scorer":
            raise ValueError("not an MSRC risk scorer file")

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

        calibrator, threshold, names = revive(payload.get("hallucination"))
        scorer = cls(calibrator, threshold)
        scorer.provenance = dict(payload.get("provenance") or {})
        if names:
            expected = list(SIGNAL_NAMES)
            if names != expected:
                scorer.provenance.setdefault("warnings", []).append(
                    "the scorer was fitted on %s, this version produces %s"
                    % (names, expected))
            scorer.signal_names = names
        return scorer
