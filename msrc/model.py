"""The fitted parts: a regularised logistic risk map, and a guarded fusion head.

NumPy only, solved by damped Newton, no scikit-learn. Two decisions worth
stating because they are the ones that go wrong:

**The map is fitted, not written.** The signals in :mod:`msrc.signals` are
per-item readings -- "three of four re-phrasings agreed", "mean token
probability 0.72". None of those is a probability of anything. The calibrator is
what turns a vector of them into ``P(risk)``, and it does so by being fitted on
labelled items. That is also why a deployment must bring its own labels: the
signals transfer between models, the coefficients do not.

**The fusion head is guarded.** Stacking a second-stage head on top of the
calibrated score is standard, but an unconstrained head fitted on a development
set where the extra channel carries no signal can come out with a *negative*
weight on the score it is meant to sharpen -- inverting the ranking while
returning a confident number. The guard refuses that fit rather than shipping it.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["RidgeLogistic", "RiskCalibrator", "FusionHead", "sigmoid"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def _as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {arr.shape}")
    return arr


class RidgeLogistic:
    """L2-regularised logistic regression by damped Newton.

    ``coef_[0]`` is the intercept and ``coef_[1:]`` the feature weights, matching
    the layout the exported scorer files use.
    """

    def __init__(self, l2: float = 0.05, max_iter: int = 200, tol: float = 1e-9,
                 grad_tol: float = 1e-6):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        #: Stationarity tolerance. The fit is only "converged" when the gradient of
        #: the mean penalised objective is this close to zero.
        self.grad_tol = float(grad_tol)
        self.coef_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.success_ = False
        self.n_iter_ = 0
        self.grad_norm_ = float("inf")

    # -- internals ------------------------------------------------------
    def _standardize_fit(self, features: np.ndarray) -> None:
        self.mean_ = features.mean(axis=0)
        std = features.std(axis=0)
        # A column that never varies carries no information. Scaling it to unit
        # variance would divide by zero; leaving std at 1 keeps it as a constant
        # the L2 penalty then shrinks toward zero, which is the honest outcome.
        self.std_ = np.where(std < 1e-12, 1.0, std)

    def _standardize_apply(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean_) / self.std_

    @staticmethod
    def _objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
        """**Mean** negative log-likelihood, excluding the penalty.

        The mean and not the sum, because the penalty added to it is a fixed
        ``l2``. On the sum the effective regularisation is ``l2 / n``: the same
        ``--l2 0.05`` would mean something different at every dataset size, and a
        *larger* development set would make the problem *less* regularised. That is
        exactly what happened here -- at 758 development rows the fit was
        effectively unpenalised, the damped Newton stalled after 49 steps, and the
        coefficients it returned put 2.3x the base rate on the data they were
        fitted to. At 220 rows the same code was accidentally fine, which is why
        the defect stayed hidden.
        """
        z = X @ w + b
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    # -- public API -----------------------------------------------------
    def fit(self, features: Any, labels: Any) -> "RidgeLogistic":
        X = _as_2d(features)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if len(X) != len(y):
            raise ValueError("features and labels must have the same length")
        if len(y) == 0:
            raise ValueError("no training rows")
        if len(np.unique(y)) < 2:
            raise ValueError(
                "labels contain only one class; a calibrator fitted on one class "
                "learns a constant and cannot rank anything"
            )

        self._standardize_fit(X)
        Xs = self._standardize_apply(X)
        n, d = Xs.shape

        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        stalled = False
        # Ridge on the weights, not the intercept.
        reg = self.l2 * np.eye(d, dtype=np.float64)

        for step in range(self.max_iter):
            z = Xs @ w + b
            p = sigmoid(z)
            # Gradients and Hessian of the *mean* objective, so that `l2` weights
            # the penalty the same way at any n. See `_objective`.
            grad_w = (Xs.T @ (p - y)) / n + self.l2 * w
            grad_b = float(np.sum(p - y)) / n
            weights = np.maximum(p * (1.0 - p), 1e-12)
            H = (Xs.T @ (Xs * weights[:, None])) / n + reg
            try:
                delta = np.linalg.solve(H, np.column_stack([grad_w, np.full(d, grad_b)]))
            except np.linalg.LinAlgError:
                break
            dw, db = delta[:, 0], float(delta[0, 1])

            # Damped step: halve until the objective does not increase.
            before = self._objective(Xs, y, w, b) + 0.5 * self.l2 * float(w @ w)
            scale = 1.0
            improved = False
            for _ in range(30):
                cand_w = w - scale * dw
                cand_b = b - scale * db
                after = self._objective(Xs, y, cand_w, cand_b) + 0.5 * self.l2 * float(cand_w @ cand_w)
                if after <= before:
                    improved = True
                    break
                scale *= 0.5
            if not improved:
                stalled = True
                break

            shift = max(float(np.max(np.abs(scale * dw))), abs(scale * db))
            w, b = w - scale * dw, b - scale * db
            self.n_iter_ = step + 1
            if shift < self.tol:
                break

        z = Xs @ w + b
        p = sigmoid(z)
        grad = np.concatenate([
            [float(np.sum(p - y)) / n],
            (Xs.T @ (p - y)) / n + self.l2 * w,
        ])
        self.grad_norm_ = float(np.max(np.abs(grad)))
        self.coef_ = np.concatenate([[b], w])

        # Convergence means the gradient reached zero, not that the loop ended with
        # finite numbers in it. The previous version reported success whenever the
        # coefficients were finite, which is true of every run that does not blow
        # up -- including the stalled one above, whose intercept was off by a
        # factor of three and which no caller could have noticed.
        self.success_ = self.grad_norm_ < self.grad_tol
        if not self.success_:
            warnings.warn(
                f"the calibrator did not converge: max |gradient| = "
                f"{self.grad_norm_:.3e} after {self.n_iter_} Newton steps"
                + (" (the damped step stopped improving the objective)" if stalled else "")
                + ". The coefficients are returned as they are, because a stalled "
                "fit still ranks, but the probabilities may be over- or "
                "under-confident. Check the calibration on held-out data before "
                "quoting an ECE.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        X = _as_2d(features)
        if self.coef_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("calibrator is not fitted")
        Xs = self._standardize_apply(X)
        return np.clip(sigmoid(Xs @ self.coef_[1:] + self.coef_[0]), 1e-6, 1.0 - 1e-6)

    def weights(self, names: Sequence[str]) -> List[Dict[str, float]]:
        """Feature weights, sorted by magnitude. For audit, not for decisions."""
        if self.coef_ is None:
            return []
        pairs = sorted(
            zip(names, self.coef_[1:]), key=lambda kv: -abs(float(kv[1]))
        )
        return [{"signal": n, "weight": float(w)} for n, w in pairs]


class RiskCalibrator(RidgeLogistic):
    """The risk map. A named subclass so exported files say what they are."""

    kind = "RiskCalibrator"


class FusionHead:
    """Second-stage head over ``[calibrated_score, *extra]``, with a sign guard.

    The guard exists because of a failure this framework's predecessor shipped: a
    head fitted on a development set whose extra channel carried no signal came
    out with a negative weight on the calibrated score -- meaning "the more likely
    this is wrong, the less I warn about it" -- and returned an inverted ranking
    with no warning attached. A head that contradicts the score it is sharpening
    is never what anyone wants, so it is refused and the calibrated score is
    returned unchanged.
    """

    def __init__(self, l2: float = 0.02):
        self.inner = RidgeLogistic(l2=l2)
        self.fitted = False
        self.refused_reason: str = ""

    def fit(
        self,
        base_scores: Sequence[float],
        extras: Sequence[Sequence[float]],
        labels: Sequence[int],
        extra_names: Optional[Sequence[str]] = None,
    ) -> "FusionHead":
        base = np.asarray(base_scores, dtype=np.float64).reshape(-1, 1)
        extra = np.asarray(extras, dtype=np.float64)
        if extra.ndim == 1:
            extra = extra.reshape(-1, 1)
        X = np.column_stack([base, extra])
        names = ["calibrated_score"] + list(extra_names or [f"extra_{i}" for i in range(extra.shape[1])])

        # Refuse when every extra column is constant: there is nothing to learn
        # from that channel, and fitting anyway is what produces the inversion.
        constant = [n for i, n in enumerate(names[1:], start=1)
                    if float(np.ptp(X[:, i])) <= 1e-12]
        if len(constant) == len(names) - 1:
            self.refused_reason = (
                "every extra signal is constant across the development set "
                f"({', '.join(constant)}); nothing to learn, head not fitted"
            )
            return self

        candidate = RidgeLogistic(l2=self.inner.l2).fit(X, labels)
        base_weight = float(candidate.coef_[1]) if candidate.coef_ is not None else 0.0
        if base_weight < 0.0:
            self.refused_reason = (
                f"head placed a negative weight ({base_weight:+.4f}) on the calibrated "
                "score, which would invert the ranking; not fitted"
            )
            return self

        self.inner = candidate
        self.fitted = True
        self.names = names
        return self

    def predict_proba(self, base_scores: Sequence[float], extras: Sequence[Sequence[float]]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fusion head is not fitted")
        base = np.asarray(base_scores, dtype=np.float64).reshape(-1, 1)
        extra = np.asarray(extras, dtype=np.float64)
        if extra.ndim == 1:
            extra = extra.reshape(-1, 1)
        return self.inner.predict_proba(np.column_stack([base, extra]))
