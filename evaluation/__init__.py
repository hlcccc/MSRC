"""Evaluation layer: metrics, dataset adapters, and the reporting helper.

Kept separate from :mod:`msrc` so the decision layer can be imported and used
without the evaluation machinery, and so the metrics can be tested against
textbook cases rather than only against whatever the pipeline happens to produce.
"""

from evaluation.metrics import (
    auroc,
    brier,
    ece,
    format_summary,
    reliability_curve,
    summarise,
    threshold_metrics,
)

__all__ = [
    "auroc",
    "brier",
    "ece",
    "format_summary",
    "reliability_curve",
    "summarise",
    "threshold_metrics",
]
