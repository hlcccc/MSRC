"""MSRC — multi-signal risk calibration for multimodal generation.

Two risk families, one decision layer:

    evidence  ->  signals  ->  calibrated risk
    (model)      (9 readings)   (fitted map)

``factual``  scores whether an answer contradicts the image (hallucination).
``safety``   scores whether an answer violates a stated content policy.

Both use the same nine signals and the same calibrator; what differs is the
label, the evidence gathered, and -- for the two family-specific signals -- which
one applies. A deployment brings its own labels and, for the safety family, its
own policy taxonomy: "what counts as unsafe" is a policy decision, and a framework
that hard-coded an answer to it would be wrong everywhere except one place.
"""

from msrc.conformal import (
    VALIDATED,
    benjamini_hochberg,
    benjamini_yekutieli,
    conformal_pvalues,
    select,
    selective_report,
)
from msrc.model import FusionHead, RiskCalibrator, RidgeLogistic
from msrc.pipeline import DEFAULT_THRESHOLD, MSRCConfig, MSRCPipeline
from msrc.provider import ModelProvider, PromptSet, StubProvider, gather_evidence
from msrc.signals import ALL_SIGNALS, SIGNAL_NAMES, Evidence, build_report
from msrc.types import (
    RISK_FACTUAL,
    RISK_FAMILIES,
    RISK_SAFETY,
    SIGNAL_EXTERNAL,
    SIGNAL_INTERNAL,
    RiskRequest,
    RiskResult,
    Sample,
    SignalReport,
    SignalValue,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_SIGNALS",
    "DEFAULT_THRESHOLD",
    "Evidence",
    "FusionHead",
    "MSRCConfig",
    "MSRCPipeline",
    "ModelProvider",
    "PromptSet",
    "RISK_FACTUAL",
    "RISK_FAMILIES",
    "RISK_SAFETY",
    "RiskCalibrator",
    "RiskRequest",
    "RiskResult",
    "RidgeLogistic",
    "SIGNAL_EXTERNAL",
    "SIGNAL_INTERNAL",
    "SIGNAL_NAMES",
    "Sample",
    "SignalReport",
    "SignalValue",
    "StubProvider",
    "VALIDATED",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "build_report",
    "conformal_pvalues",
    "gather_evidence",
    "select",
    "selective_report",
    "__version__",
]
