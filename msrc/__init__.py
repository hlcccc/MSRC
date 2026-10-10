"""MSRC — multi-signal risk calibration for multimodal generation.

One risk, one decision layer:

    evidence  ->  signals  ->  calibrated risk
    (model)      (8 readings)  (fitted map)

The risk is **factual hallucination**: does the answer contradict the image? It is
scored from eight signals -- three internal and five external -- by a calibrator
fitted on the deployment's own labels. Content-safety risk is a different question
with a different label and different evidence, and it is modelled elsewhere.
"""

from msrc.conformal import (
    VALIDATED,
    benjamini_hochberg,
    benjamini_yekutieli,
    conformal_pvalues,
    select,
    selective_report,
)
from msrc.gpu import cap_gpu_memory
from msrc.model import FusionHead, RiskCalibrator, RidgeLogistic
from msrc.pipeline import DEFAULT_THRESHOLD, MSRCConfig, MSRCPipeline
from msrc.provider import ModelProvider, PromptSet, StubProvider, gather_evidence
from msrc.scorer import RiskScore, RiskScorer
from msrc.signals import (
    ALL_SIGNALS,
    SIGNAL_NAMES,
    Evidence,
    build_report,
)
from msrc.types import (
    RISK_FACTUAL,
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
    "RiskCalibrator",
    "RiskRequest",
    "RiskResult",
    "RiskScore",
    "RiskScorer",
    "RidgeLogistic",
    "cap_gpu_memory",
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
