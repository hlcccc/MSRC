"""The signal layer: eight independent readings, five external and three internal.

Every signal obeys two invariants that the rest of the framework relies on:

1. **``risk`` is oriented.** Higher always means more risk. A signal whose raw
   reading runs the other way is responsible for flipping it. The calibrator then
   only has to weight signals, never to discover their signs -- and a sign error,
   which is the fastest way to invert a score, becomes a unit-testable property
   of one class instead of a property of the fitted model.
2. **Missing is not zero.** A signal the deployment cannot supply returns
   ``available=False`` and its name is reported. It is never silently filled with
   a neutral value, because a constant column trains a calibrator that looks fine
   and means nothing.

Provenance, one line each:

* ``self_consistency``   -- SelfCheckGPT (Manakul et al., EMNLP 2023)
* ``resample_consistency`` -- Semantic Entropy (Kuhn et al., ICLR 2024)
* ``typed_verification`` -- typed evidence probing; QACD's direct verification
* ``cross_model_agreement`` -- cross-model / judge agreement
* ``grounding_check``    -- retrieval- and rule-grounded checking (factual only)
* ``policy_probe``       -- policy self-report, Llama-Guard style (safety only)
* ``sequence_confidence`` -- mean token probability; LLM-Check, SAPLMA
* ``output_entropy``     -- mean per-step output entropy
* ``visual_attention``   -- attention mass on image tokens
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from msrc.types import (
    RISK_FACTUAL,
    RISK_SAFETY,
    SIGNAL_EXTERNAL,
    SIGNAL_INTERNAL,
    Sample,
    SignalReport,
    SignalValue,
)

__all__ = [
    "Evidence",
    "Signal",
    "ALL_SIGNALS",
    "build_report",
    "normalize_text",
    "agreement",
    "cluster_entropy",
    "SIGNAL_NAMES",
]

# ---------------------------------------------------------------------------
# Evidence: what the provider gathered, before any signal looks at it
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """Everything a provider collected for one item.

    Signals are pure functions of this object, which is what makes them testable
    without a model: construct an ``Evidence`` by hand and assert on the readings.
    """

    question: str
    answer: str
    image: str = ""
    risk_family: str = RISK_FACTUAL

    #: The generation under evaluation.
    primary: Sample = field(default_factory=lambda: Sample(text=""))

    #: The same question asked K different ways (self-consistency).
    views: List[Sample] = field(default_factory=list)
    #: The identical prompt sampled K times (semantic entropy).
    resamples: List[Sample] = field(default_factory=list)
    #: Typed yes/no probe answers, one per claim.
    verifications: List[Sample] = field(default_factory=list)
    #: A second model's answers to the same question.
    cross_model: List[Sample] = field(default_factory=list)
    #: Free-text policy verdicts (safety) -- "safe" / "unsafe: <category>".
    policy_verdicts: List[Sample] = field(default_factory=list)
    #: Strings read off the image by an OCR engine.
    ocr_texts: List[str] = field(default_factory=list)

    def texts(self, which: str) -> List[str]:
        return [s.text for s in getattr(self, which)]

    # -- serialisation --------------------------------------------------
    # Evidence is collected by running a model, which is the expensive part. It
    # is cached as JSONL so that fitting, threshold selection and evaluation can
    # all be re-run on the same readings without touching a GPU again -- and so
    # that a reviewer can inspect exactly what each signal was computed from.
    def to_dict(self) -> Dict[str, object]:
        def sample(s: Sample) -> Dict[str, object]:
            return {
                "text": s.text,
                "sequence_confidence": s.sequence_confidence,
                "sequence_entropy": s.sequence_entropy,
                "visual_attention_mass": s.visual_attention_mass,
            }

        return {
            "question": self.question,
            "answer": self.answer,
            "image": self.image,
            "risk_family": self.risk_family,
            "primary": sample(self.primary),
            "views": [sample(s) for s in self.views],
            "resamples": [sample(s) for s in self.resamples],
            "verifications": [sample(s) for s in self.verifications],
            "cross_model": [sample(s) for s in self.cross_model],
            "policy_verdicts": [sample(s) for s in self.policy_verdicts],
            "ocr_texts": list(self.ocr_texts),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Evidence":
        def sample(d: Optional[Dict[str, object]]) -> Sample:
            d = d or {}
            return Sample(
                text=str(d.get("text", "")),
                sequence_confidence=d.get("sequence_confidence"),
                sequence_entropy=d.get("sequence_entropy"),
                visual_attention_mass=d.get("visual_attention_mass"),
            )

        return cls(
            question=str(payload.get("question", "")),
            answer=str(payload.get("answer", "")),
            image=str(payload.get("image", "")),
            risk_family=str(payload.get("risk_family", RISK_FACTUAL)),
            primary=sample(payload.get("primary")),
            views=[sample(d) for d in payload.get("views", [])],
            resamples=[sample(d) for d in payload.get("resamples", [])],
            verifications=[sample(d) for d in payload.get("verifications", [])],
            cross_model=[sample(d) for d in payload.get("cross_model", [])],
            policy_verdicts=[sample(d) for d in payload.get("policy_verdicts", [])],
            ocr_texts=[str(t) for t in payload.get("ocr_texts", [])],
        )


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize_text(value: object) -> str:
    """Lower-case, drop articles and punctuation, collapse whitespace.

    Also maps number words to digits, because "two" and "2" are the same answer
    and treating them as a disagreement would inject noise into every
    consistency signal.
    """
    text = str(value or "").strip().lower()
    text = _ARTICLES.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    words = [_NUMBER_WORDS.get(w, w) for w in text.split()]
    return " ".join(words)


def agreement(a: str, b: str) -> float:
    """How much two short answers agree, in ``[0, 1]``.

    Exact match after normalisation scores 1. ``a`` containing ``b`` (or the
    reverse) scores high but not perfect, because "Dakota Digital" and "Dakota"
    are the same answer at different resolutions. Anything else falls back to a
    token-overlap ratio, which is what catches "a red car" against "the car is
    red" without pretending they are identical strings.
    """
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.9
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _mean_agreement(texts: Sequence[str], reference: str) -> float:
    if not texts:
        return 0.0
    return sum(agreement(reference, t) for t in texts) / len(texts)


def cluster_entropy(texts: Sequence[str]) -> float:
    """Shannon entropy of the answers after clustering, normalised to ``[0, 1]``.

    This is the semantic-entropy idea in its simplest form: answers are grouped by
    whether they mean the same thing (normalised string equality here), and the
    entropy of the resulting distribution measures how undecided the model is. A
    model that says the same thing five times has entropy 0; one that gives five
    different answers has entropy 1.

    Normalising by ``log(n)`` makes the value comparable across sample counts,
    which matters because K is a deployment dial -- an unnormalised entropy would
    make a K=5 deployment look more uncertain than a K=3 one for the same model
    behaviour, and the threshold would stop meaning anything.
    """
    import math

    cleaned = [normalize_text(t) for t in texts if normalize_text(t)]
    if len(cleaned) <= 1:
        return 0.0
    counts: Dict[str, int] = {}
    for c in cleaned:
        counts[c] = counts.get(c, 0) + 1
    n = len(cleaned)
    entropy = -sum((k / n) * math.log(k / n) for k in counts.values())
    return entropy / math.log(n) if n > 1 else 0.0


# ---------------------------------------------------------------------------
# Signal protocol
# ---------------------------------------------------------------------------


class Signal:
    """Base class. Subclasses set ``name``/``kind`` as class attributes and
    implement :meth:`compute`.

    Deliberately **not** a dataclass: decorating it would turn ``name``, ``kind``
    and ``families`` into instance fields with defaults, and instantiating a
    subclass would then overwrite the subclass's own class attributes with the
    base defaults -- leaving every signal anonymous and every one reported as
    external. Class attributes are the right mechanism here; there is no
    per-instance state.
    """

    #: Stable identifier. It is the column name in the fitted matrix, so changing
    #: it invalidates a fitted scorer.
    name: str = ""
    #: SIGNAL_EXTERNAL or SIGNAL_INTERNAL.
    kind: str = SIGNAL_EXTERNAL
    #: Which risk families this signal applies to. A signal that only makes sense
    #: for one family is listed for that family only, so the signal *count*
    #: reported for a run is the count of signals that actually ran.
    families: tuple = (RISK_FACTUAL, RISK_SAFETY)
    #: One-line description, surfaced in the payload.
    description: str = ""

    def applies_to(self, family: str) -> bool:
        return family in self.families

    def compute(self, ev: Evidence) -> SignalValue:  # pragma: no cover - interface
        raise NotImplementedError

    def unavailable(self, why: str) -> SignalValue:
        return SignalValue(
            name=self.name, kind=self.kind, risk=0.0, available=False, detail=why
        )


# ---------------------------------------------------------------------------
# External signals
# ---------------------------------------------------------------------------


class SelfConsistency(Signal):
    """Are the answers to re-phrasings of the question the same?

    The classic black-box hallucination detector: a model that knows the answer
    tends to give it whatever way you ask; a model that is making something up
    tends to drift. Costs K extra generations and needs no model internals.
    """

    name = "self_consistency"
    kind = SIGNAL_EXTERNAL
    description = "agreement between the answer and K re-phrasings of the question"

    def compute(self, ev: Evidence) -> SignalValue:
        if not ev.views:
            return self.unavailable("no view generations supplied")
        agree = _mean_agreement(ev.texts("views"), ev.answer)
        return SignalValue(
            name=self.name,
            kind=self.kind,
            risk=1.0 - agree,
            raw=agree,
            detail=f"mean agreement with {len(ev.views)} re-phrasings = {agree:.3f}",
        )


class ResampleConsistency(Signal):
    """How undecided is the model when asked the identical question K times?"""

    name = "resample_consistency"
    kind = SIGNAL_EXTERNAL
    description = "normalised entropy over K samples of the identical prompt"

    def compute(self, ev: Evidence) -> SignalValue:
        if len(ev.resamples) < 2:
            return self.unavailable("needs at least 2 samples of the same prompt")
        entropy = cluster_entropy(ev.texts("resamples"))
        return SignalValue(
            name=self.name,
            kind=self.kind,
            risk=entropy,
            raw=entropy,
            detail=(
                f"{len({normalize_text(t) for t in ev.texts('resamples')})} distinct "
                f"answers in {len(ev.resamples)} samples, normalised entropy = {entropy:.3f}"
            ),
        )


class TypedVerification(Signal):
    """Ask targeted yes/no questions about the answer's claims.

    Supplies its own direction: a probe that answers "no" more often is risk.
    """

    name = "typed_verification"
    kind = SIGNAL_EXTERNAL
    description = "support rate across typed yes/no verification probes"

    _NO = ("no", "false", "incorrect", "unsupported", "contradict", "not")
    _YES = ("yes", "true", "correct", "supported", "support")

    @classmethod
    def _is_support(cls, text: str) -> Optional[bool]:
        t = normalize_text(text)
        if not t:
            return None
        first = t.split()[0]
        if first in cls._NO:
            return False
        if first in cls._YES:
            return True
        return None

    def compute(self, ev: Evidence) -> SignalValue:
        if not ev.verifications:
            return self.unavailable("no verification probes supplied")
        verdicts = [self._is_support(s.text) for s in ev.verifications]
        decided = [v for v in verdicts if v is not None]
        if not decided:
            return self.unavailable("no probe answer could be parsed as yes/no")
        support = sum(decided) / len(decided)
        return SignalValue(
            name=self.name,
            kind=self.kind,
            risk=1.0 - support,
            raw=support,
            detail=(
                f"{sum(decided)}/{len(decided)} probes supported the claim"
                + (f" ({len(verdicts) - len(decided)} unparseable)" if len(decided) != len(verdicts) else "")
            ),
        )


class CrossModelAgreement(Signal):
    """Does a second, independent model give the same answer?

    Borrowed from judge-based evaluation. It is a genuinely different signal from
    self-consistency: sampling one model many times explores its own uncertainty,
    while asking a different model explores disagreement that no amount of
    resampling of the first model would reveal.
    """

    name = "cross_model_agreement"
    kind = SIGNAL_EXTERNAL
    description = "agreement with an independent second model"

    def compute(self, ev: Evidence) -> SignalValue:
        if not ev.cross_model:
            return self.unavailable("no second-model answers supplied")
        agree = _mean_agreement(ev.texts("cross_model"), ev.answer)
        return SignalValue(
            name=self.name,
            kind=self.kind,
            risk=1.0 - agree,
            raw=agree,
            detail=f"agreement with {len(ev.cross_model)} answer(s) from the second model = {agree:.3f}",
        )


class GroundingCheck(Signal):
    """Does the claim appear in the text actually visible in the image?

    Factual family only. For a safety verdict "is this string in the picture" is
    not evidence of anything, so it is not offered there.
    """

    name = "grounding_check"
    kind = SIGNAL_EXTERNAL
    families = (RISK_FACTUAL,)
    description = "whether the answer is supported by text read off the image"

    def compute(self, ev: Evidence) -> SignalValue:
        if not ev.ocr_texts:
            return self.unavailable("no OCR text for this image")
        answer = normalize_text(ev.answer)
        if not answer:
            return self.unavailable("empty answer")
        joined = " ".join(normalize_text(t) for t in ev.ocr_texts)
        if answer in joined:
            risk, why = 0.0, "answer appears verbatim in the image text"
        else:
            tokens = [w for w in answer.split() if len(w) > 2]
            hit = sum(1 for w in tokens if w in joined)
            ratio = hit / len(tokens) if tokens else 0.0
            risk = 1.0 - ratio
            why = f"{hit}/{len(tokens)} content words found in the image text"
        return SignalValue(
            name=self.name, kind=self.kind, risk=risk, raw=1.0 - risk, detail=why
        )


class PolicyProbe(Signal):
    """Ask whether the content violates a stated policy, and how confidently.

    Safety family only. The categories come from the deployment's own policy
    taxonomy -- this framework does not ship one, because "what counts as unsafe"
    is a policy decision, not a technical one, and hard-coding a list would make
    the framework wrong for every deployment but one.
    """

    name = "policy_probe"
    kind = SIGNAL_EXTERNAL
    families = (RISK_SAFETY,)
    description = "policy verdicts from a guard or self-report probe"

    def compute(self, ev: Evidence) -> SignalValue:
        if not ev.policy_verdicts:
            return self.unavailable("no policy verdicts supplied")
        unsafe = 0
        for s in ev.policy_verdicts:
            t = normalize_text(s.text)
            if t.startswith("unsafe") or "violat" in t:
                unsafe += 1
        rate = unsafe / len(ev.policy_verdicts)
        return SignalValue(
            name=self.name,
            kind=self.kind,
            risk=rate,
            raw=rate,
            detail=f"{unsafe}/{len(ev.policy_verdicts)} verdicts flagged a violation",
        )


# ---------------------------------------------------------------------------
# Internal signals
# ---------------------------------------------------------------------------


class SequenceConfidence(Signal):
    """Mean token probability of the generation. Needs logprobs."""

    name = "sequence_confidence"
    kind = SIGNAL_INTERNAL
    description = "mean token probability of the generated answer"

    def compute(self, ev: Evidence) -> SignalValue:
        c = ev.primary.sequence_confidence
        if c is None:
            return self.unavailable(
                "serving stack exposes no token logprobs (internal signal unavailable)"
            )
        c = min(max(float(c), 0.0), 1.0)
        return SignalValue(
            name=self.name,
            kind=self.kind,
            risk=1.0 - c,
            raw=c,
            detail=f"mean token probability = {c:.4f}",
        )


class OutputEntropy(Signal):
    """Mean per-step entropy of the output distribution, normalised.

    Distinct from confidence, and the reason both are kept: a model can put most
    of its mass on one token while the rest of the distribution is nearly flat
    (confident pick, uncertain model), or concentrate heavily everywhere
    (confident throughout). They disagree on real generations, which is exactly
    when having both helps.
    """

    name = "output_entropy"
    kind = SIGNAL_INTERNAL
    description = "mean entropy of the per-step output distribution"

    #: Entropy is in nats. ln(vocab) for a ~32k vocabulary is about 10.4, but
    #: real generations sit far below that; 4 nats is used as the reference so
    #: the normalised value spans a useful range instead of hugging zero.
    REFERENCE_NATS = 4.0

    def compute(self, ev: Evidence) -> SignalValue:
        h = ev.primary.sequence_entropy
        if h is None:
            return self.unavailable(
                "serving stack exposes no per-step distributions (internal signal unavailable)"
            )
        h = max(float(h), 0.0)
        risk = min(h / self.REFERENCE_NATS, 1.0)
        return SignalValue(
            name=self.name,
            kind=self.kind,
            risk=risk,
            raw=h,
            detail=f"mean output entropy = {h:.4f} nats (normalised by {self.REFERENCE_NATS})",
        )


class VisualAttention(Signal):
    """How much attention the generation put on image tokens.

    An answer produced while barely looking at the picture is a candidate
    hallucination regardless of how fluent it is, which is why this is worth an
    internal channel of its own.
    """

    name = "visual_attention"
    kind = SIGNAL_INTERNAL
    description = "attention mass placed on image tokens"

    def compute(self, ev: Evidence) -> SignalValue:
        mass = ev.primary.visual_attention_mass
        if mass is None:
            return self.unavailable(
                "serving stack exposes no attentions (internal signal unavailable)"
            )
        mass = min(max(float(mass), 0.0), 1.0)
        return SignalValue(
            name=self.name,
            kind=self.kind,
            risk=1.0 - mass,
            raw=mass,
            detail=f"attention mass on image tokens = {mass:.4f}",
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Order matters: it fixes the feature order the calibrator is fitted on, so it
#: must not change between fitting and scoring.
ALL_SIGNALS: List[Signal] = [
    SelfConsistency(),
    ResampleConsistency(),
    TypedVerification(),
    CrossModelAgreement(),
    GroundingCheck(),
    PolicyProbe(),
    SequenceConfidence(),
    OutputEntropy(),
    VisualAttention(),
]

SIGNAL_NAMES = [s.name for s in ALL_SIGNALS]


def build_report(ev: Evidence, signals: Optional[Sequence[Signal]] = None) -> SignalReport:
    """Run every signal that applies to this item's risk family, in order."""
    chosen = signals if signals is not None else ALL_SIGNALS
    report = SignalReport()
    for signal in chosen:
        if not signal.applies_to(ev.risk_family):
            continue
        report.signals.append(signal.compute(ev))
    return report
