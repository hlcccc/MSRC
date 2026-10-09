"""One collection pass that can answer both questions.

The framework's whole pitch to a deployment is "give me one input and I will tell
you both whether the answer is grounded and whether it is safe". Until now that was
not true of collection: the policy probe was asked only for the safety family and
the image was read only for the factual one, so a single pass produced evidence for
one head and a hole where the other head's column should be. The dual scorer would
then quietly impute that hole as a zero.

These tests pin the plumbing: `dual=True` collects both channels, the default still
collects exactly one, and the extra cost is the one guard call plus OCR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msrc.provider import StubProvider, gather_evidence  # noqa: E402
from msrc.types import RISK_FACTUAL, RISK_SAFETY  # noqa: E402


def collect(family=RISK_FACTUAL, dual=False, with_image=True, k=1):
    provider = StubProvider(answers=["No"] * 40)
    return gather_evidence(
        provider,
        question="Is there a car in the image?",
        answer="No, there is no car in the image.",
        image="/tmp/x.png" if with_image else "",
        risk_family=family,
        k=k,
        dual=dual,
    )


# ---------------------------------------------------------------------------
# The default is unchanged
# ---------------------------------------------------------------------------

def test_factual_collection_still_skips_the_policy_probe():
    ev, calls = collect(RISK_FACTUAL)
    assert ev.policy_verdicts == []
    assert calls == 4 + 1 + 3 + 1          # views + 1 resample + probes + ocr


def test_safety_collection_still_skips_the_image_read():
    ev, calls = collect(RISK_SAFETY)
    assert ev.ocr_texts == []
    assert ev.policy_verdicts, "the safety family still asks the guard"


# ---------------------------------------------------------------------------
# dual=True turns both on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", [RISK_FACTUAL, RISK_SAFETY])
def test_dual_collection_produces_both_channels_from_one_pass(family):
    ev, _ = collect(family, dual=True)
    assert ev.policy_verdicts, "the guard was asked"
    assert ev.ocr_texts, "the image was read"
    assert ev.views and ev.resamples and ev.verifications


def test_dual_costs_exactly_one_more_call_than_the_family_alone():
    """One guard call and the OCR pass, not a second collection.

    This is the number a deployment cares about: the claim is that answering both
    questions costs one extra model call, not a second run.
    """
    _, factual = collect(RISK_FACTUAL)
    _, safety = collect(RISK_SAFETY)
    _, dual = collect(RISK_FACTUAL, dual=True)
    # dual = factual + the guard call
    assert dual == factual + 1
    # and it is strictly cheaper than collecting the two families separately
    assert dual < factual + safety


def test_dual_without_an_image_still_asks_the_guard():
    """A text-only item has nothing to read but is still a safety question."""
    ev, _ = collect(RISK_FACTUAL, dual=True, with_image=False)
    assert ev.policy_verdicts
    assert ev.ocr_texts == []


# ---------------------------------------------------------------------------
# And the result is usable by both heads
# ---------------------------------------------------------------------------

def test_the_dual_evidence_carries_both_family_specific_columns():
    from msrc.signals import ALL_SIGNALS, build_report
    import dataclasses

    ev, _ = collect(RISK_FACTUAL, dual=True)
    factual = build_report(dataclasses.replace(ev, risk_family=RISK_FACTUAL))
    safety = build_report(dataclasses.replace(ev, risk_family=RISK_SAFETY))

    assert "grounding_check" in factual.names()
    assert "policy_probe" not in factual.names()
    assert "policy_probe" in safety.names()
    assert "grounding_check" not in safety.names()

    # neither head should be reporting its family column as missing
    assert "grounding_check" not in factual.missing()
    assert "policy_probe" not in safety.missing()
