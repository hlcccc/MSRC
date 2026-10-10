"""One collection pass, and exactly which channels it fills.

The signals are only as good as what the collector actually asked for: a channel
the loop never gathered comes back unavailable, and a deployment reading the report
then blames its serving stack for something the collector never requested. These
tests pin the plumbing -- which prompts are asked, and how many model calls the
whole pass costs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msrc.provider import StubProvider, gather_evidence  # noqa: E402


def collect(with_image=True, k=1):
    provider = StubProvider(answers=["No"] * 40)
    return gather_evidence(
        provider,
        question="Is there a car in the image?",
        answer="No, there is no car in the image.",
        image="/tmp/x.png" if with_image else "",
        k=k,
    )


def test_collection_fills_every_channel_a_single_model_can_feed():
    ev, calls = collect()
    assert ev.views and ev.resamples and ev.verifications
    assert ev.ocr_texts, "the image was read, so grounding_check can run"
    assert calls == 4 + 1 + 3 + 1          # views + 1 resample + probes + ocr


def test_collection_without_an_image_skips_the_image_read():
    """A text-only item has nothing to read, and nothing is invented for it."""
    ev, calls = collect(with_image=False)
    assert ev.ocr_texts == []
    assert calls == 4 + 1 + 3


def test_no_second_model_means_no_cross_model_channel():
    """The default must not quietly attach a model nobody asked for."""
    ev, _ = collect()
    assert ev.cross_model == []
