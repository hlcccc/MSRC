"""The first-step margin, tested against the tensor code that produces it.

`DecisionMargin` has its own tests, but those feed it a number. The number comes
from `HFVisionLanguageProvider._from_scores`, which walks the per-step score
tensors, and that code cannot be exercised without torch. Those tests therefore
skip on a CPU-only checkout -- which is exactly how a wrong index or a swapped
top-2 would survive to a collection run.

Run on a machine with torch:

    python -m pytest tests/test_decision_margin_extraction.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="the score-walking code needs torch")

from msrc.providers.hf_vision import HFVisionLanguageProvider  # noqa: E402


def make_output(step_logits):
    """``output.scores`` is a list of ``[batch, vocab]`` tensors, one per step."""
    return SimpleNamespace(scores=[torch.tensor([row], dtype=torch.float32)
                                   for row in step_logits])


def test_the_margin_is_the_gap_between_the_top_two_at_step_one():
    # Step 0 puts most of its mass on token 0; the runner-up is token 1.
    logits = [[3.0, 1.0] + [0.0] * 6]
    out = make_output(logits)
    gen_ids = torch.tensor([0])
    _, _, margin = HFVisionLanguageProvider._from_scores(out, gen_ids, torch)

    probs = torch.softmax(torch.tensor(logits[0]), dim=-1)
    expected = float(probs[0] - probs[1])
    assert margin == pytest.approx(expected, abs=1e-6)
    assert margin > 0.5


def test_a_coin_flip_scores_near_zero():
    logits = [[1.0, 1.0] + [0.0] * 6]
    _, _, margin = HFVisionLanguageProvider._from_scores(
        make_output(logits), torch.tensor([0]), torch)
    assert margin == pytest.approx(0.0, abs=1e-6)


def test_it_reads_step_one_and_not_the_easiest_step():
    """A torn decision followed by confident boilerplate must still read as torn.

    This is the whole reason the signal exists: the mean over both steps stays
    high, and only the margin sees the hesitation.
    """
    torn_then_easy = [[1.0, 1.0] + [0.0] * 6, [9.0, 0.0] + [0.0] * 6]
    confidence, _, margin = HFVisionLanguageProvider._from_scores(
        make_output(torn_then_easy), torch.tensor([0, 0]), torch)
    assert margin == pytest.approx(0.0, abs=1e-6)
    assert confidence > 0.5, "the mean is diluted by the easy second step"


def test_a_later_coin_flip_does_not_move_the_margin():
    """Step 2 being torn is `output_entropy`'s business, not this signal's."""
    easy_then_torn = [[9.0, 0.0] + [0.0] * 6, [1.0, 1.0] + [0.0] * 6]
    _, _, margin = HFVisionLanguageProvider._from_scores(
        make_output(easy_then_torn), torch.tensor([0, 0]), torch)
    assert margin > 0.9


def test_the_runner_up_is_read_not_just_the_winner():
    """Two steps with the same top-1 probability but a different runner-up."""
    close = [[2.0, 1.8] + [0.0] * 6]
    far = [[2.0, 0.0] + [0.0] * 6]
    _, _, m_close = HFVisionLanguageProvider._from_scores(
        make_output(close), torch.tensor([0]), torch)
    _, _, m_far = HFVisionLanguageProvider._from_scores(
        make_output(far), torch.tensor([0]), torch)
    assert m_close < m_far


def test_no_scores_at_all_stays_missing_rather_than_defaulting():
    """A placeholder would make the column constant and the calibrator meaningless."""
    c, e, m = HFVisionLanguageProvider._from_scores(
        SimpleNamespace(scores=None), torch.tensor([0]), torch)
    assert (c, e, m) == (None, None, None)


def test_a_single_token_vocabulary_does_not_invent_a_runner_up():
    """topk(2) on a one-entry distribution cannot produce a pair."""
    out = make_output([[1.0]])
    gen_ids = torch.tensor([0])
    try:
        _, _, margin = HFVisionLanguageProvider._from_scores(out, gen_ids, torch)
    except RuntimeError:
        pytest.skip("topk refused k=2 on a 1-wide vocab; the guard is moot here")
    assert margin is None


def test_the_other_two_readings_are_unchanged():
    """The extra return value must not have disturbed confidence or entropy."""
    logits = [[2.0, 2.0] + [0.0] * 6, [2.0, 2.0] + [0.0] * 6]
    confidence, entropy, _ = HFVisionLanguageProvider._from_scores(
        make_output(logits), torch.tensor([0, 1]), torch)
    probs = torch.softmax(torch.tensor(logits[0]), dim=-1)
    expected_h = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    assert entropy == pytest.approx(expected_h, abs=1e-6)
    assert confidence == pytest.approx(float(probs[0]), abs=1e-6)
    assert math.isfinite(confidence) and math.isfinite(entropy)
