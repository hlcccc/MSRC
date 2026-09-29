"""The visual-attention reading, which returned None for a whole 800-item run.

Two independent mistakes did it, and neither raised: the image token id was read
off the generation output, which has no config, and the image span was computed by
subtracting the prompt length from a position that is already absolute in the key
axis. The signals layer then reported the column as unavailable, which is the
correct behaviour for a missing reading and the reason the run "worked" while
producing six signals instead of seven.

The extraction is a pure function of the output, the inputs and two integers, so
it can be tested with synthetic tensors -- no model, no GPU, no weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msrc.providers.hf_llava import HFLLaVAProvider  # noqa: E402

MASS = HFLLaVAProvider._visual_attention_mass

IMAGE_TOKEN_ID = 32000


class _Config:
    def __init__(self, image_token_index=IMAGE_TOKEN_ID):
        self.image_token_index = image_token_index


class _Output:
    """Stands in for the generation output. Note it has no `config`."""

    def __init__(self, attentions):
        self.attentions = attentions


def _attentions(n_steps, n_layers, heads, queries, keys, *, image_span, placeholder):
    """Attention tensors with a known, exact mass on the image span.

    Shaped ``[batch, heads, queries, keys]`` per layer, which is what the model
    returns: the batch axis is why the extraction indexes ``[0]`` after picking a
    layer.
    """
    steps = []
    for _ in range(n_steps):
        layers = []
        for _ in range(n_layers):
            layer = torch.zeros(1, heads, queries, keys)
            # Every query attends uniformly to the image span and nowhere else, so
            # the mass is 1.0 by construction and any other answer is a bug.
            layer[:, :, :, placeholder:placeholder + image_span] = 1.0 / image_span
            layers.append(layer)
        steps.append(tuple(layers))
    return tuple(steps)


def _inputs(prompt_ids):
    return {"input_ids": torch.tensor([prompt_ids])}


def _prompt_with_placeholder(placeholder, length=18):
    ids = [7] * length
    ids[placeholder] = IMAGE_TOKEN_ID
    return ids


def test_the_image_span_is_found_and_its_mass_measured():
    prompt_len = 18
    placeholder = 5
    patches = 576
    total_keys = prompt_len - 1 + patches
    output = _Output(_attentions(4, 2, 3, 1, total_keys,
                                 image_span=patches, placeholder=placeholder))

    mass = MASS(output, _inputs(_prompt_with_placeholder(placeholder, prompt_len)),
                prompt_len, _Config())
    assert mass == pytest.approx(1.0)


def test_the_answer_does_not_depend_on_the_output_carrying_a_config():
    """The first bug: the config was read off the output, which never has one."""
    prompt_len = 18
    placeholder = 3
    patches = 576
    output = _Output(_attentions(2, 1, 2, 1, prompt_len - 1 + patches,
                                 image_span=patches, placeholder=placeholder))
    assert not hasattr(output, "config")
    assert MASS(output, _inputs(_prompt_with_placeholder(placeholder, prompt_len)),
                prompt_len, _Config()) is not None


def test_a_prompt_longer_than_the_placeholder_does_not_blank_the_reading():
    """The second bug: subtracting prompt_len pushed the span below zero.

    With the placeholder early in the prompt and a long prompt, `start` went
    negative and `hi <= lo`, so every step was skipped and the function returned
    None -- which is exactly what the 800-item run saw.
    """
    prompt_len = 40
    placeholder = 4
    patches = 576
    output = _Output(_attentions(3, 1, 2, 1, prompt_len - 1 + patches,
                                 image_span=patches, placeholder=placeholder))
    assert MASS(output, _inputs(_prompt_with_placeholder(placeholder, prompt_len)),
                prompt_len, _Config()) == pytest.approx(1.0)


def test_attention_on_text_only_gives_zero():
    prompt_len = 12
    patches = 576
    total_keys = prompt_len - 1 + patches
    layer = torch.zeros(1, 2, 1, total_keys)
    layer[:, :, :, 0] = 1.0  # all mass on the first text token
    output = _Output(tuple((layer,) for _ in range(2)))
    mass = MASS(output, _inputs(_prompt_with_placeholder(3, prompt_len)),
                prompt_len, _Config())
    assert mass == pytest.approx(0.0)


def test_no_attentions_returns_none():
    prompt_len = 12
    output = _Output(None)
    assert MASS(output, _inputs(_prompt_with_placeholder(3, prompt_len)),
                prompt_len, _Config()) is None
    assert MASS(_Output(()), _inputs(_prompt_with_placeholder(3, prompt_len)),
                prompt_len, _Config()) is None


def test_a_config_without_an_image_token_index_returns_none():
    """A text-only checkpoint has no image span to measure."""
    output = _Output(_attentions(1, 1, 2, 1, 590, image_span=576, placeholder=5))
    assert MASS(output, _inputs(_prompt_with_placeholder(5, 18)), 18,
                _Config(image_token_index=None)) is None
    assert MASS(output, _inputs(_prompt_with_placeholder(5, 18)), 18, None) is None


def test_a_prompt_without_the_placeholder_returns_none():
    output = _Output(_attentions(1, 1, 2, 1, 590, image_span=576, placeholder=5))
    assert MASS(output, _inputs([7] * 18), 18, _Config()) is None


def test_a_span_that_does_not_fit_the_layout_returns_none():
    """Keys equal to the unexpanded prompt mean no patches were spliced in.

    The patch count is `keys - (prompt_len - 1)`; at 18 keys for an 18-token prompt
    that is 1, which is not a LLaVA layout, and one key fewer makes it zero. Either
    way the reading is refused rather than invented.
    """
    layout = _inputs(_prompt_with_placeholder(5, 18))

    zero_patches = _Output(_attentions(1, 1, 2, 1, 17, image_span=1, placeholder=5))
    assert MASS(zero_patches, layout, 18, _Config()) is None

    # One key more is a single "patch", which the function accepts: it cannot tell
    # a 1-patch layout from a real one, and refusing every small image would be
    # worse than measuring what is there.
    one_patch = _Output(_attentions(1, 1, 2, 1, 18, image_span=1, placeholder=5))
    assert MASS(one_patch, layout, 18, _Config()) == pytest.approx(1.0)
