from types import SimpleNamespace

import pytest

from msrc.batch_collection import collect_chunk, primary_prompt
from msrc.providers.batched import score_readings, trim_at_eos
from msrc.providers.hf_vision import HFInternVLProvider
from msrc.signals import Evidence, build_report
from msrc.types import Sample


def provider():
    return HFInternVLProvider(model_fn=lambda image, prompt, sampled: Sample(
        text='yes', sequence_confidence=.8, sequence_entropy=.4, first_token_margin=.7))


def rows():
    return [dict(id=str(i), image='/image', question='Is this a cat?', gold_answers=['yes'],
                 answer_format='yes_no', answer_parser='pope_official') for i in range(3)]


def test_eos_trimming_includes_first_eos_but_not_batch_padding():
    assert trim_at_eos([3, 2, 2, 2], {2}) == [3, 2]
    assert trim_at_eos([3, 4], {2}) == [3, 4]
    assert trim_at_eos([2, 8], {2, 9}) == [2]


def test_batch_evidence_counts_and_routing():
    model = provider()
    model.ocr = lambda image: ['cat']
    result = collect_chunk(model, rows(), batch_size=2)
    assert model.calls == 3 * 11
    for row in result:
        evidence = Evidence.from_dict(row['evidence'])
        assert len(evidence.views) == 4 and len(evidence.resamples) == 3 and len(evidence.verifications) == 3
        assert row['model_calls'] == 12 and row['label'] == 0
        assert evidence.primary.sequence_confidence == .8


def test_donor_does_not_collect_unused_signals():
    model = provider()
    result = collect_chunk(model, rows(), batch_size=2, donor_only=True)
    assert model.calls == 3
    assert all('evidence' not in row and row['donor_only'] for row in result)


def test_fresh_inputs_only_and_prompt_protocol():
    assert primary_prompt(rows()[0]) == rows()[0]['question']
    assert primary_prompt(dict(question='What?')) == 'What?\nAnswer with a short phrase only.'
    with pytest.raises(ValueError):
        collect_chunk(provider(), [dict(rows()[0], answer='old')], batch_size=2)
    with pytest.raises(ValueError):
        collect_chunk(provider(), [rows()[0]] * 2, batch_size=2)


def test_scores_do_not_include_other_rows_padding():
    torch = pytest.importorskip('torch')
    scores = [torch.tensor([[1., 2., 3.], [3., 2., 1.]]),
              torch.tensor([[3., 1., 2.], [1., 3., 2.]]),
              torch.tensor([[1., 2., 3.], [1., 2., 3.]])]
    tokens = [[2], [0, 1, 2]]
    result = score_readings(scores, tokens, torch)
    expected = scores[0][0].softmax(-1)[2].item()
    assert result[0][0] == pytest.approx(expected)
    expected = sum(scores[i][1].softmax(-1)[token].item() for i, token in enumerate(tokens[1])) / 3
    assert result[1][0] == pytest.approx(expected)


def test_missing_scores_stay_missing():
    assert score_readings(None, [[2], [3, 2]], None) == [(None, None, None)] * 2


def test_ocr_failure_remains_unavailable_as_in_serial_collection():
    model = provider()
    def unavailable(image):
        raise RuntimeError('OCR unavailable')
    model.ocr = unavailable
    result = collect_chunk(model, rows()[:1], batch_size=1)[0]
    assert Evidence.from_dict(result['evidence']).ocr_texts == []
    assert result['collection_warnings'] == [dict(stage='ocr', error='RuntimeError')]
    assert result['model_calls'] == 11
