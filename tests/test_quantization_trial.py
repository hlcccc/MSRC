from types import SimpleNamespace

import pytest

from msrc.providers.hf_vision import HFInternVLProvider
from msrc.providers.quantization import audit_quantization, quantization_config, SKIP_MODULES


def test_original_provider_stays_unquantized_by_default():
    provider = HFInternVLProvider()
    assert provider.quantization is None
    assert provider.quantization_audit is None
    with pytest.raises(ValueError, match='quantization'):
        HFInternVLProvider(quantization='int4')


@pytest.mark.parametrize('model,dtype,device,mode', [
    ('Unknown', 'bfloat16', 'cuda:0', 'bnb_nf4'),
    ('InternVLForConditionalGeneration', 'float16', 'cuda:0', 'bnb_nf4'),
    ('InternVLForConditionalGeneration', 'bfloat16', 'cpu', 'bnb_nf4'),
    ('InternVLForConditionalGeneration', 'bfloat16', 'cuda:0', 'other'),
])
def test_unsupported_quantization_fails_before_loading(model, dtype, device, mode):
    with pytest.raises(ValueError, match='NF4 trial'):
        quantization_config(model, dtype, device, mode)


def test_quantization_configuration_preserves_visual_modules(monkeypatch):
    torch = pytest.importorskip('torch')
    import transformers
    monkeypatch.setattr(transformers, 'BitsAndBytesConfig', lambda **kwargs: kwargs, raising=False)
    for model, excluded in SKIP_MODULES.items():
        result = quantization_config(model, 'bfloat16', 'cuda:0')
        assert result['load_in_4bit']
        assert result['bnb_4bit_quant_type'] == 'nf4'
        assert result['bnb_4bit_compute_dtype'] is torch.bfloat16
        assert result['llm_int8_skip_modules'] == list(excluded)


class Linear4:
    compute_dtype = 'bfloat16'
    weight = SimpleNamespace(quant_type='nf4')


def model_with(names, device='0'):
    return SimpleNamespace(
        named_modules=lambda: [(name, Linear4()) for name in names],
        hf_device_map={'': device}, parameters=lambda: [SimpleNamespace(dtype='bfloat16')],
        get_memory_footprint=lambda: 1234)


def test_quantization_audit_checks_actual_modules_not_just_requested_flags():
    audit = audit_quantization(model_with(['model.language_model.layers.0.q_proj']), Linear4)
    assert audit['converted_linear_count'] == 1
    assert not audit['integer_matmul_claim']
    for names in [[], ['model.vision_tower.q_proj'], ['lm_head'],
                  ['model.language_model.layers.0.q_proj', 'model.multi_modal_projector.1']]:
        with pytest.raises(ValueError):
            audit_quantization(model_with(names), Linear4)
    with pytest.raises(ValueError, match='offload'):
        audit_quantization(model_with(['model.language_model.layers.0.q_proj'], 'cpu'), Linear4)


def test_probe_cohort_uses_no_test_items_or_historical_answers():
    from pipeline.quantization_probe import cohort
    rows = [dict(id=str(i), question='x' * (i + 1), acceptance_split='dev') for i in range(20)]
    rows += [dict(id='test', question='longest' * 100, acceptance_split='test')]
    selected = cohort(rows, 4)
    assert all(row['acceptance_split'] == 'dev' for row in selected)
    assert selected[-1]['id'] == '19'
    assert len(cohort(rows, 4, longest=False)) == 4
    with pytest.raises(ValueError):
        cohort([dict(row, answer='old') for row in rows], 4)
    with pytest.raises(ValueError):
        cohort(rows[:1], 4)
