"""Opt-in language-only NF4 storage with BF16 compute, not an INT4 GEMM claim."""
from __future__ import annotations

from collections import Counter


SKIP_MODULES = {
    'InternVLForConditionalGeneration': (
        'model.vision_tower', 'model.multi_modal_projector', 'lm_head'),
    'Qwen2_5_VLForConditionalGeneration': ('model.visual', 'lm_head'),
}


def quantization_config(model_class, dtype, device, mode='bnb_nf4'):
    if (mode != 'bnb_nf4' or model_class not in SKIP_MODULES
            or dtype != 'bfloat16' or not str(device).startswith('cuda')):
        raise ValueError('NF4 trial requires a supported VLM, BF16 compute and CUDA')
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=list(SKIP_MODULES[model_class]),
    )


def audit_quantization(model, linear_class=None):
    """Fail closed if conversion touches vision, projection, or the output head."""
    if linear_class is None:
        from bitsandbytes.nn import Linear4bit
        linear_class = Linear4bit
    converted = []
    quant_types = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_class):
            if not name.startswith('model.language_model.'):
                raise ValueError('NF4 conversion escaped the language model: ' + name)
            if str(getattr(module, 'compute_dtype', None)) not in ('torch.bfloat16', 'bfloat16'):
                raise ValueError('NF4 compute dtype must be BF16: ' + name)
            converted.append(name)
            quant_types.add(getattr(module.weight, 'quant_type', None))
    if not converted or quant_types != {'nf4'}:
        raise ValueError('expected actual NF4 language-layer conversion')
    devices = {str(value) for value in getattr(model, 'hf_device_map', {}).values()}
    if any(value in ('cpu', 'disk') for value in devices):
        raise ValueError('CPU/disk offload is not part of the GPU speed comparison')
    return dict(mode='bnb_nf4', weight_storage_bits=4, activation_compute='bfloat16',
        integer_matmul_claim=False, vision_projector_and_output_head_quantized=False,
        converted_linear_count=len(converted), converted_linear_names=converted,
        parameter_dtype_counts=dict(Counter(str(p.dtype) for p in model.parameters())),
        model_footprint_bytes=model.get_memory_footprint(), device_map=sorted(devices))
