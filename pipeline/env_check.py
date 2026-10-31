"""Exercise the actual Python, vision, OCR and CUDA runtime before downloading models."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def pinned_versions():
    pins = {}
    path = Path(__file__).resolve().parents[1] / 'requirements-acceptance.txt'
    for line in path.read_text(encoding='utf-8').splitlines():
        if '==' in line and not line.lstrip().startswith('#'):
            name, expected = line.strip().split('==')
            pins[name] = expected
    return pins


def runtime_smoke(cpu_only=False):
    modules = ('numpy', 'torch', 'torchvision', 'transformers', 'accelerate', 'PIL.Image',
               'pandas', 'pyarrow', 'sentencepiece', 'timm', 'cv2', 'onnxruntime',
               'rapidocr_onnxruntime', 'tokenizers', 'safetensors')
    loaded = {name: importlib.import_module(name) for name in modules}
    import numpy as np
    import torch
    from torchvision.ops import nms
    from rapidocr_onnxruntime import RapidOCR
    from transformers import (AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText,
                              InternVLForConditionalGeneration, LlavaForConditionalGeneration,
                              Qwen2_5_VLForConditionalGeneration)
    nms(torch.tensor([[0., 0., 10., 10.]]), torch.tensor([1.]), .5)
    engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
    engine(np.full((48, 160, 3), 255, dtype=np.uint8))
    result = dict(imports=list(loaded), torchvision_ops=True, ocr_cpu_inference=True,
                  model_classes=[c.__name__ for c in (AutoProcessor, AutoModelForCausalLM,
                      AutoModelForImageTextToText, InternVLForConditionalGeneration,
                      LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration)])
    if not cpu_only:
        if not torch.cuda.is_available():
            raise RuntimeError('PyTorch cannot initialize CUDA with the installed NVIDIA driver')
        tensor = torch.ones(1, device='cuda', dtype=torch.bfloat16)
        if (tensor + tensor).item() != 2:
            raise RuntimeError('CUDA kernel smoke check failed')
        matrix = torch.eye(16, device='cuda', dtype=torch.bfloat16)
        if not torch.equal(matrix @ matrix, matrix):
            raise RuntimeError('CUDA bfloat16 matrix multiplication smoke check failed')
        torch.cuda.synchronize()
        result['cuda'] = dict(device=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)),
            build=torch.version.cuda, kernel_smoke=True, bfloat16=True,
            matmul_smoke=True, compiled_architectures=torch.cuda.get_arch_list())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--cpu-only', action='store_true', help='diagnostic only; never emits ENV_OK')
    args = parser.parse_args()
    pins = pinned_versions()
    actual = {name: version(name) for name in pins}
    info = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages=actual, required_versions=pins, cpu_only=args.cpu_only, complete=False)
    errors = []
    if sys.version_info[:2] != (3, 10):
        errors.append('CPython 3.10 required, not ' + platform.python_version())
    mismatches = {name: dict(expected=expected, actual=actual[name]) for name, expected in pins.items()
                  if actual[name] is None or actual[name].split('+')[0] != expected}
    if mismatches:
        errors.append('dependency versions differ: ' + json.dumps(mismatches))
    try:
        dependency = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
        info['pip_check'] = dependency.stdout + dependency.stderr
        if dependency.returncode:
            errors.append('pip check failed: ' + info['pip_check'])
        if not errors:
            info['runtime_smoke'] = runtime_smoke(args.cpu_only)
    except Exception as error:
        errors.append('runtime smoke failed: ' + repr(error))
    info.update(errors=errors, complete=not errors)
    if args.out:
        from pipeline.common import save
        save(args.out, info)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if errors:
        print('ENV_INCOMPLETE')
        return 2
    print('ENV_CPU_OK_NOT_GPU_ACCEPTANCE' if args.cpu_only else 'ENV_OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
