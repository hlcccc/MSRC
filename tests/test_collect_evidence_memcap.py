# -*- coding: utf-8 -*-
"""The shared-GPU memory cap.

Two things have to hold. With no cap the flag must be inert -- a run that owns its
device must behave exactly as before, and must not even import torch to decide that.
With a cap the value must be range-checked before anything is loaded, because a
typo like 25 meaning 25% would otherwise either disable the protection or raise
halfway through a collection that has already spent an hour on the GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_evidence_memcap_under_test", ROOT / "scripts" / "collect_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_collector()


def test_no_cap_is_the_default_and_touches_nothing(capsys):
    assert collector.cap_gpu_memory(0.0, "cuda:0", "cuda:1") is None
    assert capsys.readouterr().out == "", "an uncapped run must not announce a cap"


def test_a_fraction_above_one_is_refused():
    """0.25 and 25 are both plausible things to type; only one is a fraction."""
    for bad in (-0.5, 1.5, 25.0):
        with pytest.raises(SystemExit, match="gpu-memory-fraction"):
            collector.cap_gpu_memory(bad, "cuda:0")


def test_a_cpu_device_is_left_alone(capsys):
    """There is no device memory to cap, and importing torch to say so would make a
    CPU-only run depend on a GPU build."""
    assert collector.cap_gpu_memory(0.5, "cpu") is None
    assert capsys.readouterr().out == ""


def test_the_flag_exists_and_defaults_to_no_cap():
    """Guards the CLI surface the queue scripts call."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.0)
    assert parser.parse_args([]).gpu_memory_fraction == 0.0

    source = (ROOT / "scripts" / "collect_evidence.py").read_text(encoding="utf-8")
    assert '"--gpu-memory-fraction"' in source
    assert "cap_gpu_memory(args.gpu_memory_fraction" in source
