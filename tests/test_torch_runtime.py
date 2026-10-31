"""Blackwell changes CUDA wheels, never the pinned releases or acceptance checks."""
from types import SimpleNamespace

import pytest

from pipeline import torch_runtime as runtime


@pytest.mark.parametrize("capabilities, expected", [
    ("8.0\n8.0\n", False), ("9.0\n", False), ("12.0\n", True),
    ("8.0\n10.0\n", True), ("N/A\n", False)])
def test_selects_cuda128_only_for_blackwell(monkeypatch, capabilities, expected):
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw:
        SimpleNamespace(stdout=capabilities))
    assert runtime.needs_cuda128() is expected


def test_reference_gpu_never_installs_alternate_wheels(monkeypatch):
    monkeypatch.setattr(runtime, "needs_cuda128", lambda: False)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected install"))
    runtime.prepare()


def configure_blackwell(monkeypatch):
    monkeypatch.setattr(runtime, "needs_cuda128", lambda: True)
    monkeypatch.setattr(runtime, "pinned_versions", lambda: {"torch": "2.7.1", "torchvision": "0.22.1"})
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name:
        {"torch": "2.7.1", "torchvision": "0.22.1"}[name])


def test_blackwell_installs_only_the_pinned_official_cuda128_pair(monkeypatch):
    configure_blackwell(monkeypatch)
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    runtime.prepare()
    command, options = calls[0]
    assert "torch==2.7.1+cu128" in command and "torchvision==0.22.1+cu128" in command
    assert command[-2:] == ["--index-url", runtime.CUDA128_INDEX]
    assert options["check"] is True


def test_installed_cuda128_pair_is_not_downloaded_again(monkeypatch):
    configure_blackwell(monkeypatch)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name:
        {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128"}[name])
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected install"))
    runtime.prepare()


def test_offline_missing_cuda128_wheels_fails_without_network(tmp_path, monkeypatch):
    configure_blackwell(monkeypatch)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected network"))
    with pytest.raises(RuntimeError, match="missing: torch, torchvision"):
        runtime.prepare(tmp_path)


def test_offline_cuda128_pair_uses_only_local_wheels(tmp_path, monkeypatch):
    configure_blackwell(monkeypatch)
    for name, version in (("torch", "2.7.1"), ("torchvision", "0.22.1")):
        (tmp_path / (name + "-" + version + "+cu128-cp310-cp310-linux_x86_64.whl")).touch()
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    runtime.prepare(tmp_path)
    assert calls[0][-3:] == ["--no-index", "--find-links", str(tmp_path.resolve())]
    assert "--index-url" not in calls[0]
