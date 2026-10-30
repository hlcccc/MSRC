"""Uncapped new runs still wait for an available GPU."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    code = compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec')
    exec(code, namespace)
    return namespace[name]


def collection_protocol(tmp_path):
    from pipeline.common import sha
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(parameter_mode='fresh_fit',
        collector_entry='scripts/collect_evidence.py', collector_sha256=sha(ROOT / 'scripts/collect_evidence.py'))))
    return {'json': json, 'sha': sha}


def test_factual_collection_waits_before_uncapped_loads(tmp_path, monkeypatch):
    from pipeline import preflight
    calls, waits = [], []
    monkeypatch.setattr(preflight, 'wait_for_gpu', lambda: waits.append(True))

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    collect = load_function(ROOT / 'pipeline/factual.py', 'collect', {
        'MODELS': {'llava': '/weights/llava'}, 'SOURCE': ROOT, 'sys': sys,
        'SEED': 20260920, 'inside': lambda path: path,
        'subprocess': SimpleNamespace(run=run, STDOUT=-2), **collection_protocol(tmp_path),
    })
    collect(tmp_path)
    assert len(calls) == len(waits) == 2
    for command in calls:
        assert '--gpu-memory-fraction' not in command
        assert command[command.index('--dtype') + 1] == 'bfloat16'
        assert command[command.index('--max-new-tokens') + 1] == '128'


def test_safety_waits_and_keeps_threads_and_seed_without_cap(monkeypatch):
    from pipeline import preflight
    calls = []
    monkeypatch.setattr(preflight, 'wait_for_gpu', lambda: calls.append(('preflight',)))

    def reject_cap(*args, **kwargs):
        raise AssertionError('new inference must not impose a fixed memory fraction')

    torch = SimpleNamespace(
        set_num_threads=lambda n: calls.append(('threads', n)),
        manual_seed=lambda n: calls.append(('seed', n)),
        cuda=SimpleNamespace(set_per_process_memory_fraction=reject_cap),
    )
    monkeypatch.setitem(sys.modules, 'torch', torch)
    configure = load_function(ROOT / 'pipeline/safety.py', 'configure_gpu', {'SEED': 20260920})
    assert configure() is torch
    assert calls == [('preflight',), ('threads', 2), ('seed', 20260920)]


def test_factual_sharing_passes_only_the_authorized_memory_cap(tmp_path, monkeypatch):
    from pipeline import gpu_sharing, preflight
    calls = []
    monkeypatch.setattr(preflight, 'wait_for_gpu', lambda: [{'total_mib': 81920}])
    monkeypatch.setattr(gpu_sharing, 'load_shared_policy', lambda **kw: {'allocator_cap_gib': 36})
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    collect = load_function(ROOT / 'pipeline/factual.py', 'collect', {
        'MODELS': {'llava': '/weights/llava'}, 'SOURCE': ROOT, 'sys': sys,
        'SEED': 20260920, 'inside': lambda path: path,
        'subprocess': SimpleNamespace(run=run, STDOUT=-2), **collection_protocol(tmp_path),
    })
    collect(tmp_path)
    assert len(calls) == 2
    for command in calls:
        assert float(command[command.index('--gpu-memory-fraction') + 1]) == 0.45
        assert command[command.index('--max-new-tokens') + 1] == '128'
        assert command[command.index('--k') + 1] == '3'
