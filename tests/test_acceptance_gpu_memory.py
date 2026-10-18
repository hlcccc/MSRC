"""Acceptance entry points do not impose an implicit GPU memory fraction."""
import ast
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


def test_factual_collection_uses_uncapped_default(tmp_path, monkeypatch):
    from pipeline import preflight
    calls = []
    monkeypatch.setattr(preflight, 'wait_for_gpu', lambda: None, raising=False)

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    collect = load_function(ROOT / 'pipeline/factual.py', 'collect', {
        'MODELS': {'llava': '/weights/llava'}, 'SOURCE': ROOT, 'sys': sys,
        'SEED': 20260920, 'inside': lambda path: path,
        'subprocess': SimpleNamespace(run=run, STDOUT=-2),
    })
    collect(tmp_path)
    assert len(calls) == 2
    for command in calls:
        assert '--gpu-memory-fraction' not in command
        assert command[command.index('--dtype') + 1] == 'bfloat16'
        assert command[command.index('--max-new-tokens') + 1] == '128'


def test_safety_keeps_threads_and_seed_without_cap(monkeypatch):
    from pipeline import preflight
    calls = []
    monkeypatch.setattr(preflight, 'wait_for_gpu', lambda: None, raising=False)

    def reject_cap(*args, **kwargs):
        raise AssertionError('acceptance inference must not set a memory fraction')

    torch = SimpleNamespace(
        set_num_threads=lambda n: calls.append(('threads', n)),
        manual_seed=lambda n: calls.append(('seed', n)),
        cuda=SimpleNamespace(set_per_process_memory_fraction=reject_cap),
    )
    monkeypatch.setitem(sys.modules, 'torch', torch)
    configure = load_function(ROOT / 'pipeline/safety.py', 'configure_gpu', {'SEED': 20260920})
    assert configure() is torch
    assert calls == [('threads', 2), ('seed', 20260920)]
