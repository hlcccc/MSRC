"""Cold-start source contracts, runtime checks, report exits and promised outputs."""
import contextlib
import io
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest

from pipeline import common, env_check, factual, factual_parameters as parameters, runtime_libraries
from extensions.multilayer import variants


def test_fitted_collector_matches_actual_published_source_not_current_scheduler():
    name = parameters.fitted_collector()
    expected = 'bea54e392755a093e596eeddc240cd91659eb522edf25a978a25f5769088b20a'
    assert parameters.text_sha256(factual.SOURCE / name) == expected
    assert parameters.text_sha256(factual.SOURCE / 'scripts/collect_evidence.py') != expected
    provided = parameters.source_fingerprints(name)
    fresh = parameters.source_fingerprints()
    assert provided['scripts/collect_evidence.py'] == expected
    assert [name for name in provided if provided[name] != fresh[name]] == ['scripts/collect_evidence.py']


def test_parameter_contract_does_not_disable_any_other_source_or_model_check():
    collector = parameters.fitted_collector()
    package = dict(format=parameters.FORMAT, sources=parameters.source_fingerprints(collector),
        models_recipe_sha256=parameters.text_sha256(factual.SOURCE / 'resources/reproduction/models.json'))
    parameters.validate_code_contract(package, collector)
    with pytest.raises(ValueError, match='signal/model/scoring'):
        parameters.validate_code_contract(package, 'scripts/collect_evidence.py')
    package['sources']['msrc/model.py'] = 'tampered'
    with pytest.raises(ValueError, match='signal/model/scoring'):
        parameters.validate_code_contract(package, collector)


@pytest.mark.parametrize('mode,entry', [('provided', 'scripts/collect_evidence_fitted.py'),
                                      ('fresh_fit', 'scripts/collect_evidence.py')])
def test_actual_collect_executes_only_the_recorded_mode_specific_entry(tmp_path, monkeypatch, mode, entry):
    from pipeline import preflight, gpu_sharing
    monkeypatch.setattr(factual, 'MODELS', {'llava': '/fixture/model'})
    monkeypatch.setattr(factual, 'inside', lambda p: p)
    monkeypatch.setattr(preflight, 'wait_for_gpu', lambda: [])
    monkeypatch.setattr(gpu_sharing, 'load_shared_policy', lambda **kw: None)
    calls = []
    monkeypatch.setattr(factual.subprocess, 'run', lambda command, **kw:
        calls.append(command) or types.SimpleNamespace(returncode=0))
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(parameter_mode=mode,
        collector_entry=entry, collector_sha256=common.sha(factual.SOURCE / entry))))
    factual.collect(tmp_path)
    assert len(calls) == 2 and all(Path(call[3]) == factual.SOURCE / entry for call in calls)
    protocol = json.loads((tmp_path / 'protocol.json').read_text())
    protocol['collector_entry'] = 'scripts/../other.py'
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='collector'):
        factual.collect(tmp_path)


@pytest.mark.parametrize('passed,exit_code', [(True, 0), (False, 3)])
def test_real_joint_report_cli_return_code_is_nonzero_when_completed_but_failed(tmp_path, monkeypatch, passed, exit_code):
    root = tmp_path / 'runs/new/multilayer'
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(variants.original, 'run_root', lambda run: root)
    monkeypatch.setattr(variants, 'check_materials', lambda *a, **kw: None)
    monkeypatch.setattr(variants, 'VariantExperiment', lambda *a: types.SimpleNamespace(
        setup=lambda: None, path=lambda name: root / name))
    monkeypatch.setattr(variants, 'joint_report', lambda *a: dict(acceptance_pass=passed))
    monkeypatch.setitem(sys.modules, 'fcntl', types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setattr(sys, 'argv', ['variant', '--mode', 'no_fit', '--run-id', 'new', '--phase', 'joint-report'])
    assert variants.main() == exit_code


def test_env_ok_requires_actual_import_and_inference_smokes(monkeypatch):
    monkeypatch.setattr(env_check, 'pinned_versions', lambda: {'fixture': '1.0'})
    monkeypatch.setattr(env_check, 'version', lambda _: '1.0')
    monkeypatch.setattr(env_check.sys, 'version_info', (3, 10))
    monkeypatch.setattr(env_check.subprocess, 'run', lambda *a, **kw: types.SimpleNamespace(
        returncode=0, stdout='No broken requirements', stderr=''))
    def fail(cpu_only=False):
        raise ImportError('libGL.so.1 missing')
    monkeypatch.setattr(env_check, 'runtime_smoke', fail)
    monkeypatch.setattr(sys, 'argv', ['environment'])
    with contextlib.redirect_stdout(io.StringIO()) as output:
        assert env_check.main() == 2
    assert 'ENV_OK' not in output.getvalue() and 'ENV_INCOMPLETE' in output.getvalue()


def test_cpu_diagnostic_cannot_masquerade_as_cuda_env_ok(monkeypatch):
    monkeypatch.setattr(env_check, 'pinned_versions', lambda: {})
    monkeypatch.setattr(env_check.sys, 'version_info', (3, 10))
    monkeypatch.setattr(env_check.subprocess, 'run', lambda *a, **kw: types.SimpleNamespace(returncode=0, stdout='', stderr=''))
    monkeypatch.setattr(env_check, 'runtime_smoke', lambda cpu_only=False: {'cpu_only': cpu_only})
    monkeypatch.setattr(sys, 'argv', ['environment', '--cpu-only'])
    with contextlib.redirect_stdout(io.StringIO()) as output:
        assert env_check.main() == 0
    assert 'ENV_CPU_OK_NOT_GPU_ACCEPTANCE' in output.getvalue() and '\nENV_OK\n' not in output.getvalue()


def test_missing_system_libraries_are_not_silently_accepted(monkeypatch):
    monkeypatch.setattr(runtime_libraries, 'missing_libraries', lambda: [{'library': 'libGL.so.1'}])
    with pytest.raises(RuntimeError, match='missing OCR libraries'):
        runtime_libraries.prepare(check=True)


def test_two_published_flows_state_actual_reports_and_preflight_order():
    for mode in ('FITTED', 'NO_FIT'):
        text = (common.ROOT / 'docs' / ('FLOW_SAFETY_MULTILAYER_' + mode + '.md')).read_text(encoding='utf-8')
        for name in ('signal_report.html', 'calibration_report.html', 'warning_report.html',
                     'signal_summary.csv', 'warning_by_source.csv', 'outputs.json', 'signal_usage/'):
            assert name in text
        assert 'acceptance-multilayer-delivery-20261030' in text
        assert all('\u6307\u6807' + number in text for number in ('1.11', '1.12', '1.13'))
        if mode == 'FITTED':
            assert text.index('bash scripts/02_prepare_fitted.sh') < text.index('bash scripts/02_prepare_assets_data.sh')
