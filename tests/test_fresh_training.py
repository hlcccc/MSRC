"""A loaded project checkpoint must not pass the from-scratch provenance check."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from pipeline import common, fresh_training


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fresh_run(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    run = tmp_path / 'runs' / 'fresh'
    def write(relative, value):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
    write('factual/report.json', dict(status='FULL_FACTUAL_FLOW_COMPLETE', mode='full',
          old_evidence_used=False, cells=[dict(cell=cell) for cell in sorted(fresh_training.CELLS)]))
    write('factual/protocol.json', dict(mode='full'))
    for dataset in ('pope', 'textvqa'):
        path = run / 'factual' / 'inputs' / (dataset + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(dict(id=i, acceptance_split=split))
                                 for i, split in [('a', 'dev'), ('b', 'dev'), ('c', 'test')]),
                        encoding='utf-8')
    for cell in fresh_training.CELLS:
        write('factual/calibrators/' + cell + '.json', dict(
            fitting_scope='development only', converged=True, train_ids=['a', 'b']))
    write('safety/protocol.json', dict(backend='plugguard', execution_kind='fresh_inference',
          training_mode='development_refit', development_generated_this_run=True,
          fit_n=2688, calibration_n=672, test_n=3360, created_ns=100))
    head = run / 'safety/head.pt'
    head.write_bytes(b'synthetic fixture, not model weights')
    write('safety/threshold.json', dict(head_sha256=common.sha(head), frozen_before_test_ns=200,
          test_used_for_fitting_or_selection=False))
    write('safety/report.json', dict(status='FULL_SAFETY_FLOW_COMPLETE', old_scores_used=False,
          training_mode='development_refit', development_generated_this_run=True))
    return run, write


def test_fresh_scope_is_not_a_claim_that_metrics_passed(fresh_run):
    run, _ = fresh_run
    result = fresh_training.audit(run)
    assert result['imported_project_training_state'] is False
    assert len(result['factual_calibrators']) == 6
    assert 'acceptance_pass' not in result
    assert 'metrics' in result['scope']


@pytest.mark.parametrize('change', ['loaded_head', 'copied_threshold', 'old_scores',
                                  'old_dev', 'future_start', 'test_selection', 'changed_head'])
def test_nonfresh_state_is_rejected(fresh_run, change):
    run, write = fresh_run
    if change in ('loaded_head', 'old_dev', 'future_start'):
        value = fresh_training.load(run / 'safety/protocol.json')
        value.update({'loaded_head': dict(training_mode='loaded_development_frozen_head'),
                      'old_dev': dict(development_generated_this_run=False),
                      'future_start': dict(created_ns=300)}[change])
        write('safety/protocol.json', value)
    elif change in ('copied_threshold', 'test_selection'):
        value = fresh_training.load(run / 'safety/threshold.json')
        value.update({'copied_threshold': dict(checkpoint_reused=True),
                      'test_selection': dict(test_used_for_fitting_or_selection=True)}[change])
        write('safety/threshold.json', value)
    elif change == 'old_scores':
        value = fresh_training.load(run / 'safety/report.json')
        write('safety/report.json', dict(value, old_scores_used=True))
    else:
        (run / 'safety/head.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):
        fresh_training.audit(run)


def test_calibrator_cannot_include_test_ids(fresh_run):
    run, write = fresh_run
    write('factual/calibrators/qwen_pope.json', dict(
        fitting_scope='development only', converged=True, train_ids=['a', 'b', 'c']))
    with pytest.raises(ValueError, match='calibrator'):
        fresh_training.audit(run)


def test_scope_cli_writes_a_separate_receipt_not_a_pass(fresh_run, monkeypatch, capsys):
    run, _ = fresh_run
    monkeypatch.setattr(fresh_training, 'runs_dir', lambda: run.parent)
    monkeypatch.setattr('sys.argv', ['fresh_training', '--run-id', run.name])
    fresh_training.main()
    output = capsys.readouterr().out
    assert 'FRESH_TRAINING_SCOPE_OK fresh' in output
    assert 'ACCEPTANCE_PASS' not in output
    assert (run / 'FRESH_TRAINING_VERIFIED.json').is_file()


def bash_path():
    bash = shutil.which('bash') or (r'D:\git\Git\bin\bash.exe' if os.name == 'nt' else None)
    if not bash or not Path(bash).is_file():
        pytest.skip('Bash is unavailable')
    return bash


@pytest.mark.parametrize('phase,expected', [('collect', '--phase prepare'),
                                           ('fit', '--phase evaluate'), ('run', '--mode full')])
def test_factual_split_phase_plans_use_existing_implementation(phase, expected):
    import sys
    env = dict(os.environ, MSRC_PYTHON=sys.executable.replace('\\', '/'), PYTHONIOENCODING='utf-8')
    result = subprocess.run([bash_path(), 'scripts/03_collect_factual.sh', '--run-id', 'plan_only',
                             '--phase', phase], cwd=ROOT, env=env, capture_output=True,
                            text=True, encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert 'FACTUAL_COMPLETED' not in result.stdout


def test_documented_flow_trains_project_parameters_instead_of_importing_them():
    document = (ROOT / 'docs/ACCEPTANCE_FLOW.md').read_text(encoding='utf-8')
    assert '--phase collect --execute' in document
    assert '--phase fit --execute' in document
    assert '--mode full --refit --execute' in document
    assert 'bash scripts/check_fresh_training.sh' in document
    assert document.count('## （') == 6
