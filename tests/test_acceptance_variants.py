"""The four procedures differ in risk domain and supplied state, not test-label fitting."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from pipeline import common, factual, factual_parameters as parameters
from pipeline import provided_assets, report, variant_provenance


@pytest.fixture
def feature_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    rng = np.random.default_rng(23)
    x = rng.uniform(.1, .9, (40, 8))
    y = (x[:, 0] + x[:, 1] > 1).astype(int)
    available = np.ones_like(x, dtype=bool)
    available[3, 2] = False
    dev = np.arange(len(x)) < 24
    raw = [dict(id=str(i), acceptance_split='dev' if dev[i] else 'test') for i in range(len(x))]
    rows = [dict(id=r['id'], answer='synthetic fixture') for r in raw]
    from types import SimpleNamespace
    readings = [{name: SimpleNamespace(risk=float(x[i, j]), available=bool(available[i, j]), detail='fixture')
                 for j, name in enumerate(factual.SIGNALS)} for i in range(len(x))]
    data = (raw, rows, readings, x, available, y, dev)
    monkeypatch.setattr(parameters, 'features', lambda *args: data)
    def new_out(name, **protocol):
        out = tmp_path / name
        out.mkdir()
        (out / 'protocol.json').write_text(json.dumps(dict(mode='full', **protocol)))
        (out / 'evidence').mkdir()
        for model in factual.MODELS:
            for dataset in ('pope', 'textvqa'):
                (out / 'evidence' / (model + '_' + dataset + '.jsonl')).write_text('fixture')
        return out
    return data, new_out


def test_provided_evaluation_cannot_call_fit(feature_fixture, monkeypatch):
    data, new_out = feature_fixture
    fresh = new_out('fresh')
    factual.evaluate(fresh)
    states = {p.stem: json.loads(p.read_text()) for p in (fresh / 'calibrators').glob('*.json')}
    loaded = new_out('loaded', parameter_mode='provided', parameters_sha256=common.sha(fresh / 'report.json'))
    shutil.copyfile(fresh / 'report.json', loaded / 'provided_parameters.json')
    monkeypatch.setattr(parameters, 'load_package', lambda *args: dict(states=states))
    from msrc.model import RiskCalibrator
    def forbidden(*args, **kwargs):
        raise AssertionError('provided variant must not fit')
    monkeypatch.setattr(RiskCalibrator, 'fit', forbidden)
    factual.evaluate(loaded)
    for cell, state in states.items():
        assert json.loads((loaded / 'calibrators' / (cell + '.json')).read_text()) == state
        assert common.sha(loaded / 'signal_usage' / (cell + '.jsonl')) == common.sha(
            fresh / 'signal_usage' / (cell + '.jsonl'))
    assert json.loads((loaded / 'report.json').read_text())['parameter_mode'] == 'provided'


def test_changing_test_labels_cannot_change_any_fitted_state(feature_fixture):
    data, new_out = feature_fixture
    first, second = new_out('first'), new_out('second')
    factual.evaluate(first)
    data[5][~data[6]] = 1 - data[5][~data[6]]
    factual.evaluate(second)
    for path in (first / 'calibrators').glob('*.json'):
        assert json.loads(path.read_text()) == json.loads((second / 'calibrators' / path.name).read_text())


def test_warning_threshold_retains_historical_highest_tie_rule():
    state = parameters.warning_threshold([.5, .5], [0, 1])
    assert state['value'] == 1.0
    assert state['tie_rule'] == 'highest'
    assert state['test_used_for_selection'] is False


def test_actual_factual_warning_gate_and_ledger_are_checked(feature_fixture, monkeypatch):
    from pipeline import acceptance_check, factual_warning
    _, new_out = feature_fixture
    source = new_out('source')
    factual.evaluate(source)
    monkeypatch.setattr(acceptance_check, 'factual_check', lambda *args: None)
    root = source.parent / 'warning'
    result = factual_warning.run(source, root)
    assert len(result['cells']) == 6
    assert result['not_streaming_content_safety'] is True
    assert all(r['all_alarm'] + r['never_alarm'] == 1 for r in result['cells'])
    cell = result['cells'][0]['cell']
    path = root / (cell + '_release.jsonl')
    rows = common.read_rows(path)
    rows[0]['release_start_ns'] = 0
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='decision/release'):
        factual_warning.audit(root, source)


def test_fitted_archive_roundtrip_and_file_hash_validation(tmp_path, monkeypatch):
    import io
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    data = {name: b'synthetic parameter/provenance fixture' for name in provided_assets.EXPECTED}
    import hashlib
    manifest = dict(format='msrc-provided-parameters-v1', contains_base_model_weights=False,
                    contains_test_predictions=False,
                    files={name: hashlib.sha256(value).hexdigest() for name, value in data.items()})
    data['manifest.json'] = json.dumps(manifest).encode()
    archive = tmp_path / 'valid.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, value in data.items():
            member = tarfile.TarInfo(name)
            member.size = len(value)
            tar.addfile(member, io.BytesIO(value))
    out = tmp_path / 'provided'
    provided_assets.unpack(archive, out)
    assert provided_assets.verify(out) == manifest
    (out / 'factual.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        provided_assets.verify(out)


def test_parameter_identity_is_portable_but_image_content_is_bound(tmp_path):
    one, two = tmp_path / 'a.jpg', tmp_path / 'b.jpg'
    one.write_bytes(b'image bytes fixture')
    two.write_bytes(one.read_bytes())
    a = [dict(id='x', question='q', image=str(one), acceptance_split='dev', label=0)]
    b = [dict(a[0], image=str(two))]
    assert parameters.identity(a) == parameters.identity(b)
    two.write_bytes(b'different image')
    assert parameters.identity(a) != parameters.identity(b)


def test_code_fingerprint_accepts_only_line_ending_differences(tmp_path):
    a, b = tmp_path / 'linux.py', tmp_path / 'windows.py'
    a.write_bytes(b'x = 1\ny = 2\n')
    b.write_bytes(b'x = 1\r\ny = 2\r\n')
    assert common.sha(a) != common.sha(b)
    assert parameters.text_sha256(a) == parameters.text_sha256(b)
    b.write_bytes(b'x = 1\r\ny = 3\r\n')
    assert parameters.text_sha256(a) != parameters.text_sha256(b)


@pytest.mark.parametrize('kind', ['symlink', 'traversal', 'prediction'])
def test_fitted_archive_refuses_extra_unsafe_or_test_prediction_members(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    path = tmp_path / 'bad.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        member = tarfile.TarInfo({'symlink': 'factual.json', 'traversal': '../factual.json',
                                 'prediction': 'test_delivery.jsonl'}[kind])
        if kind == 'symlink':
            member.type, member.linkname = tarfile.SYMTYPE, '/outside'
        archive.addfile(member)
    with pytest.raises(ValueError):
        provided_assets.unpack(path, tmp_path / 'fitted')
    assert not (tmp_path / 'fitted').exists()


def test_factual_profile_report_does_not_require_or_claim_content_safety(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    run = tmp_path / 'run'
    (run / 'factual').mkdir(parents=True)
    (run / 'factual_warning').mkdir()
    cells = [dict(cell=m + '_' + d, relative_improvement=.3,
                  minimum_actual_signals_per_test_sample=6) for m in factual.MODELS for d in ('pope', 'textvqa')]
    (run / 'factual/report.json').write_text(json.dumps(dict(cells=cells)))
    warning_cells = [dict(cell=r['cell'], n=20, threshold=.5, TP=8, TN=8, FP=2, FN=2,
                          accuracy=.8) for r in cells]
    (run / 'factual_warning/report.json').write_text(json.dumps(dict(cells=warning_cells, minimum_accuracy=.8)))
    monkeypatch.setattr(report, 'check', lambda *args: [])
    monkeypatch.setattr(variant_provenance, 'audit', lambda *args: dict(scope='synthetic fixture'))
    result = report.build(run, 'milestone', 'factual_no_fit')
    assert result['verdict'] == 'PASS'
    assert 'safety' not in result['signals']
    assert result['warning_domain'] == 'factual'
    text = (run / 'report/report.html').read_text(encoding='utf-8')
    assert '不是内容有害性' in text
    assert 'PlugGuard适配版' not in text
    assert report.build(run, 'completion', 'factual_no_fit')['verdict'] == 'FAIL'


def test_user_no_fit_name_means_no_supplied_parameters():
    assert variant_provenance.VARIANTS == {
        'factual_no_fit': ('factual', 'fresh_fit'), 'factual_fitted': ('factual', 'provided'),
        'safety_no_fit': ('safety', 'fresh_fit'), 'safety_fitted': ('safety', 'provided')}


def test_apply_dry_plan_uses_frozen_evaluation_not_training():
    bash = shutil.which('bash') or (r'D:\git\Git\bin\bash.exe' if os.name == 'nt' else None)
    if not bash or not Path(bash).is_file():
        pytest.skip('Bash unavailable')
    env = dict(os.environ, MSRC_PYTHON=sys.executable.replace('\\', '/'), PYTHONIOENCODING='utf-8')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([bash, 'scripts/03_collect_factual.sh', '--run-id', 'provided', '--phase', 'apply'],
                            cwd=root, env=env, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    assert '--phase evaluate' in result.stdout
    assert 'FACTUAL_COMPLETED' not in result.stdout


@pytest.mark.parametrize('variant', sorted(variant_provenance.VARIANTS))
def test_four_one_click_entries_have_the_documented_scope(variant):
    bash = shutil.which('bash') or (r'D:\git\Git\bin\bash.exe' if os.name == 'nt' else None)
    if not bash or not Path(bash).is_file():
        pytest.skip('Bash unavailable')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([bash, 'scripts/run_' + variant + '.sh', '--run-id', variant + '_test'],
                            cwd=root, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    text = result.stdout
    assert '--phase collect' in text and '--variant ' + variant in text
    assert ('02_prepare_fitted.sh' in text) == variant.endswith('_fitted')
    assert ('--parameters assets/fitted/factual.json' in text) == variant.endswith('_fitted')
    assert ('--phase apply' in text) == variant.endswith('_fitted')
    assert ('--phase fit' in text) == variant.endswith('_no_fit')
    assert ('04_score_factual_warning.sh' in text) == variant.startswith('factual_')
    assert ('04_score_safety.sh' in text) == variant.startswith('safety_')
    assert ('--head-package' in text) == (variant == 'safety_fitted')
    assert ('--refit' in text) == (variant == 'safety_no_fit')


@pytest.mark.parametrize('name', ['FACTUAL_NO_FIT', 'FACTUAL_FITTED', 'SAFETY_NO_FIT', 'SAFETY_FITTED'])
def test_four_procedures_cover_all_indicators_and_use_real_script_paths(name):
    import re
    root = Path(__file__).resolve().parents[1]
    document = (root / 'docs' / ('FLOW_' + name + '.md')).read_text(encoding='utf-8')
    assert document.count('## （') == 6
    for indicator in ('指标2.1', '2.2', '2.3'):
        assert indicator in document
    assert 'acceptance-four-variants-20261023' in document
    for path in re.findall(r'bash (scripts/[a-zA-Z0-9_]+\.sh)', document):
        assert (root / path).is_file(), path
    assert ('--head-package' in document) == (name == 'SAFETY_FITTED' or name == 'SAFETY_NO_FIT')
