"""Check the delivered MSRC environment and local materials before downloading."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
ROLES = ('internvl', 'qwen', 'generator', 'judge')
PYTHON_ARCHIVE = 'cpython-3.10.19+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz'


def message(state, item, detail=''):
    print('[%s] %s%s' % (state, item, ': ' + str(detail) if detail else ''), flush=True)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def local_write(path):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('write path leaves this project: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def runtime_environment(python=None):
    env = os.environ.copy()
    for key, name in (
        ('TMPDIR', 'tmp'), ('PIP_CACHE_DIR', 'cache/pip'),
        ('HF_HOME', 'cache/hf'), ('HF_HUB_CACHE', 'cache/hf/hub'),
        ('HF_DATASETS_CACHE', 'cache/hf/datasets'), ('TORCH_HOME', 'cache/torch'),
        ('XDG_CACHE_HOME', 'cache'), ('CUDA_CACHE_PATH', 'cache/cuda'),
        ('TRITON_CACHE_DIR', 'cache/triton'), ('NUMBA_CACHE_DIR', 'cache/numba'),
        ('MPLCONFIGDIR', 'cache/matplotlib'),
    ):
        path = local_write(ROOT / name)
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PIP_NO_INDEX='1',
               MSRC_CONFIG=str(ROOT / 'configs/paths.json'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'ARROW_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '2'
    # A venv's executable may be a symlink; preserve its invocation path.
    if python and Path(python).absolute().parent.parent == ROOT / '.venv':
        libs = ROOT / '.venv/system-libs'
        names = ('usr/lib/x86_64-linux-gnu', 'lib/x86_64-linux-gnu',
                 'usr/lib/aarch64-linux-gnu', 'lib/aarch64-linux-gnu')
        env['LD_LIBRARY_PATH'] = ':'.join(str(libs / n) for n in names) + (
            ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def verify_sources():
    manifest = ROOT / 'check_sources.sha256'
    entries = manifest.read_text(encoding='utf-8').splitlines()
    if not entries:
        raise ValueError('empty source manifest')
    seen = set()
    for line in entries:
        expected, name = line.split('  ', 1)
        path = ROOT / name
        if (len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected)
                or name in seen or not path.resolve().is_relative_to(ROOT)
                or Path(name).is_absolute() or '..' in Path(name).parts):
            raise ValueError('invalid source manifest entry')
        seen.add(name)
        if digest(path) != expected:
            raise ValueError('source differs from this package: ' + name)
    message('REUSE_OK', 'code', '%d files verified' % len(seen))


def candidates():
    explicit = os.environ.get('MSRC_PYTHON')
    if explicit:
        values = [explicit]
    else:
        values = [str(Path(os.environ[key]) / 'bin/python')
                  for key in ('VIRTUAL_ENV', 'CONDA_PREFIX') if os.environ.get(key)]
        values += [str(ROOT / '.venv/bin/python'), sys.executable, 'python3.10', 'python3']
    seen = set()
    for value in values:
        path = shutil.which(value)
        if path:
            path = os.path.abspath(path)
            if path not in seen:
                seen.add(path)
                yield path


def run(command, env, *, timeout=None):
    message('CHECK', 'command', shlex.join([str(x) for x in command]))
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                          text=True, errors='replace', timeout=timeout)


def choose_environment(gpu=False):
    smoke_failed = False
    probe = ('import importlib.metadata as m,json,sys; '
             'print(json.dumps(dict(python=list(sys.version_info[:2]), '
             'torch=next((d.version for d in m.distributions(name="torch")),None))))')
    for python in candidates():
        env = runtime_environment(python)
        try:
            result = run([python, '-B', '-c', probe], env, timeout=30)
            info = json.loads(result.stdout) if result.returncode == 0 else {}
            if info.get('python') != [3, 10] or info.get('torch') != '2.7.1+cu128':
                message('NOT_COMPATIBLE', python, info or result.stderr.strip())
                continue
            command = [python, '-B', '-m', 'pipeline.env_check']
            if not gpu:
                command.append('--cpu-only')
            result = run(command, env, timeout=300)
            print(result.stdout, end='', flush=True)
            if result.stderr:
                print(result.stderr, file=sys.stderr, end='', flush=True)
            marker = 'ENV_OK' if gpu else 'ENV_CPU_OK_NOT_GPU_ACCEPTANCE'
            if result.returncode == 0 and marker in result.stdout.splitlines():
                message('ENV_REUSE_OK', python)
                return python, 0
            # Dependency mismatches can be fixed by installing a private env;
            # driver, permission and native-library failures need diagnosis.
            try:
                payload = json.loads(result.stdout.rsplit('\nENV_', 1)[0])
                errors = payload.get('errors', [])
                version_only = bool(errors) and all(
                    x.startswith(('CPython 3.10 required', 'dependency versions differ',
                                  'pip check failed')) for x in errors)
            except (ValueError, TypeError):
                version_only = False
            smoke_failed |= not version_only
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            message('CHECK_ERROR', python, error)
            smoke_failed = True
    if smoke_failed:
        message('REPAIR', 'environment', 'inspect errors; do not blindly reinstall or download models')
        return None, 4
    message('NEED_ENV_INSTALL', 'environment', 'no compatible environment found')
    return None, 2


def install_inputs(materials):
    wheels = materials / 'env/wheels'
    patch = materials / 'env_patch/SHA256SUMS.txt'
    message('LOCAL_INPUT' if wheels.is_dir() and any(wheels.glob('*.whl')) else 'NEED_DOWNLOAD',
            'env/wheels', 'local files still require installer validation' if wheels.is_dir() else wheels)
    message('LOCAL_INPUT' if patch.is_file() else 'NEED_DOWNLOAD',
            'env_patch', 'local manifest present; installer verifies it before use' if patch.is_file() else patch.parent)
    probe = 'import ensurepip,sys; assert sys.version_info[:2] == (3,10)'
    base = os.environ.get('MSRC_BASE_PYTHON', 'python3.10')
    try:
        usable = run([base, '-B', '-c', probe], runtime_environment(), timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        usable = False
    if not usable:
        archive = materials / 'env/python' / PYTHON_ARCHIVE
        message('LOCAL_INPUT' if archive.is_file() else 'NEED_DOWNLOAD',
                'env/python/' + PYTHON_ARCHIVE)
    else:
        message('REUSE_OK', 'base CPython 3.10', 'offline Python archive is not required')


def model_mapping(materials, explicit=None):
    current = ROOT / 'configs/paths.json'
    if explicit:
        source = Path(explicit).expanduser().absolute()
        value, base = json.loads(source.read_text(encoding='utf-8')), source.parent
    elif current.is_file():
        value = json.loads(current.read_text(encoding='utf-8')).get('models', {})
        base = ROOT
    else:
        source = materials / 'local-models.fast3.json'
        if source.is_file():
            value, base = json.loads(source.read_text(encoding='utf-8')), source.parent
        else:
            # This shipped mapping contains directory names, not validated models.
            value = json.loads((ROOT / 'configs/local-models.fast3.json').read_text(encoding='utf-8'))
            base = materials
    result = {}
    for role in ROLES:
        item = value.get(role, value.get('qwen') if role == 'generator' else None)
        path = item.get('path') if isinstance(item, dict) else item
        if not isinstance(path, str) or not path:
            raise ValueError('missing model path for ' + role)
        result[role] = str((base / Path(path).expanduser()).resolve())
    if result['qwen'] != result['generator']:
        raise ValueError('Qwen donor and generator must share the same pinned weights')
    return result


@contextlib.contextmanager
def model_config(models):
    path = local_write(ROOT / 'tmp/model-check.json')
    # Use a unique, project-local read configuration; never overwrite paths.json.
    with tempfile.TemporaryDirectory(prefix='model-check-', dir=path.parent) as folder:
        temporary = Path(folder) / 'paths.json'
        temporary.write_text(json.dumps({'models': models}), encoding='utf-8')
        previous = os.environ.get('MSRC_CONFIG')
        os.environ['MSRC_CONFIG'] = str(temporary)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop('MSRC_CONFIG', None)
            else:
                os.environ['MSRC_CONFIG'] = previous


def check_models(models):
    from pipeline.fast3 import verify_models
    recipe = json.loads((ROOT / 'resources/reproduction/models.json').read_text(encoding='utf-8'))['models']
    specs = {role: recipe[role] for role in ('internvl', 'qwen')}
    specs.update(json.loads((ROOT / 'assets/plugguard/base_models.json').read_text(encoding='utf-8')))
    missing = set()
    for role, spec in specs.items():
        for name in spec['files']:
            path = Path(models[role]) / name
            if not path.is_file():
                missing.add(str(path))
    if missing:
        for path in sorted(missing):
            message('NEED_DOWNLOAD_OR_PATH', 'model file', path)
        return False
    with model_config(models):
        verify_models()
    message('REUSE_OK', 'models', 'all required file SHA256 values verified')
    return True


def check_fitted(materials, prepare):
    from pipeline import provided_assets as fitted
    destination = ROOT / 'assets/fitted'
    if destination.exists():
        fitted.prepare(check=True)
        message('REUSE_OK', 'fitted parameters')
        return True
    registry = json.loads((ROOT / 'resources/reproduction/fitted.json').read_text(encoding='utf-8'))
    invalid = []
    for archive in (ROOT / 'cache/fitted/parameters.tar.gz',
                    materials / 'fitted_only/parameters-20261023.tar.gz'):
        if not archive.is_file():
            continue
        if archive.stat().st_size != registry['bytes'] or digest(archive) != registry['sha256']:
            invalid.append(archive)
            continue
        for unused in invalid:
            message('NOT_USED', 'fitted archive', 'invalid cache; using another verified archive: ' + str(unused))
        if not prepare:
            message('NEED_PREPARE', 'fitted parameters', 'verified local archive: ' + str(archive))
            return False
        # Unpack explicitly instead of prepare(False), which may download.
        fitted.unpack(archive, destination)
        fitted.prepare(check=True)
        message('REUSE_OK', 'fitted parameters')
        return True
    for archive in invalid:
        message('INVALID', 'fitted archive', 'wrong size or SHA256: ' + str(archive))
    message('NEED_DOWNLOAD', 'fitted_only/parameters-20261023.tar.gz')
    return False


def check_data(package, prepare):
    from pipeline import prepared_bundle
    assets = ROOT / 'assets'
    if (assets / 'prepared_bundle.json').is_file():
        prepared_bundle.verify(assets)
        message('REUSE_OK', 'prepared data and fixed splits', 'original transport archive is not required')
        return True
    if any((assets / name).exists() for name in ('data', 'raw', 'partitions')):
        raise ValueError('partial/unverified data preserved; inspect it before restoring into this project')
    if not package.exists():
        message('NEED_DOWNLOAD', 'prepared_data', package)
        return False
    if not prepare:
        message('NEED_PREPARE', 'prepared_data',
                'local transport present, not yet approved; --prepare verifies before restoring: ' + str(package))
        return False
    prepared_bundle.restore(package, assets)
    message('REUSE_OK', 'prepared data and fixed splits')
    return True


def check_configuration(models, prepare, inputs_ready):
    from pipeline import common
    path = ROOT / 'configs/paths.json'
    expected_data = {name: str(ROOT / 'assets/data' / suffix) for name, suffix in (
        ('factual', 'factual'), ('safety', 'safety'), ('mmsafety', 'MM-SafetyBench/data'))}
    value = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    actual_models = {k: str(common.root_path(v.get('path') if isinstance(v, dict) else v))
                     for k, v in value.get('models', {}).items() if k in ROLES}
    actual_data = {k: str(common.root_path(v)) for k, v in value.get('data', {}).items() if k in expected_data}
    if actual_models == models and actual_data == expected_data:
        message('REUSE_OK', 'project paths')
        return True
    if not prepare or not inputs_ready:
        message('NEED_PREPARE', 'project paths', 'no download needed for this item')
        return False
    value.setdefault('models', {}).update(models)
    value.setdefault('data', {}).update(expected_data)
    value.setdefault('gpu', '0')
    value.setdefault('runs_dir', 'runs')
    common.save(path, value)
    message('REUSE_OK', 'project paths', 'configured to the verified local inputs')
    return True


def materials_check(args):
    ready = {}
    models = None
    actions = [
        ('fitted parameters', lambda: check_fitted(args.materials_dir, args.prepare)),
        ('prepared data', lambda: check_data(args.prepared_package or args.materials_dir / 'prepared_data', args.prepare)),
    ]
    try:
        models = model_mapping(args.materials_dir, args.local_models)
        actions.append(('models', lambda: check_models(models)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        message('REPAIR', 'model paths', error)
        ready['models'] = False
    for name, action in actions:
        try:
            ready[name] = action()
        except (OSError, ValueError, KeyError, TypeError) as error:
            message('REPAIR', name, error)
            ready[name] = False
    if models is not None:
        try:
            ready['configuration'] = check_configuration(models, args.prepare, all(ready.values()))
        except (OSError, ValueError, KeyError, TypeError) as error:
            message('REPAIR', 'project paths', error)
            ready['configuration'] = False
    if all(ready.values()):
        message('MATERIALS_OK', 'local inputs', 'no material download or preparation is needed')
        return 0
    message('MATERIALS_INCOMPLETE', 'local inputs', 'download only listed missing/corrupt files; --prepare uses local files only')
    return 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='also execute the original CUDA smoke test')
    parser.add_argument('--environment-only', action='store_true')
    parser.add_argument('--prepare', action='store_true', help='verify and prepare missing local assets; never download')
    parser.add_argument('--materials-dir', type=Path, default=ROOT.parent)
    parser.add_argument('--prepared-package', type=Path)
    parser.add_argument('--local-models', type=Path)
    parser.add_argument('--_materials-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.materials_dir = args.materials_dir.expanduser().resolve()
    try:
        verify_sources()
        if args._materials_worker:
            return materials_check(args)
        selected = local_write(ROOT / 'cache/check-first/python.path')
        selected.write_text('', encoding='utf-8')
        python, status = choose_environment(args.gpu)
        if python is None:
            if status == 2:
                install_inputs(args.materials_dir)
            message('NOT_CHECKED', 'materials', 'finish environment check, then rerun; do not download all models now')
            return status
        selected.write_text(python + '\n', encoding='utf-8')
        if args.environment_only:
            return 0
        command = [python, '-B', str(Path(__file__).resolve()), '--_materials-worker',
                   '--materials-dir', str(args.materials_dir)]
        for flag, value in (('--prepared-package', args.prepared_package), ('--local-models', args.local_models)):
            if value is not None:
                command += [flag, str(value.expanduser().absolute())]
        if args.prepare:
            command.append('--prepare')
        result = subprocess.run(command, cwd=ROOT, env=runtime_environment(python))
        if result.returncode == 0:
            message('PRECHECK_OK' if args.gpu else 'CPU_PREPARATION_OK_NOT_GPU_ACCEPTANCE', 'MSRC')
        return result.returncode
    except (OSError, ValueError, KeyError, TypeError) as error:
        message('REPAIR', 'check stopped', error)
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
