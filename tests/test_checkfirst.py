"""Local control-flow tests, not an MSRC model/GPU acceptance run."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / 'check_env.py'
spec = importlib.util.spec_from_file_location('msrc_checkfirst', SOURCE)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
BASH = shutil.which('bash')
if os.name == 'nt' and shutil.which('git'):
    git_bash = Path(shutil.which('git')).resolve().parents[1] / 'bin/bash.exe'
    if git_bash.is_file():
        BASH = str(git_bash)


def result(stdout='', returncode=0, stderr=''):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class CheckFirstTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = (Path(self.temporary.name) / 'MSRC').resolve()
        self.root.mkdir()
        self.root_patch = patch.object(check, 'ROOT', self.root)
        self.root_patch.start()
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.output = io.StringIO()
        self.stream = redirect_stdout(self.output)
        self.stream.__enter__()

    def tearDown(self):
        self.stream.__exit__(None, None, None)
        self.env_patch.stop()
        self.root_patch.stop()
        self.temporary.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding='utf-8')
        if isinstance(value, str) and value.startswith('#!/usr/bin/env bash'):
            path.chmod(path.stat().st_mode | 0o111)
        return path

    def pipeline(self, **modules):
        values = {'pipeline': types.ModuleType('pipeline')}
        for name, value in modules.items():
            module = types.SimpleNamespace(**value)
            values['pipeline.' + name] = module
            setattr(values['pipeline'], name, module)
        return patch.dict(sys.modules, values)

    def test_source_mismatch_stops_before_environment_or_install(self):
        self.write('core.py', 'original')
        self.write('check_sources.sha256', '0' * 64 + '  core.py\n')
        with patch.object(check, 'choose_environment') as choose, patch.object(sys, 'argv', ['check_env.py']):
            self.assertEqual(check.main(), 4)
        choose.assert_not_called()
        self.assertIn('source differs', self.output.getvalue())

    def test_source_manifest_hashes_real_bytes(self):
        path = self.write('core.py', 'same bytes')
        self.write('check_sources.sha256', check.digest(path) + '  core.py\n')
        check.verify_sources()
        self.assertIn('REUSE_OK', self.output.getvalue())

    def test_source_manifest_rejects_parent_escape(self):
        self.write('check_sources.sha256', '0' * 64 + '  ../escape\n')
        with self.assertRaisesRegex(ValueError, 'invalid source'):
            check.verify_sources()

    def test_explicit_python_does_not_fall_back(self):
        os.environ['MSRC_PYTHON'] = 'chosen'
        with patch.object(check.shutil, 'which', side_effect=lambda x: '/x/' + x):
            self.assertEqual(list(check.candidates()), [os.path.abspath('/x/chosen')])

    def test_activated_python_is_considered_before_project(self):
        os.environ['VIRTUAL_ENV'] = '/active'
        with patch.object(check.shutil, 'which', side_effect=lambda x: x):
            values = list(check.candidates())
        self.assertEqual(values[0], os.path.abspath('/active/bin/python'))
        self.assertEqual(values[1], str(self.root / '.venv/bin/python'))

    def test_wrong_cuda_wheel_never_runs_cpu_smoke(self):
        with patch.object(check, 'candidates', return_value=iter(['/candidate'])), \
                patch.object(check, 'run', return_value=result('{"python":[3,10],"torch":"2.7.1+cu126"}')) as run:
            self.assertEqual(check.choose_environment(), (None, 2))
        self.assertEqual(run.call_count, 1)
        self.assertNotIn('ENV_REUSE_OK', self.output.getvalue())

    def test_qualified_python_runs_original_cpu_check(self):
        calls = [result('{"python":[3,10],"torch":"2.7.1+cu128"}'),
                 result('{}\nENV_CPU_OK_NOT_GPU_ACCEPTANCE\n')]
        with patch.object(check, 'candidates', return_value=iter(['/candidate'])), \
                patch.object(check, 'run', side_effect=calls) as run:
            self.assertEqual(check.choose_environment(), ('/candidate', 0))
        self.assertEqual(run.call_args_list[1].args[0][-3:], ['-m', 'pipeline.env_check', '--cpu-only'])
        self.assertNotIn('install', ' '.join(str(x) for x in run.call_args_list))

    def test_full_gpu_check_is_not_cpu_only(self):
        with patch.object(check, 'candidates', return_value=iter(['/candidate'])), \
                patch.object(check, 'run', side_effect=[
                    result('{"python":[3,10],"torch":"2.7.1+cu128"}'), result('{}\nENV_OK\n')]) as run:
            self.assertEqual(check.choose_environment(gpu=True), ('/candidate', 0))
        self.assertNotIn('--cpu-only', run.call_args_list[1].args[0])

    def test_runtime_error_is_repair_not_automatic_reinstall(self):
        payload = json.dumps({'errors': ['runtime smoke failed: missing native library']})
        with patch.object(check, 'candidates', return_value=iter(['/candidate'])), \
                patch.object(check, 'run', side_effect=[
                    result('{"python":[3,10],"torch":"2.7.1+cu128"}'),
                    result(payload + '\nENV_INCOMPLETE\n', 2)]):
            self.assertEqual(check.choose_environment(), (None, 4))
        self.assertNotIn('NEED_ENV_INSTALL', self.output.getvalue())

    def test_dependency_mismatch_allows_private_install_branch(self):
        payload = json.dumps({'errors': ['dependency versions differ: numpy']})
        with patch.object(check, 'candidates', return_value=iter(['/candidate'])), \
                patch.object(check, 'run', side_effect=[
                    result('{"python":[3,10],"torch":"2.7.1+cu128"}'),
                    result(payload + '\nENV_INCOMPLETE\n', 2)]):
            self.assertEqual(check.choose_environment(), (None, 2))

    def test_failed_candidate_does_not_hide_another_valid_environment(self):
        with patch.object(check, 'candidates', return_value=iter(['/wrong', '/right'])), \
                patch.object(check, 'run', side_effect=[result('{}'),
                    result('{"python":[3,10],"torch":"2.7.1+cu128"}'),
                    result('{}\nENV_CPU_OK_NOT_GPU_ACCEPTANCE\n')]):
            self.assertEqual(check.choose_environment(), ('/right', 0))

    def test_missing_environment_does_not_download_or_prepare_materials(self):
        record = self.write('cache/check-first/python.path', '/old-python\n')
        with patch.object(check, 'verify_sources'), \
                patch.object(check, 'choose_environment', return_value=(None, 2)), \
                patch.object(check, 'install_inputs') as needed, \
                patch.object(check.subprocess, 'run') as run, \
                patch.object(sys, 'argv', ['check_env.py', '--prepare']):
            self.assertEqual(check.main(), 2)
        run.assert_not_called()
        needed.assert_called_once()
        self.assertEqual(record.read_text(), '')
        self.assertIn('NOT_CHECKED', self.output.getvalue())

    def test_worker_uses_exact_checked_interpreter(self):
        with patch.object(check, 'verify_sources'), \
                patch.object(check, 'choose_environment', return_value=('/qualified/python', 0)), \
                patch.object(check.subprocess, 'run', return_value=result()) as run, \
                patch.object(sys, 'argv', ['check_env.py']):
            self.assertEqual(check.main(), 0)
        self.assertEqual(run.call_args.args[0][0], '/qualified/python')
        self.assertEqual((self.root / 'cache/check-first/python.path').read_text(), '/qualified/python\n')
        self.assertIn('CPU_PREPARATION_OK_NOT_GPU_ACCEPTANCE', self.output.getvalue())

    def test_environment_only_does_not_check_materials(self):
        with patch.object(check, 'verify_sources'), \
                patch.object(check, 'choose_environment', return_value=('/qualified/python', 0)), \
                patch.object(check.subprocess, 'run') as run, \
                patch.object(sys, 'argv', ['check_env.py', '--environment-only']):
            self.assertEqual(check.main(), 0)
        run.assert_not_called()

    def test_existing_prepared_data_does_not_require_transport(self):
        self.write('assets/prepared_bundle.json', {})
        verify, restore = Mock(), Mock(side_effect=AssertionError('must not restore'))
        with self.pipeline(prepared_bundle={'verify': verify, 'restore': restore}):
            self.assertTrue(check.check_data(self.root / 'absent-package', prepare=False))
            self.assertTrue(check.check_data(self.root / 'absent-package', prepare=True))
        self.assertEqual(verify.call_count, 2)
        restore.assert_not_called()

    def test_invalid_prepared_data_does_not_fall_back_to_restore(self):
        self.write('assets/prepared_bundle.json', {})
        restore = Mock()
        with self.pipeline(prepared_bundle={'verify': Mock(side_effect=ValueError('changed data')), 'restore': restore}):
            with self.assertRaisesRegex(ValueError, 'changed data'):
                check.check_data(self.root / 'package', prepare=True)
        restore.assert_not_called()

    def test_partial_data_is_preserved(self):
        self.write('assets/data/example', 'partial')
        restore = Mock()
        with self.pipeline(prepared_bundle={'restore': restore}):
            with self.assertRaisesRegex(ValueError, 'preserved'):
                check.check_data(self.root / 'package', prepare=True)
        restore.assert_not_called()
        self.assertEqual((self.root / 'assets/data/example').read_text(), 'partial')

    def test_present_unverified_transport_is_not_called_reusable(self):
        package = self.write('package.tar', 'not actually a valid archive')
        restore = Mock()
        with self.pipeline(prepared_bundle={'restore': restore}):
            self.assertFalse(check.check_data(package, prepare=False))
        restore.assert_not_called()
        self.assertIn('not yet approved', self.output.getvalue())

    def test_prepare_calls_original_restore(self):
        package = self.write('package.tar', 'fixture')
        restore = Mock()
        with self.pipeline(prepared_bundle={'restore': restore}):
            self.assertTrue(check.check_data(package, prepare=True))
        restore.assert_called_once_with(package, self.root / 'assets')

    def test_existing_fitted_is_checked_only(self):
        (self.root / 'assets/fitted').mkdir(parents=True)
        prepare, unpack = Mock(), Mock()
        with self.pipeline(provided_assets={'prepare': prepare, 'unpack': unpack}):
            self.assertTrue(check.check_fitted(self.root.parent, prepare=True))
        prepare.assert_called_once_with(check=True)
        unpack.assert_not_called()

    def fitted_archive(self):
        archive = self.write('materials/fitted_only/parameters-20261023.tar.gz', 'fixture-bytes')
        self.write('resources/reproduction/fitted.json', {'bytes': archive.stat().st_size, 'sha256': check.digest(archive)})
        return archive

    def test_qualified_fitted_archive_is_not_unpacked_during_check(self):
        archive = self.fitted_archive()
        prepare, unpack = Mock(), Mock()
        with self.pipeline(provided_assets={'prepare': prepare, 'unpack': unpack}):
            self.assertFalse(check.check_fitted(archive.parents[1], prepare=False))
        prepare.assert_not_called()
        unpack.assert_not_called()
        self.assertIn('verified local archive', self.output.getvalue())

    def test_fitted_prepare_cannot_invoke_downloader(self):
        archive = self.fitted_archive()
        prepare, unpack = Mock(), Mock()
        with self.pipeline(provided_assets={'prepare': prepare, 'unpack': unpack}):
            self.assertTrue(check.check_fitted(archive.parents[1], prepare=True))
        unpack.assert_called_once_with(archive, self.root / 'assets/fitted')
        prepare.assert_called_once_with(check=True)

    def test_invalid_cache_can_use_another_valid_archive_without_redownload(self):
        archive = self.fitted_archive()
        self.write('cache/fitted/parameters.tar.gz', 'bad')
        with self.pipeline(provided_assets={}):
            self.assertFalse(check.check_fitted(archive.parents[1], prepare=False))
        self.assertNotIn('NEED_DOWNLOAD', self.output.getvalue())

    def test_corrupt_fitted_archive_is_not_approved(self):
        archive = self.fitted_archive()
        archive.write_text('bad')
        with self.pipeline(provided_assets={}):
            self.assertFalse(check.check_fitted(archive.parents[1], prepare=False))
        self.assertIn('NEED_DOWNLOAD', self.output.getvalue())

    def test_external_model_mapping_is_read_only(self):
        mapping = {k: 'qwen' if k in ('qwen', 'generator') else k for k in check.ROLES}
        source = self.write('material-models.json', mapping)
        before = source.read_bytes()
        actual = check.model_mapping(self.root.parent, explicit=source)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(actual['qwen'], actual['generator'])
        self.assertEqual(actual['internvl'], str(self.root / 'internvl'))

    def test_different_qwen_generator_is_rejected(self):
        mapping = {k: k for k in check.ROLES}
        source = self.write('material-models.json', mapping)
        with self.assertRaisesRegex(ValueError, 'same pinned weights'):
            check.model_mapping(self.root.parent, explicit=source)

    def test_model_check_uses_original_verifier_and_temporary_config(self):
        models = {k: str(self.root / 'models' / k) for k in check.ROLES}
        self.write('resources/reproduction/models.json', {'models': {k: {'files': {'config.json': 'unused'}} for k in ('internvl', 'qwen')}})
        self.write('assets/plugguard/base_models.json', {k: {'files': {'config.json': 'unused'}} for k in ('generator', 'judge')})
        for role in check.ROLES:
            self.write('models/' + role + '/config.json', '{}')
        def verify():
            config = Path(os.environ['MSRC_CONFIG'])
            self.assertTrue(config.is_relative_to(self.root))
            self.assertEqual(json.loads(config.read_text())['models'], models)
        os.environ['MSRC_CONFIG'] = 'prior-value'
        verifier = Mock(side_effect=verify)
        with self.pipeline(fast3={'verify_models': verifier}):
            self.assertTrue(check.check_models(models))
        verifier.assert_called_once()
        self.assertEqual(os.environ['MSRC_CONFIG'], 'prior-value')
        self.assertFalse((self.root / 'configs/paths.json').exists())

    def test_missing_model_never_emits_reuse(self):
        self.write('resources/reproduction/models.json', {'models': {k: {'files': {'model.safetensors': 'unused'}} for k in ('internvl', 'qwen')}})
        self.write('assets/plugguard/base_models.json', {})
        verify = Mock()
        with self.pipeline(fast3={'verify_models': verify}):
            self.assertFalse(check.check_models({k: str(self.root / k) for k in check.ROLES}))
        verify.assert_not_called()
        self.assertIn('NEED_DOWNLOAD_OR_PATH', self.output.getvalue())

    def test_configuration_is_never_written_by_default_check(self):
        save = Mock()
        with self.pipeline(common={'save': save, 'root_path': lambda x: self.root / x}):
            self.assertFalse(check.check_configuration({}, prepare=False, inputs_ready=True))
        save.assert_not_called()

    def test_configuration_prepare_waits_for_all_input_checks(self):
        save = Mock()
        with self.pipeline(common={'save': save, 'root_path': lambda x: self.root / x}):
            self.assertFalse(check.check_configuration({}, prepare=True, inputs_ready=False))
        save.assert_not_called()

    def test_runtime_caches_are_project_local_and_offline(self):
        env = check.runtime_environment()
        for key in ('TMPDIR', 'PIP_CACHE_DIR', 'HF_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH'):
            self.assertTrue(Path(env[key]).is_relative_to(self.root))
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['PIP_NO_INDEX'], '1')

    @unittest.skipUnless(BASH, 'bash is not installed')
    def test_shell_files_have_valid_bash_syntax(self):
        for filename in ('activate.sh', 'install_env.sh'):
            outcome = subprocess.run([BASH, '-n', str(SOURCE.parent / filename)], capture_output=True)
            self.assertEqual(outcome.returncode, 0, outcome.stderr)

    @unittest.skipUnless(BASH, 'bash is not installed')
    def test_real_shell_install_entry_skips_an_already_qualified_fixture(self):
        shutil.copyfile(SOURCE.parent / 'install_env.sh', self.root / 'install_env.sh')
        self.write('fixture-bin/python3', '#!/usr/bin/env bash\nprintf "ENV_REUSE_OK\\n"\nexit 0\n')
        self.write('scripts/01_setup_envs.sh', 'printf "UNEXPECTED_INSTALL\\n"\nexit 99\n')
        outcome = subprocess.run([BASH, '--noprofile', '--norc', '-c',
            'export PATH="$PWD/fixture-bin:/usr/bin:/bin"; bash install_env.sh'],
            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        self.assertIn('skip installation', outcome.stdout)
        self.assertNotIn('UNEXPECTED_INSTALL', outcome.stdout)
        self.assertFalse((self.root / 'cache/install-wheels').exists())

    @unittest.skipUnless(BASH, 'bash is not installed')
    def test_real_shell_install_entry_preserves_a_check_error(self):
        shutil.copyfile(SOURCE.parent / 'install_env.sh', self.root / 'install_env.sh')
        self.write('fixture-bin/python3', '#!/usr/bin/env bash\nexit 4\n')
        self.write('scripts/01_setup_envs.sh', 'printf "UNEXPECTED_INSTALL\\n"\nexit 99\n')
        outcome = subprocess.run([BASH, '--noprofile', '--norc', '-c',
            'export PATH="$PWD/fixture-bin:/usr/bin:/bin"; bash install_env.sh'],
            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(outcome.returncode, 4, outcome.stderr)
        self.assertNotIn('UNEXPECTED_INSTALL', outcome.stdout)

    @unittest.skipUnless(BASH, 'bash is not installed')
    def test_real_shell_activation_refuses_an_unchecked_environment(self):
        shutil.copyfile(SOURCE.parent / 'activate.sh', self.root / 'activate.sh')
        outcome = subprocess.run([BASH, '--noprofile', '--norc', '-c',
            'export PATH="/usr/bin:/bin"; source activate.sh'],
            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(outcome.returncode, 1, outcome.stderr)
        self.assertIn('No checked Python', outcome.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
