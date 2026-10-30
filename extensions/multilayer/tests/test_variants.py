import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from extensions.multilayer import runner, variants


class VariantTests(unittest.TestCase):
    def test_all_parameters_are_supplied_and_pair_with_original_head(self):
        manifest, origin = variants.state_package()
        self.assertEqual(set(manifest['files']), variants.STATE_FILES)
        registry = runner.load(runner.common.ROOT / 'resources/reproduction/fitted.json')
        self.assertEqual(origin['baseline_head_sha256'], registry['safety_head_sha256'])
        self.assertEqual(origin['baseline_threshold_sha256'], registry['safety_threshold_sha256'])
        self.assertEqual({k: len(v) for k, v in origin['roles'].items()},
                         dict(fit=2688, calibration=672, test=3360))

    def test_state_is_fitted_only_on_development(self):
        _, origin = variants.state_package()
        calibration = runner.load(variants.STATE / 'calibration.json')
        self.assertFalse(origin['test_used_for_fitting_or_selection'])
        self.assertFalse(calibration['test_used'])
        self.assertEqual(len(calibration['requests']), 672)
        self.assertEqual({r['id'] for r in calibration['requests']}, set(origin['roles']['calibration']))
        self.assertFalse(set(origin['roles']['fit']) & set(origin['roles']['test']))
        self.assertFalse(set(origin['roles']['calibration']) & set(origin['roles']['test']))

    def test_supplied_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            import shutil
            destination = Path(directory) / 'state'
            shutil.copytree(variants.STATE, destination)
            (destination / 'directions.json').write_text('{}')
            with patch.object(variants, 'STATE', destination), self.assertRaises(ValueError):
                variants.state_package()

    def test_algorithm_core_is_not_overridden_in_either_mode(self):
        for name in ('decode_request', 'development', 'label', 'fit', 'full_test', 'judge', 'controls'):
            self.assertIs(getattr(variants.VariantExperiment, name), getattr(runner.Experiment, name))

    def test_fitted_cli_rejects_training(self):
        import sys
        for phase in ('development', 'label', 'fit'):
            with patch.object(sys, 'argv', ['variant', '--mode', 'fitted', '--run-id', 'fresh', '--phase', phase]):
                with self.assertRaises(SystemExit) as error:
                    variants.main()
                self.assertEqual(error.exception.code, 2)

    def test_report_missing_inputs_cannot_be_pass(self):
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old_html = root / 'runs/fresh/multilayer/joint/report.html'
            old_html.parent.mkdir(parents=True)
            for name in ('report.html', 'report.md', 'signal_report.html', 'calibration_report.html', 'warning_report.html'):
                (old_html.parent / name).write_text('ACCEPTANCE_PASS (old synthetic fixture)')
            with patch.object(runner.common, 'ROOT', root), \
                 patch.object(runner.common, 'runs_dir', return_value=root / 'runs'), \
                 patch.dict(sys.modules, {'fcntl': types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None)}), \
                 patch.object(sys, 'argv', ['variant', '--mode', 'fitted', '--run-id', 'fresh', '--phase', 'joint-report']):
                with self.assertRaises(FileNotFoundError):
                    variants.main()
                summary = runner.load(root / 'runs/fresh/multilayer/joint/summary.json')
                self.assertEqual(summary['verdict'], 'INCOMPLETE')
                self.assertIsNone(summary['acceptance_pass'])
                for name in ('report.html', 'report.md', 'signal_report.html', 'calibration_report.html', 'warning_report.html'):
                    text = (old_html.parent / name).read_text(encoding='utf-8')
                    self.assertIn('INCOMPLETE', text)
                    self.assertNotIn('ACCEPTANCE_PASS', text)
                self.assertFalse(runner.load(root / 'runs/fresh/multilayer/joint/outputs.json')['complete'])

    def test_wrong_mode_preparation_is_rejected(self):
        receipt = dict(format='msrc-multilayer-variant-materials-v1', mode='fitted', sources={})
        with patch.object(runner, 'load', return_value=receipt), \
             patch.object(variants, 'variant_sources', return_value={}):
            with self.assertRaisesRegex(ValueError, 'never mix'):
                variants.check_materials(Path('unused'), 'no_fit')

    def test_joint_report_uses_current_metrics_and_both_parameter_modes(self):
        for mode, factual_mode, accuracy, gain, expected in (
                ('fitted', 'provided', .80, .5, 'PASS'),
                ('no_fit', 'fresh_fit', .70, .5, 'FAIL'),
                ('fitted', 'provided', .80, None, 'FAIL')):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                entry = root / 'pipeline/factual.py'
                entry.parent.mkdir()
                entry.write_text('# fixture\n')
                with patch.object(runner.common, 'ROOT', root):
                    factual = root / 'runs/fresh/factual'
                    cells = [dict(cell=str(i), minimum_actual_signals_per_test_sample=3,
                                  ece_before=.2 if gain is not None else 0., ece_after=.1, relative_improvement=gain)
                             for i in range(6)]
                    runner.common.save(factual / 'report.json', dict(cells=cells))
                    runner.common.save(factual / 'protocol.json', dict(
                        parameter_mode=factual_mode, source_hashes={}, script_sha256=runner.sha(entry)))
                    experiment = types.SimpleNamespace(root=root / 'runs/fresh/multilayer')
                    for cell in cells:
                        for suffix in ('.csv', '.jsonl'):
                            path = factual / 'signal_usage' / (cell['cell'] + suffix)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text('synthetic fixture')
                    (factual / 'summary.csv').write_text('synthetic fixture')
                    output = experiment.root / 'experiments/multilayer_001'
                    for name in ('report.json', 'frozen.json', 'live_benchmark.json'):
                        runner.common.save(output / name, {})
                    for name in ('test', 'answers'):
                        (output / 'records' / name).mkdir(parents=True)
                    tp = int(accuracy * 3360) - 700
                    safety = dict(parameter_modes=dict(head=mode), primary=dict(temporal_hold=dict(
                        accuracy=accuracy, TP=tp, TN=700, FP=140, FN=2520-tp, n=3360)))
                    safety['by_source'] = {'temporal_hold': {suite: dict(safety['primary']['temporal_hold'])
                        for suite in ('SD', 'SD_TYPO', 'TYPO', 'benign')}}
                    with patch('pipeline.acceptance_check.factual_check') as factual_check, \
                         patch.object(variants, 'audit_safety', return_value=safety), \
                         patch.object(variants, 'variant_sources', return_value={'fixture': 'test'}):
                        result = variants.joint_report(experiment, mode)
                    factual_check.assert_called_once()
                    self.assertEqual(result['verdict'], expected)
                    self.assertEqual(result['acceptance_pass'], expected == 'PASS')
                    self.assertEqual(result['factual_parameter_mode'], factual_mode)
                    self.assertTrue((experiment.root / 'joint/report.html').is_file())
                    self.assertTrue((experiment.root / 'joint/report.md').is_file())
                    self.assertEqual(result['calibration_aggregate']['passed_cells'], 6 if gain is not None else 0)
                    self.assertIn('六组平均', (experiment.root / 'joint/report.md').read_text(encoding='utf-8'))
                    self.assertEqual(runner.load(experiment.root / 'joint/summary.json')['verdict'], expected)
                    inventory = runner.load(experiment.root / 'joint/outputs.json')
                    for artifact in inventory['artifacts']:
                        self.assertTrue((root / artifact['path']).exists(), artifact)
                        if artifact['kind'] == 'file':
                            self.assertEqual(artifact['sha256'], runner.sha(root / artifact['path']))
                    self.assertIn('joint_markdown', {item['key'] for item in inventory['artifacts']})
                    for name in ('signal_report.html', 'calibration_report.html', 'warning_report.html',
                                 'warning_by_source.csv', 'signal_summary.csv'):
                        self.assertTrue((experiment.root / 'joint' / name).is_file())

    def test_no_fit_recipe_never_downloads_project_parameters_on_its_branch(self):
        text = (runner.common.ROOT / 'scripts/run_multilayer_variant.sh').read_text()
        supplied = text.split('if [ "$VARIANT" = fitted ]; then', 1)[1].split('\nfi\n', 1)[0]
        self.assertIn('02_prepare_fitted.sh', supplied)
        self.assertIn('--refit --execute', text)
        self.assertIn('PHASE=apply', supplied)
        self.assertIn('PHASE=fit', text)
        self.assertIn('require_new_dir', text)


if __name__ == '__main__':
    unittest.main()
