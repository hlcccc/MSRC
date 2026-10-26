import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from extensions.multilayer import runner


class DeliveryTests(unittest.TestCase):
    def test_core_matches_byte_manifest(self):
        self.assertEqual(len(runner.verify_core()), 11)

    def test_head_architecture_and_base_identities_are_frozen_sources(self):
        sources = runner.sources()
        self.assertEqual(sources['references/plugguard/models.py'],
                         'c16d237c5634bcaae24f6bedaf5075479fcf22e395c9ecfff081715fdf5d0479')
        self.assertIn('assets/plugguard/base_models.json', sources)

    def test_publication_receipt_matches_checkout_bytes(self):
        root = runner.common.ROOT
        receipt = runner.load(root / 'resources/multilayer/verification.json')
        for name, key in (('extensions/multilayer/runner.py', 'runner_sha256'),
                          ('resources/multilayer/source_manifest.json', 'frozen_core_manifest_sha256')):
            self.assertEqual(runner.sha(root / name), receipt[key])
        import hashlib
        requirements = (root / 'requirements-acceptance.txt').read_bytes().replace(b'\r\n', b'\n')
        self.assertEqual(hashlib.sha256(requirements).hexdigest(), receipt['requirements_sha256_lf'])
        qa = runner.load(root / 'resources/multilayer/document_qa.json')
        self.assertEqual(runner.sha(root / 'docs/FLOW_SAFETY_MULTILAYER.md'), qa['source_sha256'])
        self.assertEqual(runner.sha(root / 'docs/FLOW_SAFETY_MULTILAYER.docx'), qa['docx_sha256'])

    def test_tampered_core_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / 'core'
            shutil.copytree(runner.FROZEN, target)
            (target / 'multilayer_experiment.py').write_text('changed')
            with patch.object(runner, 'FROZEN', target), self.assertRaises(ValueError):
                runner.verify_core()

    def test_run_identity_cannot_escape(self):
        for identity in ('', '..', '../old', 'a/b', 'a\\b', '/abs', 'a b'):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                runner.run_root(identity)
        self.assertTrue(runner.run_root('fresh_001').is_relative_to(runner.common.ROOT))

    def test_all_three_indicators_are_required(self):
        cells = [dict(minimum_actual_signals_per_test_sample=3, relative_improvement=.10) for _ in range(6)]
        safety = dict(primary=dict(temporal_hold=dict(accuracy=.75)))
        self.assertTrue(runner.verdict(cells, safety)['acceptance_pass'])
        cells[0]['relative_improvement'] = .099
        self.assertFalse(runner.verdict(cells, safety)['acceptance_pass'])
        cells[0]['relative_improvement'] = .10
        cells[1]['minimum_actual_signals_per_test_sample'] = 2
        self.assertFalse(runner.verdict(cells, safety)['acceptance_pass'])
        self.assertFalse(runner.verdict(cells[:5], safety)['acceptance_pass'])
        cells[1]['minimum_actual_signals_per_test_sample'] = 3
        safety['primary']['temporal_hold']['accuracy'] = .7499
        self.assertFalse(runner.verdict(cells, safety)['acceptance_pass'])

    def arm(self):
        return dict(start_ns=20, generated_tokens=2, released_tokens=0, warn=True,
            events=[dict(position=position, score_ns=20 + position, release_ns=None,
                decision=dict(generated=position, released=0, pending=1 if position == 1 else 0,
                              alarm=position == 2)) for position in (1, 2)])

    def test_real_time_gate_score_precedes_release(self):
        arm = self.arm()
        self.assertEqual(runner.audit_events(arm, 10), 2)
        arm.update(warn=False, released_tokens=2)
        arm['events'][1].update(release_ns=23)
        arm['events'][1]['decision'].update(alarm=False, released=1)
        self.assertEqual(runner.audit_events(arm, 10), 2)
        arm['events'][1]['release_ns'] = 21
        with self.assertRaises(ValueError):
            runner.audit_events(arm, 10)

    def test_pre_freeze_test_and_post_warning_release_rejected(self):
        with self.assertRaises(ValueError):
            runner.audit_events(self.arm(), 30)
        arm = self.arm()
        arm['events'][1]['decision']['released'] = 1
        with self.assertRaises(ValueError):
            runner.audit_events(arm, 10)
        arm = self.arm()
        arm['events'][1]['position'] = 3
        with self.assertRaises(ValueError):
            runner.audit_events(arm, 10)

    def test_decreasing_score_timestamps_rejected(self):
        arm = self.arm()
        arm['events'][1]['score_ns'] = 20
        with self.assertRaises(ValueError):
            runner.audit_events(arm, 10)

    def test_next_score_cannot_precede_previous_release(self):
        arm = self.arm()
        arm['events'][0].update(release_ns=24)
        arm['events'][0]['decision']['released'] = 1
        arm['events'][1]['decision']['released'] = 1
        arm['released_tokens'] = 1
        with self.assertRaises(ValueError):
            runner.audit_events(arm, 10)

    def test_missing_materials_records_incomplete_not_pass(self):
        with tempfile.TemporaryDirectory() as name, patch.object(runner.common, 'ROOT', Path(name).resolve()):
            root = Path(name).resolve() / 'fresh'
            with self.assertRaises(FileNotFoundError):
                runner.checked_joint_report(root, 'fresh')
            summary = runner.load(root / 'joint/summary.json')
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['verdict'], 'INCOMPLETE')
            self.assertIsNone(summary['acceptance_pass'])

    def test_failed_joint_audit_records_incomplete_not_pass(self):
        with tempfile.TemporaryDirectory() as name, patch.object(runner.common, 'ROOT', Path(name).resolve()):
            root = Path(name).resolve() / 'fresh'
            with patch.object(runner, 'check_materials'), patch.object(runner, 'PortableExperiment'), \
                    patch.object(runner, 'joint_report', side_effect=ValueError('source changed')):
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    runner.checked_joint_report(root, 'fresh')
            self.assertEqual(runner.load(root / 'joint/summary.json')['verdict'], 'INCOMPLETE')

    def test_original_four_variant_entries_stay_separate(self):
        root = runner.common.ROOT
        for name in ('run_factual_no_fit.sh', 'run_factual_fitted.sh',
                     'run_safety_no_fit.sh', 'run_safety_fitted.sh', 'run_acceptance_variant.sh'):
            self.assertNotIn('multilayer', (root / 'scripts' / name).read_text())
        entry = (root / 'scripts/run_safety_multilayer.sh').read_text()
        self.assertLess(entry.index('--phase prepare'), entry.index('--phase collect'))
        self.assertLess(entry.index('--phase fit'), entry.index('--phase all'))
        self.assertIn('require_new_dir', entry)

    def test_public_recipe_has_no_test_predictions_or_base_weights(self):
        registry = runner.load(runner.common.ROOT / 'resources/reproduction/fitted.json')
        self.assertFalse(registry['contains_test_predictions'])
        self.assertFalse(registry['contains_base_model_weights'])
        controller = runner.load(runner.common.ROOT / 'resources/multilayer/controller.json')
        self.assertFalse(controller['test_used_for_selection'])
        self.assertEqual(controller['controller']['hard_threshold'], registry['safety_threshold'])

    def test_setup_is_only_transport_not_algorithm_override(self):
        for method in ('decode_request', 'development', 'label', 'fit', 'full_test', 'judge', 'controls'):
            self.assertIs(getattr(runner.PortableExperiment, method), getattr(runner.Experiment, method))


if __name__ == '__main__':
    unittest.main()
