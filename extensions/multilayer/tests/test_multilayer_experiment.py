import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from multilayer_experiment import (ARMS, Experiment, aggregate_label, calibration_record,
                                  judgment_key, validate_commits, validate_prefixes)
from safety_lab.policy import Policy


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.assets = self.base / 'assets'
        self.assets.mkdir()
        self.run = Experiment(self.base / 'run', self.assets)
        self.source = dict(id='one', role='fit', group='g1', suite='benign',
                           split='dev', label=0, question='question')

    def commit(self):
        return dict(id='one', arms=[dict(id='one', arm=arm, group='g1', suite='benign',
                    split='test', label=0, warn=False, trigger_token=None,
                    generated_tokens=3, released_tokens=3, answer='a') for arm in ARMS])

    def test_write_boundary_atomic_identity_and_duplicate_journals(self):
        with self.assertRaises(ValueError):
            self.run.path('../outside.json')
        self.run.commit('development', 'one', dict(id='one', x=1))
        self.run.commit('development', 'one', dict(id='one', x=1))
        with self.assertRaises(ValueError):
            self.run.commit('development', 'one', dict(id='one', x=2))
        with self.assertRaises(ValueError):
            self.run.commit('development', 'two', dict(id='one'))
        directory = self.run.path('records/development')
        (directory / 'duplicate.json').write_text(json.dumps(dict(id='one', x=1)))
        with self.assertRaises(ValueError):
            self.run.records('development')

    def test_resolved_symlink_escape(self):
        outside = self.base / 'outside'
        outside.mkdir()
        try:
            self.run.path('escape').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('symlink creation unavailable')
        with self.assertRaises(ValueError):
            self.run.path('escape/answer.json')

    def test_exact_question_and_prefix_judgment_identity(self):
        key = judgment_key('question', 'answer')
        self.assertNotEqual(key, judgment_key('another question', 'answer'))
        self.assertNotEqual(key, judgment_key('question', 'answer '))
        self.assertNotEqual(judgment_key('ab', 'c'), judgment_key('a', 'bc'))
        self.assertEqual(key, judgment_key('question', 'answer'))

    def test_prefix_coverage_identity_and_unknown_not_safe(self):
        result = dict(self.source, generated_tokens=9, prefixes=[
            dict(position=p, text=str(p), prefix_sha256=hashlib.sha256(str(p).encode()).hexdigest())
            for p in (8, 9)])
        labels = [dict(id='one', role='fit', position=p, label=label,
                      group='g1', input_sha256=judgment_key('question', str(p)),
                      status='safe' if label == 0 else 'controversial',
                      judge_raw='Safety: Safe' if label == 0 else 'Safety: Controversial',
                      label_source='response_prefix_judge',
                      prefix_sha256=prefix['prefix_sha256'])
                  for p, label, prefix in zip((8, 9), (0, None), result['prefixes'])]
        validate_prefixes([result], [self.source], labels, complete=True)
        self.assertIsNone(aggregate_label(labels))
        self.assertEqual(aggregate_label([dict(label=1), dict(label=None)]), 1)
        with self.assertRaises(ValueError):
            validate_prefixes([result], [self.source], labels[:1], complete=True)
        changed = copy.deepcopy(result)
        changed['prefixes'][0]['text'] = 'future changed'
        with self.assertRaises(ValueError):
            validate_prefixes([changed], [self.source], labels)
        labels[0]['input_sha256'] = judgment_key('different question', '8')
        with self.assertRaises(ValueError):
            validate_prefixes([result], [self.source], labels)

    def test_shadow_continues_without_releasing_blocked_tokens(self):
        class Block(torch.nn.Module):
            def forward(self, value):
                return value + 1

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block() for _ in range(4)])
                self.norm = torch.nn.Identity()

            def forward(self, value):
                for layer in self.layers:
                    value = layer(value)
                return self.norm(value)

        class Model:
            generation_config = SimpleNamespace(eos_token_id=7)

            def __init__(self):
                self.model = SimpleNamespace(language_model=Decoder())
                self.position = 0

            def _prepare_position_ids_for_generation(self, ids, kwargs):
                return None

            def prepare_inputs_for_generation(self, ids, is_first_iteration=False, **kwargs):
                return dict(ids=ids if is_first_iteration else ids[:, -1:], first=is_first_iteration)

            def _update_model_kwargs_for_generation(self, out, kwargs):
                return kwargs

            def __call__(self, ids, first, **kwargs):
                self.position = 0 if first else self.position + 1
                self.model.language_model(ids.float().unsqueeze(-1).expand(-1, -1, 2))
                logits = torch.full((1, 1, 8), -10.)
                logits[0, 0, self.position + 1 if self.position < 3 else 7] = 10.
                return SimpleNamespace(logits=logits)

        def step(head, hidden, state):
            risk = [.1, .9, .1][state]
            return torch.tensor([[math.log(1 - risk), math.log(risk)]]), state + 1

        import math
        self.run.policy = Policy(.8, None)
        self.run.torch = torch
        self.run.pg = SimpleNamespace(
            inputs_for=lambda proc, row: dict(input_ids=torch.tensor([[0]])),
            native_greedy_processors=lambda *args: lambda ids, logits: logits,
            prefix_init=lambda *args: 0, online_step=step,
            prepare_decode_inputs=lambda model, ids, kwargs: model.prepare_inputs_for_generation(ids, **kwargs))
        processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: ','.join(map(str, ids))))
        fitted = dict(weight=np.zeros((3, 2)), bias=np.full(3, 2.), semantic=1., temporal=None, hold_threshold=None)
        with patch.object(torch.cuda, 'synchronize'):
            arms, tokens, _, _ = self.run.decode_request(Model(), processor, None, self.source, fitted)
        indexed = {r['arm']: r for r in arms}
        self.assertEqual(tokens, [1, 2, 3])
        self.assertEqual(indexed['baseline']['trigger_token'], 2)
        self.assertEqual(indexed['semantic']['trigger_token'], 1)
        self.assertTrue(all(r['released_tokens'] == 0 for r in arms))
        self.assertEqual(len(indexed['semantic']['events']), 1)

    def test_five_arm_coverage_and_baseline_constraints(self):
        source = dict(self.source, split='test', role='test')
        commit = self.commit()
        validate_commits([commit], {'one': source}, complete=True)
        with self.assertRaises(ValueError):
            validate_commits([commit, commit], {'one': source})
        with self.assertRaises(ValueError):
            validate_commits([dict(id='one', arms=commit['arms'][:-1])], {'one': source})
        commit['arms'][0].update(warn=True, trigger_token=2, released_tokens=0, generated_tokens=2)
        with self.assertRaises(ValueError):
            validate_commits([commit], {'one': source})

    def test_reachable_calibration_excludes_after_controller_stop(self):
        source = dict(self.source, role='calibration')
        policy = Policy(.8, .7)
        record = calibration_record(source, 1, [.1, .9, .1], [.1] * 3,
                                    np.array([[0.] * 3, [0.] * 3, [100.] * 3]), policy)
        self.assertEqual(record['controller_stop'], 2)
        self.assertEqual(record['semantic']['score'], 0.)
        self.assertEqual(record['semantic']['incremental_score'], 0.)
        self.assertTrue(record['semantic']['baseline_warn'])

    def test_frozen_hash_guards_and_test_before_freeze(self):
        self.run.save('protocol.json', dict(sources={}))
        with self.assertRaises(ValueError):
            self.run.verify_frozen()
        artifact = self.run.path('model.json')
        self.run.save('model.json', dict(weight=[1]))
        frozen = dict(files={'model.json': hashlib.sha256(artifact.read_bytes()).hexdigest()})
        self.run.save('frozen.json', frozen)
        self.run.verify_frozen(check_sources=False)
        with patch.object(self.run, 'sources', return_value={'unexpected': 'source'}):
            with self.assertRaises(ValueError):
                self.run.verify_frozen()
        self.run.save('model.json', dict(weight=[2]))
        with self.assertRaises(ValueError):
            self.run.verify_frozen(check_sources=False)

    def test_inherited_controller_changes_are_not_new_detector_success(self):
        source = dict(self.source, role='test', split='test', label=1)
        commit = self.commit()
        for arm in commit['arms']:
            n = 3 if arm['arm'] == 'baseline' else 2
            arm.update(label=1, warn=True, trigger_token=n, generated_tokens=n, released_tokens=0)
        self.run.tests = {'one': source}
        self.run.commit('test', 'one', commit)
        labels = [dict(arm=arm, input_sha256=judgment_key('question', 'a'),
                       judge_raw='Safety: Safe', label=0, status='safe') for arm in ARMS]
        self.run.commit('answers', 'one', dict(id='one', labels=labels))
        self.run.save('calibration.json', dict(selections={mode: dict(enabled=False) for mode in ('semantic', 'temporal')}, requests=[]))
        with patch.object(self.run, 'verify_frozen', return_value={}), patch.object(self.run, 'status'):
            report = self.run.report()
        self.assertTrue(report['meets_75_percent']['temporal'])
        self.assertEqual(report['changes']['temporal']['earlier_warnings'], 1)
        self.assertEqual(report['changes']['temporal']['earlier_vs_controller'], 0)
        self.assertFalse(report['new_detection_changes_observed'])


if __name__ == '__main__':
    unittest.main()
