from dataclasses import asdict
import unittest

from safety_lab.multilayer_policy import Evidence, MultiLayerController, select_threshold
from safety_lab.policy import Policy, StreamController


class MultilayerPolicyTests(unittest.TestCase):
    def records(self):
        return [dict(id=str(i), group=str(i), role='calibration', label=0,
                     score=float(i), incremental_score=float(i), baseline_warn=False)
                for i in range(20)] + [dict(id='u', group='u', role='calibration',
                    label=1, score=100., incremental_score=100., baseline_warn=False)]

    def test_calibration_order_statistic_budget_and_strict_ties(self):
        result = select_threshold(self.records(), min_groups=1)
        self.assertTrue(result['enabled'])
        self.assertEqual(result['threshold'], 18)
        self.assertEqual(result['safe_false_alarms'], 1)
        tied = self.records()
        for r in tied[:20]:
            r['score'] = r['incremental_score'] = 18
        result = select_threshold(tied, min_groups=1)
        self.assertEqual(result['safe_false_alarms'], 0)

    def test_post_stop_score_cannot_enable_detector(self):
        records = self.records()
        records[-1]['incremental_score'] = None
        result = select_threshold(records, min_groups=1)
        self.assertFalse(result['enabled'])
        self.assertIsNone(result['threshold'])
        self.assertEqual(result['candidate_threshold'], 18)

    def test_unknowns_baseline_alarm_and_group_counts(self):
        records = self.records()
        records[19]['label'] = None
        records[18]['baseline_warn'] = True
        result = select_threshold(records, min_groups=1)
        self.assertEqual(result['safe_requests'], 18)
        with self.assertRaisesRegex(ValueError, 'groups'):
            select_threshold(records)
        records[0]['role'] = 'test'
        with self.assertRaisesRegex(ValueError, 'calibration'):
            select_threshold(records, min_groups=1)

    def test_evidence_equation_and_future_independence(self):
        evidence = Evidence()
        score = evidence.step([1., 1., 1.])
        self.assertAlmostEqual(score['temporal'], .75)
        self.assertEqual(score['semantic'], 1)
        self.assertEqual(score['disagreement'], 0)
        next_score = evidence.step([1., 1., 1.])
        self.assertAlmostEqual(next_score['temporal'], .9 * .75 + .5 + .25 * .8)
        self.assertAlmostEqual(score['temporal'], .75)
        before = (evidence.mean, evidence.value)
        with self.assertRaises(ValueError):
            evidence.step([float('nan'), 0., 0.])
        self.assertEqual(before, (evidence.mean, evidence.value))

    def test_earlier_alarm_and_baseline_priority(self):
        policy = Policy(.9, None)
        c = MultiLayerController(policy, 1., 'semantic')
        self.assertFalse(c.step(.1, .1, [1., 1., 1.]).alarm)
        d = c.step(.1, .1, [2., 2., 2.])
        self.assertTrue(d.alarm)
        self.assertEqual(d.reason, 'multilayer_semantic_alarm')
        self.assertEqual(d.released, 0)
        with self.assertRaises(RuntimeError):
            c.finish()
        c = MultiLayerController(policy, 0., 'temporal')
        self.assertEqual(c.step(.95, .1, [100.] * 3).reason, 'baseline_hard_alarm')

    def test_disagreement_holds_but_cannot_alarm_or_cancel_existing_hold(self):
        policy = Policy(.9, None)
        c = MultiLayerController(policy, None, 'temporal', hold_threshold=.5)
        reference = StreamController(policy)
        for i in range(70):
            z = [-1., 0., 1.] if i < 40 else [0., 0., 0.]
            d = c.step(.1, .8 if i >= 40 else .1, z)
            expected = reference.step(.1, .8 if i >= 40 else .1)
            self.assertFalse(d.alarm)
            self.assertLessEqual(d.released, expected.released)
            self.assertLessEqual(d.pending, 64)
            self.assertEqual(d.state, 'HOLD')
        self.assertEqual(c.finish().released, 70)

    def test_invalid_input_does_not_mutate_state(self):
        c = MultiLayerController(Policy(.9, None), 2., 'semantic')
        c.step(.1, .2, [0., 0., 0.])
        before = (c.generated, c.evidence.mean, c.reference.generated)
        for risk, z in ((1.1, [0., 0., 0.]), (.1, [0., float('inf'), 0.])):
            with self.assertRaises(ValueError):
                c.step(risk, .1, z)
            self.assertEqual(before, (c.generated, c.evidence.mean, c.reference.generated))

    def test_disabled_detector_matches_existing_controller(self):
        policy = Policy(.9, .7)
        candidate = MultiLayerController(policy, None, 'temporal')
        reference = StreamController(policy)
        for _ in range(30):
            actual = candidate.step(.1, .2, [0.] * 3)
            expected = reference.step(.1, .2)
            for field in ('alarm', 'reason', 'state', 'generated', 'released', 'pending'):
                self.assertEqual(getattr(actual, field), getattr(expected, field))
        self.assertEqual(candidate.finish().released, reference.finish().released)


if __name__ == '__main__':
    unittest.main()
