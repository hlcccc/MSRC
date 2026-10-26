"""Contract tests for the causal streaming safety controller."""

import dataclasses
import math
import random
import unittest

from safety_lab.policy import Policy, StreamController


DECISION_FIELDS = (
    "alarm", "reason", "state", "generated", "released", "pending",
    "evidence", "buffer_limit",
)


def make_policy(**overrides):
    parameters = {"hard_threshold": 1.0, "soft_threshold": 0.5}
    parameters.update(overrides)
    return Policy(**parameters)


def decision_snapshot(decision):
    return tuple(getattr(decision, field) for field in DECISION_FIELDS)


class PolicyTests(unittest.TestCase):
    def assert_invalid_parameter(self, name, values):
        for value in values:
            with self.subTest(parameter=name, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    make_policy(**{name: value})

    def test_defaults_and_immutable_dataclass(self):
        policy = Policy(hard_threshold=0.9725, soft_threshold=0.7)
        self.assertTrue(dataclasses.is_dataclass(policy))
        expected = {
            "hard_threshold": 0.9725,
            "soft_threshold": 0.7,
            "evidence_budget": 2.5,
            "decay": 0.8,
            "base_buffer": 16,
            "hold_buffer": 64,
        }
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(policy, name), value)
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(policy, name, value)
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    delattr(policy, name)

    def test_valid_parameter_boundaries(self):
        cases = (
            {"hard_threshold": 1.0, "soft_threshold": 0.0},
            {"hard_threshold": 0.001, "soft_threshold": 0.0},
            {"soft_threshold": math.nextafter(1.0, 0.0)},
            {"soft_threshold": None},
            {"decay": 0.0},
            {"decay": math.nextafter(1.0, 0.0)},
            {"evidence_budget": 1e-12},
            {"base_buffer": 16, "hold_buffer": 16},
            {"base_buffer": 32, "hold_buffer": 128},
        )
        for parameters in cases:
            with self.subTest(parameters=parameters):
                policy = make_policy(**parameters)
                for name, value in parameters.items():
                    self.assertEqual(getattr(policy, name), value)

    def test_invalid_hard_threshold(self):
        self.assert_invalid_parameter("hard_threshold", (
            0.0, -1.0, math.nextafter(1.0, math.inf),
            math.nan, math.inf, -math.inf, None, "1.0", 1j,
        ))

    def test_invalid_soft_threshold_including_equality_with_hard(self):
        self.assert_invalid_parameter("soft_threshold", (
            -1e-12, 1.0, 1.1, math.nan, math.inf, -math.inf, "0.5", 1j,
        ))
        with self.assertRaises((TypeError, ValueError)):
            make_policy(hard_threshold=0.75, soft_threshold=0.8)

    def test_invalid_evidence_budget(self):
        self.assert_invalid_parameter("evidence_budget", (
            0.0, -1.0, math.nan, math.inf, -math.inf, None, "2.5", 1j,
        ))

    def test_invalid_decay(self):
        self.assert_invalid_parameter("decay", (
            -1e-12, 1.0, 1.1, math.nan, math.inf, -math.inf, None, "0.8", 1j,
        ))

    def test_buffers_require_integers_at_least_sixteen(self):
        for name in ("base_buffer", "hold_buffer"):
            self.assert_invalid_parameter(name, (
                -1, 0, 15, 16.0, 16.5, math.nan, math.inf, None, "16", True,
            ))

    def test_hold_buffer_cannot_be_smaller_than_base(self):
        with self.assertRaises((TypeError, ValueError)):
            make_policy(base_buffer=32, hold_buffer=31)


class StreamControllerTests(unittest.TestCase):
    def assert_counts(self, decision, generated, released, pending=None):
        self.assertEqual(decision.generated, generated)
        self.assertEqual(decision.released, released)
        if pending is not None:
            self.assertEqual(decision.pending, pending)
        for value in (decision.generated, decision.released, decision.pending):
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)
        if decision.alarm:
            self.assertLessEqual(decision.released + decision.pending, generated)
        else:
            self.assertEqual(decision.released + decision.pending, generated)

    def assert_same_decision(self, actual, expected):
        self.assertEqual(decision_snapshot(actual), decision_snapshot(expected))

    def trace(self, policy, events):
        controller = StreamController(policy)
        decisions = []
        for risk, uncertainty in events:
            decision = controller.step(risk, uncertainty=uncertainty)
            decisions.append(decision_snapshot(decision))
            if decision.alarm:
                break
        else:
            controller.finish()
        return decisions

    def test_normal_buffer_releases_only_after_strict_overflow(self):
        controller = StreamController(make_policy())
        for generated in range(1, 81):
            with self.subTest(generated=generated):
                decision = controller.step(0.0)
                self.assertFalse(decision.alarm)
                self.assert_counts(
                    decision, generated, max(0, generated - 16), min(generated, 16),
                )
                self.assertEqual(decision.buffer_limit, 16)
                self.assertEqual(decision.evidence, 0.0)

    def test_configured_buffer_limits_are_used(self):
        controller = StreamController(make_policy(base_buffer=24, hold_buffer=40))
        for generated in range(1, 31):
            decision = controller.step(0.0)
            self.assert_counts(
                decision, generated, max(0, generated - 24), min(generated, 24),
            )
            self.assertEqual(decision.buffer_limit, 24)
        for _ in range(20):
            decision = controller.step(0.4)
        self.assert_counts(decision, 50, 10, 40)
        self.assertEqual(decision.buffer_limit, 40)

    def test_hard_threshold_is_inclusive_and_does_not_release_pending(self):
        for hard in (0.25, 0.9725, 1.0):
            for risk in (hard, math.nextafter(hard, math.inf), 1.0):
                if risk > 1.0:
                    continue
                with self.subTest(hard=hard, risk=risk):
                    controller = StreamController(make_policy(
                        hard_threshold=hard, soft_threshold=None,
                    ))
                    for _ in range(17):
                        previous = controller.step(0.0)
                    decision = controller.step(risk)
                    self.assertTrue(decision.alarm)
                    self.assertTrue(decision.reason)
                    self.assert_counts(decision, 18, previous.released)

    def test_hard_alarm_on_first_token(self):
        decision = StreamController(make_policy()).step(1.0)
        self.assertTrue(decision.alarm)
        self.assert_counts(decision, 1, 0)

    def test_hard_alarm_does_not_release_a_full_hold_buffer(self):
        controller = StreamController(make_policy())
        for _ in range(64):
            previous = controller.step(0.4)
        self.assert_counts(previous, 64, 0, 64)
        decision = controller.step(1.0)
        self.assertTrue(decision.alarm)
        self.assert_counts(decision, 65, 0)

    def test_score_immediately_below_hard_does_not_hard_alarm(self):
        for hard in (0.25, 0.9725, 1.0):
            with self.subTest(hard=hard):
                controller = StreamController(make_policy(
                    hard_threshold=hard, soft_threshold=None,
                ))
                self.assertFalse(controller.step(math.nextafter(hard, 0.0)).alarm)

    def test_disabled_soft_preserves_baseline_at_every_token(self):
        hard = 0.9725
        policy = make_policy(
            hard_threshold=hard, soft_threshold=None,
            evidence_budget=1e-12, decay=0.95, hold_buffer=128,
        )
        rng = random.Random(20261023)
        scores = [0.0, math.nextafter(hard, 0.0), 0.8, 0.95]
        scores += [rng.uniform(0.0, math.nextafter(hard, 0.0)) for _ in range(120)]
        controller = StreamController(policy)
        normal_state = None
        for generated, risk in enumerate(scores, 1):
            with self.subTest(generated=generated):
                decision = controller.step(risk)
                if generated == 1:
                    normal_state = decision.state
                self.assertFalse(decision.alarm)
                self.assertEqual(decision.state, normal_state)
                self.assertEqual(decision.evidence, 0.0)
                self.assertEqual(decision.buffer_limit, 16)
                self.assert_counts(
                    decision, generated, max(0, generated - 16), min(generated, 16),
                )
        final = controller.finish()
        self.assertFalse(final.alarm)
        self.assert_counts(final, len(scores), len(scores), 0)

    def test_evidence_uses_normalized_positive_excess(self):
        controller = StreamController(make_policy(
            hard_threshold=0.9, soft_threshold=0.5,
            decay=0.0, evidence_budget=10.0,
        ))
        for risk in (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, math.nextafter(0.9, 0.0)):
            with self.subTest(risk=risk):
                decision = controller.step(risk)
                expected = min(1.0, max(0.0, (risk - 0.5) / (0.9 - 0.5)))
                self.assertAlmostEqual(decision.evidence, expected, places=12)
                self.assertFalse(decision.alarm)

    def test_evidence_decays_without_negative_contributions(self):
        controller = StreamController(make_policy())
        self.assertAlmostEqual(controller.step(0.75).evidence, 0.5)
        expected = 0.5
        for risk in (0.5, 0.25, 0.0, 0.1, 0.5):
            expected *= 0.8
            decision = controller.step(risk)
            self.assertAlmostEqual(decision.evidence, expected, places=12)
            self.assertFalse(decision.alarm)

    def test_near_hard_evidence_increment_is_bounded_by_one(self):
        policy = make_policy(hard_threshold=0.8, evidence_budget=10.0)
        controller = StreamController(policy)
        risk = math.nextafter(policy.hard_threshold, 0.0)
        expected_increment = min(1.0, (risk - 0.5) / (0.8 - 0.5))
        previous = 0.0
        for _ in range(8):
            decision = controller.step(risk)
            increment = decision.evidence - policy.decay * previous
            self.assertLessEqual(increment, 1.0)
            self.assertAlmostEqual(increment, expected_increment, places=12)
            self.assertFalse(decision.alarm)
            previous = decision.evidence

    def test_default_accumulation_alarms_at_the_first_budget_crossing(self):
        controller = StreamController(make_policy())
        expected = 0.0
        for generated in range(1, 6):
            expected = 0.8 * expected + 0.75
            decision = controller.step(0.875)
            self.assertAlmostEqual(decision.evidence, expected, places=12)
            self.assertEqual(decision.alarm, generated == 5)
            self.assert_counts(decision, generated, 0)
        self.assertGreater(decision.evidence, 2.5)
        self.assertTrue(decision.reason)

    def test_evidence_budget_equality_alarms(self):
        controller = StreamController(make_policy(decay=0.5, evidence_budget=1.125))
        first = controller.step(0.875)
        self.assertFalse(first.alarm)
        self.assertEqual(first.evidence, 0.75)
        second = controller.step(0.875)
        self.assertTrue(second.alarm)
        self.assertEqual(second.evidence, 1.125)
        self.assert_counts(second, 2, 0)

    def test_soft_alarm_does_not_release_pending_tokens(self):
        controller = StreamController(make_policy())
        for _ in range(20):
            previous = controller.step(0.0)
        for _ in range(5):
            decision = controller.step(0.875)
        self.assertTrue(decision.alarm)
        self.assert_counts(decision, 25, previous.released)

    def test_zero_soft_threshold_has_well_defined_evidence(self):
        controller = StreamController(make_policy(
            soft_threshold=0.0, decay=0.0, evidence_budget=10.0,
        ))
        for risk in (0.0, 0.25, 0.5, 0.75):
            decision = controller.step(risk)
            self.assertEqual(decision.evidence, risk)
            self.assertFalse(decision.alarm)

    def test_hysteresis_entry_and_exit_boundaries(self):
        controller = StreamController(make_policy())
        normal = controller.step(math.nextafter(0.4, 0.0))
        self.assertEqual(normal.buffer_limit, 16)
        held = controller.step(0.4)
        self.assertNotEqual(held.state, normal.state)
        self.assertEqual(held.buffer_limit, 64)
        for risk in (0.35, 0.3):
            decision = controller.step(risk)
            self.assertEqual(decision.state, held.state)
            self.assertEqual(decision.buffer_limit, 64)
            self.assertEqual(decision.evidence, 0.0)
        cleared = controller.step(math.nextafter(0.3, 0.0))
        self.assertEqual(cleared.state, normal.state)
        self.assertEqual(cleared.buffer_limit, 16)

    def test_low_risk_alone_cannot_clear_high_residual_evidence(self):
        controller = StreamController(make_policy())
        held = controller.step(0.95)
        still_held = controller.step(0.0)
        self.assertAlmostEqual(still_held.evidence, 0.72)
        self.assertEqual(still_held.state, held.state)
        self.assertEqual(still_held.buffer_limit, 64)
        cleared = controller.step(0.0)
        self.assertAlmostEqual(cleared.evidence, 0.576)
        self.assertNotEqual(cleared.state, held.state)
        self.assertEqual(cleared.buffer_limit, 16)

    def test_evidence_must_be_strictly_below_quarter_budget_to_clear(self):
        controller = StreamController(make_policy(decay=0.5, evidence_budget=1.0))
        held = controller.step(0.75)
        boundary = controller.step(0.0)
        self.assertEqual(boundary.evidence, 0.25)
        self.assertEqual(boundary.state, held.state)
        self.assertEqual(boundary.buffer_limit, 64)
        cleared = controller.step(0.0)
        self.assertEqual(cleared.evidence, 0.125)
        self.assertEqual(cleared.buffer_limit, 16)

    def test_sustained_holding_releases_only_after_hold_buffer_overflow(self):
        controller = StreamController(make_policy())
        for generated in range(1, 81):
            decision = controller.step(0.4)
            self.assertFalse(decision.alarm)
            self.assertEqual(decision.evidence, 0.0)
            self.assertEqual(decision.buffer_limit, 64)
            self.assert_counts(
                decision, generated, max(0, generated - 64), min(generated, 64),
            )

    def test_lowering_buffer_releases_all_excess_pending_tokens(self):
        controller = StreamController(make_policy())
        for _ in range(70):
            held = controller.step(0.4)
        self.assert_counts(held, 70, 6, 64)
        cleared = controller.step(0.0)
        self.assertEqual(cleared.buffer_limit, 16)
        self.assert_counts(cleared, 71, 55, 16)
        self.assertEqual(cleared.released - held.released, 49)

    def test_optional_uncertainty_none_matches_omitted_argument(self):
        omitted = StreamController(make_policy())
        explicit = StreamController(make_policy())
        rng = random.Random(701)
        for _ in range(100):
            risk = rng.choice((0.0, 0.3, 0.4, 0.5, 0.625, 0.75))
            self.assert_same_decision(omitted.step(risk), explicit.step(risk, None))
        self.assert_same_decision(omitted.finish(), explicit.finish())

    def test_uncertainty_hold_threshold_is_inclusive(self):
        controller = StreamController(make_policy())
        normal = controller.step(0.0, uncertainty=math.nextafter(0.7, 0.0))
        self.assertEqual(normal.buffer_limit, 16)
        held = controller.step(0.0, uncertainty=0.7)
        self.assertNotEqual(held.state, normal.state)
        self.assertEqual(held.buffer_limit, 64)
        extended = controller.step(0.0, uncertainty=1.0)
        self.assertEqual(extended.state, held.state)
        cleared = controller.step(0.0, uncertainty=math.nextafter(0.7, 0.0))
        self.assertEqual(cleared.state, normal.state)
        self.assertEqual(cleared.buffer_limit, 16)

    def test_uncertainty_alone_cannot_accumulate_evidence_or_alarm(self):
        controller = StreamController(make_policy(evidence_budget=1e-12))
        for generated in range(1, 201):
            decision = controller.step(0.0, uncertainty=1.0)
            self.assertFalse(decision.alarm)
            self.assertEqual(decision.evidence, 0.0)
            self.assertEqual(decision.buffer_limit, 64)
            self.assert_counts(
                decision, generated, max(0, generated - 64), min(generated, 64),
            )
        self.assert_counts(controller.finish(), 200, 200, 0)

    def test_uncertainty_does_not_change_risk_evidence_or_suppress_soft_alarm(self):
        plain = StreamController(make_policy())
        uncertain = StreamController(make_policy())
        for generated in range(1, 6):
            expected = plain.step(0.875)
            actual = uncertain.step(0.875, uncertainty=1.0)
            self.assertEqual(actual.evidence, expected.evidence)
            self.assertEqual(actual.alarm, generated == 5)
            self.assertEqual(actual.alarm, expected.alarm)

    def test_uncertainty_cannot_suppress_hard_alarm(self):
        for uncertainty in (None, 0.0, 0.7, 1.0):
            with self.subTest(uncertainty=uncertainty):
                decision = StreamController(make_policy()).step(1.0, uncertainty)
                self.assertTrue(decision.alarm)
                self.assert_counts(decision, 1, 0)

    def assert_rejected_step_is_atomic(self, prefix, **invalid_step):
        policy = make_policy()
        actual = StreamController(policy)
        reference = StreamController(policy)
        for risk in prefix:
            self.assert_same_decision(actual.step(risk), reference.step(risk))
        with self.assertRaises((TypeError, ValueError)):
            actual.step(**invalid_step)
        for risk in (0.35, 0.0, 0.25, 0.0) + (0.0,) * 20:
            self.assert_same_decision(actual.step(risk), reference.step(risk))
        self.assert_same_decision(actual.finish(), reference.finish())

    def test_invalid_risk_rejects_without_mutating_any_observable_state(self):
        invalid = (
            -1.0, math.nextafter(0.0, -math.inf),
            math.nextafter(1.0, math.inf), 2.0,
            math.nan, math.inf, -math.inf, None, "0.5", 1j,
        )
        for prefix in ((), (0.0,) * 20, (0.0,) * 20 + (0.75,)):
            for risk in invalid:
                with self.subTest(prefix=prefix, risk=risk):
                    self.assert_rejected_step_is_atomic(
                        prefix, risk=risk, uncertainty=1.0,
                    )

    def test_invalid_uncertainty_is_atomic_even_with_hard_alarm_risk(self):
        invalid = (
            -1.0, math.nextafter(0.0, -math.inf),
            math.nextafter(1.0, math.inf), 2.0,
            math.nan, math.inf, -math.inf, "0.5", 1j,
        )
        for prefix in ((), (0.0,) * 20, (0.0,) * 20 + (0.75,)):
            for uncertainty in invalid:
                for risk in (0.0, 1.0):
                    with self.subTest(prefix=prefix, risk=risk, uncertainty=uncertainty):
                        self.assert_rejected_step_is_atomic(
                            prefix, risk=risk, uncertainty=uncertainty,
                        )

    def test_finish_flushes_every_pending_token_in_normal_and_hold_modes(self):
        for risk in (0.0, 0.4):
            for length in (0, 1, 16, 17, 63, 64, 65, 80):
                with self.subTest(risk=risk, length=length):
                    controller = StreamController(make_policy())
                    for _ in range(length):
                        controller.step(risk)
                    decision = controller.finish()
                    self.assertFalse(decision.alarm)
                    self.assert_counts(decision, length, length, 0)

    def test_all_calls_after_alarm_or_finish_raise_runtime_error(self):
        for terminal in ("hard", "soft", "finish"):
            with self.subTest(terminal=terminal):
                controller = StreamController(make_policy())
                if terminal == "hard":
                    decision = controller.step(1.0)
                elif terminal == "soft":
                    for _ in range(5):
                        decision = controller.step(0.875)
                else:
                    controller.step(0.0)
                    decision = controller.finish()
                snapshot = decision_snapshot(decision)
                for risk in (0.0, 1.0, math.nan):
                    with self.assertRaises(RuntimeError):
                        controller.step(risk)
                with self.assertRaises(RuntimeError):
                    controller.step(0.0, uncertainty=math.nan)
                for _ in range(2):
                    with self.assertRaises(RuntimeError):
                        controller.finish()
                self.assertEqual(decision_snapshot(decision), snapshot)

    def test_step_alarm_and_finish_decisions_are_immutable_snapshots(self):
        controller = StreamController(make_policy())
        first = controller.step(0.0)
        initial_snapshot = decision_snapshot(first)
        for _ in range(80):
            controller.step(0.4)
        final = controller.finish()
        alarm = StreamController(make_policy()).step(1.0)
        self.assertEqual(decision_snapshot(first), initial_snapshot)
        for decision in (first, final, alarm):
            for field in DECISION_FIELDS:
                with self.subTest(alarm=decision.alarm, field=field):
                    with self.assertRaises((AttributeError, TypeError)):
                        setattr(decision, field, getattr(decision, field))
                    with self.assertRaises((AttributeError, TypeError)):
                        delattr(decision, field)

    def test_deterministic_causal_prefix_invariance_under_divergent_suffixes(self):
        rng = random.Random(20261023)
        events = [
            (rng.choice((0.0, 0.1, 0.3, 0.35, 0.4, 0.5, 0.625, 0.75)),
             rng.choice((None, 0.0, 0.69, 0.7, 1.0)))
            for _ in range(96)
        ]
        suffixes = (
            [],
            [(0.0, None)] * 80,
            [(1.0, None)],
            [(0.875, None)] * 8,
            [(0.0, 1.0)] * 80,
            [(rng.random(), rng.choice((None, 0.7))) for _ in range(80)],
        )
        policy = make_policy()
        for length in (0, 1, 15, 16, 17, 63, 64, 65, 96):
            prefix = events[:length]
            expected = self.trace(policy, prefix)
            self.assertEqual(len(expected), length)
            self.assertTrue(all(not decision[0] for decision in expected))
            for suffix_index, suffix in enumerate(suffixes):
                with self.subTest(prefix_length=length, suffix=suffix_index):
                    actual = self.trace(policy, prefix + suffix)
                    self.assertEqual(actual[:length], expected)

    def test_enhancement_never_delays_a_baseline_hard_stop(self):
        hard = 0.9725
        rng = random.Random(521)
        streams = [[hard], [0.0] * 70 + [hard], [0.95] * 12 + [hard]]
        for _ in range(40):
            prefix = [
                rng.uniform(0.0, math.nextafter(hard, 0.0))
                for _ in range(rng.randrange(1, 100))
            ]
            streams.append(prefix + [rng.choice((hard, 1.0))])
        profiles = (
            {"soft_threshold": None},
            {"soft_threshold": 0.0},
            {"soft_threshold": 0.5},
            {"soft_threshold": 0.9},
            {"soft_threshold": 0.75, "evidence_budget": 0.25, "decay": 0.0},
            {"soft_threshold": 0.75, "base_buffer": 32, "hold_buffer": 96},
        )
        for profile in profiles:
            for stream_index, scores in enumerate(streams):
                with self.subTest(profile=profile, stream=stream_index):
                    baseline = StreamController(make_policy(
                        hard_threshold=hard, soft_threshold=None,
                    ))
                    enhanced = StreamController(make_policy(
                        hard_threshold=hard, **profile,
                    ))
                    enhanced_stop = None
                    for position, risk in enumerate(scores, 1):
                        baseline_decision = baseline.step(risk)
                        if enhanced_stop is None:
                            uncertainty = (None, 0.0, 0.7, 1.0)[position % 4]
                            decision = enhanced.step(risk, uncertainty)
                            self.assertEqual(decision.generated, position)
                            self.assertGreaterEqual(decision.buffer_limit, 16)
                            self.assertLessEqual(
                                decision.released, baseline_decision.released,
                            )
                            if decision.alarm:
                                enhanced_stop = position
                        if baseline_decision.alarm:
                            self.assertIsNotNone(enhanced_stop)
                            self.assertLessEqual(enhanced_stop, position)
                            break
                    self.assertTrue(baseline_decision.alarm)


if __name__ == "__main__":
    unittest.main()
