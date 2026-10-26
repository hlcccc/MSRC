"""Causal multi-layer evidence and bounded delivery around a frozen detector."""
from dataclasses import dataclass
import math

from safety_lab.policy import Decision, Policy, StreamController, number


def layer_values(values):
    if len(values) != 3:
        raise ValueError('exactly three layer scores required')
    return tuple(number(float(x) if hasattr(x, 'item') else x, 'layer score') for x in values)


class Evidence:
    def __init__(self):
        self.mean = self.value = 0.

    def step(self, values):
        values = layer_values(values)
        score = sum(values) / 3
        disagreement = math.sqrt(sum((x - score) ** 2 for x in values) / 3)
        rise = max(0., score - self.mean)
        value = .9 * self.value + max(0., score - .5) + .25 * rise
        mean = .8 * self.mean + .2 * score
        if not all(math.isfinite(x) for x in (value, mean, disagreement)):
            raise ValueError('nonfinite causal evidence')
        self.mean, self.value = mean, value
        return dict(semantic=max(0., score), temporal=value, disagreement=disagreement)


def select_threshold(records, budget=.05, min_groups=20):
    budget = number(budget, 'false-positive budget')
    if not 0 <= budget < 1 or type(min_groups) is not int or min_groups < 1:
        raise ValueError('invalid calibration budget or group minimum')
    seen, safe, unsafe = set(), [], []
    for row in records:
        if row.get('role') != 'calibration':
            raise ValueError('only calibration requests may select a threshold')
        if (not isinstance(row.get('id'), str) or not row['id'] or
                not isinstance(row.get('group'), str) or not row['group'] or row['id'] in seen):
            raise ValueError('missing or duplicate calibration identity')
        seen.add(row['id'])
        label = row.get('label')
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError('invalid calibration response label')
        score = number(row['score'], 'reachable calibration score')
        incremental = row['incremental_score']
        if incremental is not None:
            incremental = number(incremental, 'incremental calibration score')
            if incremental < 0 or incremental > score + 1e-9:
                raise ValueError('incremental score is outside the reachable prefix')
        if score < 0 or type(row.get('baseline_warn')) is not bool:
            raise ValueError('invalid calibration score or baseline alarm')
        if label == 0 and not row['baseline_warn']:
            safe.append(row)
        elif label == 1:
            unsafe.append(row)
    if any(len({r['group'] for r in rows}) < min_groups for rows in (safe, unsafe)):
        raise ValueError('insufficient independent calibration groups')
    maximum_false = math.floor(budget * len(safe))
    candidate = sorted(r['score'] for r in safe)[len(safe) - maximum_false - 1]
    false = sum(r['score'] > candidate for r in safe)
    incremental = sum(r['incremental_score'] is not None and r['incremental_score'] > candidate
                      for r in unsafe)
    enabled = incremental > 0
    return dict(enabled=enabled, threshold=candidate if enabled else None,
                candidate_threshold=candidate, safe_requests=len(safe),
                unsafe_requests=len(unsafe), safe_false_alarms=false,
                maximum_safe_false_alarms=maximum_false,
                added_unsafe_calibration_warnings=incremental,
                false_positive_budget=budget, comparison='strict_greater_than',
                reason='enabled' if enabled else 'no_incremental_calibration_warning',
                excluded_requests=len(records) - len(safe) - len(unsafe))


@dataclass(frozen=True)
class LayerDecision(Decision):
    semantic_score: float
    temporal_score: float
    disagreement: float


class MultiLayerController:
    def __init__(self, policy, threshold, mode, hold_threshold=None):
        if mode not in ('semantic', 'temporal'):
            raise ValueError('unknown multi-layer scoring mode')
        for value in (threshold, hold_threshold):
            if value is not None and number(value, 'threshold') < 0:
                raise ValueError('nonnegative fitted threshold required')
        self.reference = StreamController(policy)
        self.policy, self.threshold, self.mode = policy, threshold, mode
        self.hold_threshold = hold_threshold
        self.evidence = Evidence()
        self.generated = self.released = self.pending = 0
        self.state, self.extra_hold = 'NORMAL', False
        self.scores = dict(semantic=0., temporal=0., disagreement=0.)

    def _active(self):
        if self.state in ('BLOCKED', 'FINISHED'):
            raise RuntimeError('controller is terminal')

    def _decision(self, alarm=False, reason='continue'):
        return LayerDecision(alarm, reason, self.state, self.generated, self.released,
                             self.pending, self.evidence.value,
                             self.policy.hold_buffer if self.state == 'HOLD' else self.policy.base_buffer,
                             self.scores['semantic'], self.scores['temporal'],
                             self.scores['disagreement'])

    def step(self, risk, uncertainty, values):
        self._active()
        risk, uncertainty = number(risk, 'risk'), number(uncertainty, 'uncertainty')
        if not 0 <= risk <= 1 or not 0 <= uncertainty <= 1:
            raise ValueError('risk and uncertainty must be in [0, 1]')
        values = layer_values(values)
        self.scores = self.evidence.step(values)
        base = self.reference.step(risk, uncertainty)
        self.generated += 1
        self.pending += 1
        if base.alarm:
            self.state, self.pending = 'BLOCKED', 0
            return self._decision(True, base.reason)
        if self.threshold is not None and self.scores[self.mode] > self.threshold:
            self.state, self.pending = 'BLOCKED', 0
            return self._decision(True, 'multilayer_' + self.mode + '_alarm')
        if self.hold_threshold is not None:
            if self.scores['disagreement'] > self.hold_threshold:
                self.extra_hold = True
            elif self.scores['disagreement'] <= .5 * self.hold_threshold:
                self.extra_hold = False
        self.state = 'HOLD' if self.extra_hold or base.state == 'HOLD' else 'NORMAL'
        limit = self.policy.hold_buffer if self.state == 'HOLD' else self.policy.base_buffer
        count = min(max(0, self.pending - limit), max(0, base.released - self.released))
        self.pending -= count
        self.released += count
        return self._decision()

    def finish(self):
        self._active()
        self.reference.finish()
        self.released += self.pending
        self.pending = 0
        self.state = 'FINISHED'
        return self._decision(False, 'finished_without_alarm')
