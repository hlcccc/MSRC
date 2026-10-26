"""Causal risk accumulation and bounded output holding around a frozen detector."""
from dataclasses import dataclass
import math


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return float(value)


@dataclass(frozen=True)
class Policy:
    hard_threshold: float
    soft_threshold: float | None
    evidence_budget: float = 2.5
    decay: float = 0.8
    base_buffer: int = 16
    hold_buffer: int = 64

    def __post_init__(self):
        hard = number(self.hard_threshold, 'hard_threshold')
        if not 0 < hard <= 1:
            raise ValueError('hard_threshold must be in (0, 1]')
        if self.soft_threshold is not None:
            soft = number(self.soft_threshold, 'soft_threshold')
            if not 0 <= soft < hard:
                raise ValueError('soft_threshold must be below hard_threshold')
        if number(self.evidence_budget, 'evidence_budget') <= 0:
            raise ValueError('evidence_budget must be positive')
        if not 0 <= number(self.decay, 'decay') < 1:
            raise ValueError('decay must be in [0, 1)')
        if any(type(v) is not int for v in (self.base_buffer, self.hold_buffer)):
            raise ValueError('buffer limits must be integers')
        if self.base_buffer < 16 or self.hold_buffer < self.base_buffer:
            raise ValueError('cannot weaken baseline output buffering')


@dataclass(frozen=True)
class Decision:
    alarm: bool
    reason: str
    state: str
    generated: int
    released: int
    pending: int
    evidence: float
    buffer_limit: int


class StreamController:
    def __init__(self, policy):
        if not isinstance(policy, Policy):
            raise TypeError('expected immutable Policy')
        self.policy = policy
        self.state = 'NORMAL'
        self.generated = self.released = self.pending = 0
        self.evidence = 0.0

    def _decision(self, alarm=False, reason='continue'):
        limit = self.policy.hold_buffer if self.state == 'HOLD' else self.policy.base_buffer
        return Decision(alarm, reason, self.state, self.generated, self.released,
                        self.pending, self.evidence, limit)

    def _active(self):
        if self.state in ('BLOCKED', 'FINISHED'):
            raise RuntimeError('controller is terminal')

    def step(self, risk, uncertainty=None):
        self._active()
        risk = number(risk, 'risk')
        if not 0 <= risk <= 1:
            raise ValueError('risk must be in [0, 1]')
        if uncertainty is not None:
            uncertainty = number(uncertainty, 'uncertainty')
            if not 0 <= uncertainty <= 1:
                raise ValueError('uncertainty must be in [0, 1]')
        # Validate the complete input before mutating any state.
        self.generated += 1
        self.pending += 1
        if risk >= self.policy.hard_threshold:
            self.state, self.pending = 'BLOCKED', 0
            return self._decision(True, 'baseline_hard_alarm')
        soft = self.policy.soft_threshold
        increment = 0.0 if soft is None else min(
            1.0, max(0.0, (risk - soft) / (self.policy.hard_threshold - soft)))
        self.evidence = self.policy.decay * self.evidence + increment
        if soft is not None and self.evidence >= self.policy.evidence_budget:
            self.state, self.pending = 'BLOCKED', 0
            return self._decision(True, 'accumulated_soft_evidence')
        uncertain = uncertainty is not None and uncertainty >= 0.7
        if uncertain or (soft is not None and risk >= 0.8 * soft):
            self.state = 'HOLD'
        elif (soft is None or risk < 0.6 * soft) and self.evidence < 0.25 * self.policy.evidence_budget:
            self.state = 'NORMAL'
        limit = self.policy.hold_buffer if self.state == 'HOLD' else self.policy.base_buffer
        count = max(0, self.pending - limit)
        self.pending -= count
        self.released += count
        return self._decision()

    def finish(self):
        self._active()
        self.released += self.pending
        self.pending = 0
        self.state = 'FINISHED'
        return self._decision(False, 'finished_without_alarm')
