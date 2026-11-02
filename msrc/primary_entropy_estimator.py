"""Cheap primary-summary control for the separate prompt-hidden-state student."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from msrc.entropy_estimator import ResampleEntropyEstimator

FORMAT = 'msrc-primary-statistics-entropy-estimator-v1'
FEATURE_RULE = 'primary answer summary statistics only; no decoder hidden states'
FEATURES = ('sequence_confidence', 'sequence_entropy', 'first_token_margin',
            'log_generated_tokens', 'log_prompt_tokens', 'log_answer_characters', 'textvqa')


def primary_features(sample, dataset):
    """Accept a primary Sample, never labels, gold answers or auxiliary answers."""
    if dataset not in {'pope', 'textvqa'}:
        raise ValueError('explicit supported dataset required')
    numbers = [sample.sequence_confidence, sample.sequence_entropy, sample.first_token_margin]
    lengths = [sample.meta.get('generated'), sample.meta.get('prompt_len'), len(sample.text)]
    if (any(value is None or not np.isfinite(value) for value in numbers+lengths)
            or any(value < 0 for value in lengths) or numbers[1] < 0
            or not 0 <= numbers[0] <= 1 or not 0 <= numbers[2] <= 1):
        raise ValueError('finite primary scores and token counts required')
    return np.asarray(numbers + np.log1p(lengths).tolist() + [float(dataset=='textvqa')])


class PrimaryStatisticsEntropyEstimator(ResampleEntropyEstimator):
    def fit(self, features, targets, records):
        if np.asarray(features).ndim != 2 or np.asarray(features).shape[1] != len(FEATURES):
            raise ValueError('the fixed primary-statistics feature order is required')
        super().fit(features, targets, records)
        self.metadata.update(format=FORMAT, feature_rule=FEATURE_RULE, features=list(FEATURES),
                             input_dimension=len(FEATURES), hidden_states_used=False)
        self.metadata.pop('hidden_dimension')
        return self

    @classmethod
    def load(cls, path, *, contract=None):
        path = Path(path)
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if (meta.get('format') != FORMAT or meta.get('feature_rule') != FEATURE_RULE
                or meta.get('features') != list(FEATURES) or meta.get('hidden_states_used') is not False
                or meta.get('input_dimension') != len(FEATURES)
                or hashlib.sha256(path.read_bytes()).hexdigest() != meta.get('weights_sha256')
                or contract is not None and meta.get('contract') != contract):
            raise ValueError('primary-statistics artifact contract mismatch')
        result = cls(meta['projection_dimension'], meta['alpha'], meta['seed'])
        with np.load(path, allow_pickle=False) as arrays:
            for name in ('mean', 'projection', 'scale', 'coef'):
                setattr(result, name, arrays[name].copy())
            result.intercept = float(arrays['intercept'])
        dimension = meta['projection_dimension']
        for name, shape in [('mean', (len(FEATURES),)), ('projection', (len(FEATURES),dimension)),
                            ('scale', (dimension,)), ('coef', (dimension,))]:
            value = getattr(result, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('invalid primary-statistics artifact dimensions or values')
        if not np.isfinite(result.intercept) or (result.scale <= 0).any():
            raise ValueError('invalid primary-statistics scaling or intercept')
        result.metadata = meta
        return result
