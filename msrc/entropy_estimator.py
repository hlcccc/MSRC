"""A development-trained surrogate of string-cluster resampling entropy.

This is not an entailment-based semantic entropy implementation. The teacher is
the existing K=3 resampling statistic; the student reads one frozen prompt state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = 'msrc-resample-entropy-estimator-v1'
FEATURE_RULE = 'InternVL final decoder block output, last prompt position, before final norm'


def feature_contract(provider):
    """Bind a student to model bytes and preprocessing, not installation paths."""
    from pipeline.common import sha
    path = Path(provider.model_path).resolve(strict=True)
    weights = sorted(path.glob('*.safetensors'))
    tokenizers = sorted({p for pattern in ('tokenizer*', '*vocab*', 'merges.txt',
        'special_tokens_map.json', 'added_tokens.json', 'chat_template*') for p in path.glob(pattern)
        if p.is_file()})
    if not weights or not tokenizers:
        raise ValueError('base-model weight and tokenizer identities required')
    # JSON normalization makes the saved and freshly computed contracts comparable.
    value = dict(feature_rule=FEATURE_RULE, dtype='bfloat16',
        model_config_sha256=sha(path/'config.json'),
        weights_sha256={p.name:sha(p) for p in weights},
        tokenizer_sha256={p.name:sha(p) for p in tokenizers},
        image_processor=provider._processor.image_processor.to_dict(),
        transformers_version=__import__('transformers').__version__,
        max_new_tokens=provider.max_new_tokens,
        prompt_function='msrc.batch_collection.primary_prompt', k_teacher=3)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def check_records(records, *, development=False):
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('nonempty records with unique IDs required')
    groups = {}
    for row in records:
        split, group = row.get('acceptance_split'), row.get('image_group')
        if split not in {'dev', 'test'} or not group:
            raise ValueError('explicit development/test and image identity required')
        if development and split != 'dev':
            raise ValueError('test data must never enter estimator fitting')
        if group in groups and groups[group] != split:
            raise ValueError('an image group crosses development/test')
        groups[group] = split


def development_folds(records, folds=4, seed=20260920):
    check_records(records, development=True)
    groups = sorted({r['image_group'] for r in records})
    if len(groups) < folds or folds < 2:
        raise ValueError('not enough independent development groups for cross-fitting')
    order = np.random.RandomState(seed).permutation(len(groups))
    assignment = {groups[int(i)]: j % folds for j, i in enumerate(order)}
    return np.asarray([assignment[r['image_group']] for r in records])


class ResampleEntropyEstimator:
    """Fixed random projection and ridge regression; all normalization is dev-only."""
    def __init__(self, dimension=64, alpha=10., seed=20260920):
        if dimension < 1 or alpha <= 0:
            raise ValueError('positive projection dimension and ridge alpha required')
        self.dimension, self.alpha, self.seed = int(dimension), float(alpha), int(seed)
        self.metadata = None

    def fit(self, features, targets, records):
        check_records(records, development=True)
        x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
        if (x.ndim != 2 or len(x) != len(records) or y.shape != (len(x),)
                or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all()
                or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any()):
            raise ValueError('finite aligned features and entropy targets in [0,1] required')
        self.mean = x.mean(0)
        self.projection = np.random.RandomState(self.seed).normal(
            size=(x.shape[1], min(self.dimension, x.shape[1]))) / np.sqrt(x.shape[1])
        z = (x - self.mean) @ self.projection
        self.scale = z.std(0)
        self.scale[self.scale < 1e-8] = 1.
        z /= self.scale
        self.intercept = float(y.mean())
        self.coef = np.linalg.solve(z.T @ z + self.alpha * np.eye(z.shape[1]),
                                    z.T @ (y - self.intercept))
        self.metadata = dict(format=FORMAT, feature_rule=FEATURE_RULE, target='K=3 normalized string-cluster entropy',
            hidden_dimension=x.shape[1], projection_dimension=z.shape[1], alpha=self.alpha, seed=self.seed,
            fit_scope='development only', train_ids=[r['id'] for r in records],
            train_groups=sorted({r['image_group'] for r in records}), constant_baseline=self.intercept,
            base_model_trained=False, test_used_for_selection=False)
        return self

    def predict(self, features):
        if self.metadata is None:
            raise ValueError('estimator is not fitted')
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('finite features with the frozen hidden dimension required')
        return np.clip(((x - self.mean) @ self.projection / self.scale) @ self.coef + self.intercept, 0., 1.)

    def save(self, path, contract):
        path = Path(path)
        if self.metadata is None or path.suffix != '.npz' or not contract:
            raise ValueError('fitted estimator, .npz destination and feature contract required')
        if path.exists() or path.with_suffix('.json').exists():
            raise ValueError('preserve the existing estimator; use a new output path')
        np.savez_compressed(path, mean=self.mean, projection=self.projection, scale=self.scale,
                            coef=self.coef, intercept=np.asarray(self.intercept))
        meta = dict(self.metadata, contract=contract,
                    weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path.with_suffix('.json').write_text(json.dumps(meta, indent=2, allow_nan=False), encoding='utf-8')
        self.metadata = meta

    def assert_contract(self, contract):
        if self.metadata is None or self.metadata.get('contract') != contract:
            raise ValueError('base-model/feature contract mismatch; retrain for the new model')

    @classmethod
    def load(cls, path, *, contract=None):
        path = Path(path)
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if (meta['format'] != FORMAT or meta['feature_rule'] != FEATURE_RULE
                or hashlib.sha256(path.read_bytes()).hexdigest() != meta['weights_sha256']
                or (contract is not None and meta['contract'] != contract)):
            raise ValueError('estimator format, weights, or base-model/feature contract mismatch')
        result = cls(meta['projection_dimension'], meta['alpha'], meta['seed'])
        with np.load(path, allow_pickle=False) as data:
            for name in ('mean', 'projection', 'scale', 'coef'):
                setattr(result, name, data[name].copy())
            result.intercept = float(data['intercept'])
        result.metadata = meta
        hidden, dimension = meta['hidden_dimension'], meta['projection_dimension']
        if any(getattr(result, name).shape != shape for name, shape in (
                ('mean',(hidden,)), ('projection',(hidden,dimension)),
                ('scale',(dimension,)), ('coef',(dimension,)))):
            raise ValueError('invalid estimator artifact dimensions')
        if (not np.isfinite(result.intercept) or not all(np.isfinite(getattr(result, n)).all()
                for n in ('mean','projection','scale','coef'))):
            raise ValueError('nonfinite estimator artifact')
        if (result.scale <= 0).any():
            raise ValueError('invalid estimator scaling')
        return result


class PromptStateCapture:
    """Retain only the first forward's last prompt state, not all layer states."""
    def __init__(self, provider):
        self.provider, self.value, self.handle = provider, None, None

    def __enter__(self):
        model = self.provider._model
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('prompt state extraction requires a frozen eval-only model')
        layers = model.model.language_model.layers
        def hook(module, inputs, output):
            if self.value is None:
                hidden = output[0] if isinstance(output, tuple) else output
                self.value = hidden[:, -1, :].detach().float().cpu().numpy().copy()
        self.handle = layers[-1].register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        self.handle.remove()

    def features(self):
        if self.value is None:
            raise ValueError('no decoder prefill was captured')
        return self.value


def generate_with_entropy(provider, images, prompts, estimator, *, contract, budget=None):
    """Generate fresh primary answers and estimate entropy without resampling."""
    from contextlib import nullcontext
    from msrc.providers.batched import generate_batch
    estimator.assert_contract(contract)
    if provider.max_new_tokens != contract['max_new_tokens'] or str(provider._model.dtype) != 'torch.bfloat16':
        raise ValueError('generation settings or precision differ from the frozen contract')
    with budget.work('primary_with_entropy_estimate') if budget else nullcontext():
        with PromptStateCapture(provider) as capture:
            samples = generate_batch(provider, images, prompts)
        scores = estimator.predict(capture.features())
    for sample, score in zip(samples, scores):
        sample.meta.update(resample_entropy_estimate=float(score),
                           entropy_is_estimated=True, entropy_extra_generations=0)
    return samples, scores
