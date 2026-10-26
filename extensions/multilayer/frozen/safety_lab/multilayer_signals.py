"""Fit-only supervised risk directions at three fixed decoder depths."""
from collections import Counter
import math

import numpy as np


def selected_layers(blocks):
    if type(blocks) is not int or blocks < 4:
        raise ValueError('at least four decoder blocks required')
    return (math.ceil(blocks / 2), math.ceil(3 * blocks / 4), blocks)


def unit_vectors(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim < 2 or x.shape[-2] != 3 or x.shape[-1] < 1 or not np.isfinite(x).all():
        raise ValueError('finite [..., 3, hidden_dimension] vectors required')
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def project(values, model):
    x = unit_vectors(values)
    weight = np.asarray(model['weight'], dtype=np.float64)
    bias = np.asarray(model['bias'], dtype=np.float64)
    if (weight.shape != x.shape[-2:] or bias.shape != (3,) or
            not np.isfinite(weight).all() or not np.isfinite(bias).all()):
        raise ValueError('finite fitted model with matching dimensions required')
    result = np.einsum('...ld,ld->...l', x, weight) + bias
    if not np.isfinite(result).all():
        raise ValueError('nonfinite projected scores')
    return result


def fit_directions(records, values, min_groups=32):
    if type(min_groups) is not int or min_groups < 1:
        raise ValueError('positive minimum group count required')
    if not records:
        raise ValueError('fit checkpoint records required')
    seen, identities, groups = set(), {}, {0: set(), 1: set()}
    for row in records:
        if (row.get('role') != 'fit' or row.get('label_source') != 'response_prefix_judge' or
                type(row.get('label')) is not int or row['label'] not in (0, 1)):
            raise ValueError('only valid fit response-prefix labels allowed')
        rid, group, position = row.get('id'), row.get('group'), row.get('position')
        if (not isinstance(rid, str) or not rid or not isinstance(group, str) or not group or
                type(position) is not int or position < 1):
            raise ValueError('fit identity, group and checkpoint position required')
        if (rid, position) in seen:
            raise ValueError('duplicate fit checkpoint')
        seen.add((rid, position))
        if rid in identities and identities[rid] != group:
            raise ValueError('fit request group changed')
        identities[rid] = group
        groups[row['label']].add(group)
    if any(len(groups[label]) < min_groups for label in (0, 1)):
        raise ValueError('insufficient independent fit groups for both response classes')
    x = unit_vectors(values)
    if x.ndim != 3 or x.shape[0] != len(records):
        raise ValueError('one three-layer vector per fit checkpoint required')
    labels = np.asarray([row['label'] for row in records])
    counts = Counter(row['id'] for row in records)
    weights = np.asarray([1. / counts[row['id']] for row in records])
    for label in (0, 1):
        weights[labels == label] *= .5 / weights[labels == label].sum()
    means = [np.einsum('n,nld->ld', weights[labels == label] * 2, x[labels == label])
             for label in (0, 1)]
    residual = x - np.asarray(means)[labels]
    diagonal = np.einsum('n,nld->ld', weights, residual * residual)
    covariance = .9 * diagonal + .1 * diagonal.mean(-1, keepdims=True) + 1e-6
    direction = (means[1] - means[0]) / covariance
    safe = labels == 0
    safe_projection = np.einsum('nld,ld->nl', x[safe], direction)
    safe_weights = weights[safe] * 2
    offset = np.einsum('n,nl->l', safe_weights, safe_projection)
    scale = np.sqrt(np.maximum(np.einsum(
        'n,nl->l', safe_weights, (safe_projection - offset) ** 2), 0)).clip(min=1e-6)
    model = dict(weight=direction / scale[:, None], bias=-offset / scale,
                 fit_groups={str(k): len(v) for k, v in groups.items()},
                 fit_requests=len(identities), fit_checkpoints=len(records),
                 fitting_role='fit', shrinkage=.1, epsilon=1e-6)
    scores = project(x, model)
    model['hold_threshold'] = float(np.quantile(scores[safe].std(-1), .95))
    return model
