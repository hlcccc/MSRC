"""Immutable three-worker transport contract; no model inference or fitting."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re

from pipeline import common

FORMAT = 'msrc-frozen-fast3-v1'
WORLD = 3
COUNTS = {'pope': (4500, 4500), 'textvqa': (2504, 2496)}
BATCH = {'qwen': {'pope': 64, 'textvqa': 16}, 'internvl': {'pope': 4, 'textvqa': 1}}
BLOCK = {'pope': 64, 'textvqa': 16}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = common.inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    common.save(temporary, value)
    temporary.replace(path)


def root_for(run_id):
    require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', run_id)), 'invalid run ID')
    return common.inside(common.runs_dir() / run_id / 'fast3')


def rank_root(root, rank):
    require(type(rank) is int and 0 <= rank < WORLD, 'rank must be 0, 1 or 2')
    return common.inside(root / ('rank-' + str(rank)))


@contextmanager
def lock(root, name):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with common.inside(root / name).open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sources():
    folders = ('pipeline', 'extensions', 'msrc', 'evaluation', 'scripts', 'references', 'resources')
    suffixes = {'.py', '.sh', '.json'}
    return {p.relative_to(common.ROOT).as_posix(): common.sha(p)
        for folder in folders for p in sorted((common.ROOT / folder).rglob('*'))
        if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts}


def fixed_tests(rows, dataset):
    dev, test = COUNTS[dataset]
    require(len(rows) == dev + test and len({r['id'] for r in rows}) == len(rows),
        'full original input count/unique IDs required')
    require(sum(r['acceptance_split'] == 'dev' for r in rows) == dev and
        sum(r['acceptance_split'] == 'test' for r in rows) == test, 'original split changed')
    selected = [r for r in rows if r['acceptance_split'] == 'test']
    require(not ({r['acceptance_image_group'] for r in rows if r['acceptance_split'] == 'dev'} &
        {r['acceptance_image_group'] for r in selected}), 'image leakage across fixed split')
    return selected


def blocks(rows, dataset, rank):
    rank_root(common.ROOT, rank)
    size = BLOCK[dataset]
    return [(i // size, rows[i:i + size]) for i in range(0, len(rows), size)
        if (i // size) % WORLD == rank]


def assigned(rows, dataset, rank):
    return [r for _, part in blocks(rows, dataset, rank) for r in part]


def seed_for(dataset, block, stage='internvl'):
    # Rank and process history are deliberately absent from the RNG identity.
    return int(digest([FORMAT, common.SEED, stage, dataset, block])[:15], 16)


def aligned(raw, values, donor=False):
    require(len(raw) == len(values), 'incomplete evidence coverage')
    for a, b in zip(raw, values):
        require(all(b.get(k) == v for k, v in a.items()) and
            isinstance(b.get('answer'), str) and b['answer'].strip(), 'evidence identity/order changed')
        if donor:
            require(b.get('donor_only') is True and b.get('model_calls') == 1, 'invalid Qwen donor')
        else:
            evidence = b.get('evidence', {})
            require(tuple(len(evidence.get(k, [])) for k in ('views', 'resamples', 'verifications')) ==
                (4, 3, 3), 'original evidence counts changed')


def merge_ordered(expected, shards):
    ids = [r['id'] for r in expected]
    require(len(set(ids)) == len(ids), 'duplicate expected ID')
    values = {}
    for shard in shards:
        for row in shard:
            require(row['id'] not in values, 'duplicate result across workers')
            values[row['id']] = row
    require(set(values) == set(ids), 'missing or unexpected result IDs')
    return [values[rid] for rid in ids]


def write_rows(path, rows):
    path = common.inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            common.emit(handle, row)


def training_forbidden(*args, **kwargs):
    raise RuntimeError('fast3 uses supplied state only; onsite fitting/backpropagation forbidden')


def forbid_training():
    from msrc.model import RiskCalibrator
    import torch
    RiskCalibrator.fit = training_forbidden
    torch.set_grad_enabled(False)
    torch.Tensor.backward = training_forbidden
    torch.autograd.backward = training_forbidden
    torch.optim.Optimizer.step = training_forbidden


def verify_plan(root, plan):
    require(plan['format'] == FORMAT and plan['world_size'] == WORLD and
        plan['project_root'] == str(common.ROOT) and plan['sources'] == sources(),
        'wrong plan, installation path or changed source; preserve run and use new ID')
    require(plan['onsite_training'] is False and plan['batches'] == BATCH and plan['blocks'] == BLOCK,
        'frozen execution configuration changed')
    require(plan['plan_sha256'] == digest({k: v for k, v in plan.items() if k != 'plan_sha256'}),
        'plan identity changed')
    for name, value in plan['immutable_files'].items():
        require(common.sha(root / name) == value, 'prepared artifact changed: ' + name)
    return plan
