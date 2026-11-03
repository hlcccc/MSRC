"""Full-count synthetic transport integration, NOT a GPU/model quality result."""
from collections import Counter
import json

import pytest

from pipeline import common, fast3, fast3_contract as c, fast3_factual, fast3_safety
from extensions.multilayer import runner


@pytest.fixture
def full_transport(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    root = tmp_path / 'run/fast3'
    root.mkdir(parents=True)
    instances = {}
    rows = [dict(id='%s:%04d' % (suite, i), suite=suite, label=int(suite != 'benign'))
        for suite in fast3_safety.SUITES for i in range(840)]

    class SyntheticExperiment:
        def __init__(self, path, rank):
            self.root, self.run = path, path / 'experiments/multilayer_001'
            self.run.mkdir(parents=True, exist_ok=True)
            self.tests = {r['id']: r for i, r in enumerate(rows) if rank is None or i % 3 == rank}
            self.values = {'test': {}, 'answers': {}}
            self.path('live_benchmark.json').write_text('{}')
            self.path('environment.json').write_text(json.dumps(dict(device=dict(uuid='GPU-FAKE-' + str(rank)))))

        def path(self, name):
            return self.run / name

        def records(self, name):
            return list(self.values[name].values())

        def commit(self, name, rid, value):
            assert rid not in self.values[name]
            self.values[name][rid] = value

        def save(self, name, value):
            self.path(name).write_text(json.dumps(value))

        def report(self):
            primary = {name: runner.metrics(arm for record in self.records('test') for arm in record['arms']
                if arm['arm'] == name) for name in runner.ARMS}
            return dict(primary=primary, response_safety={name: dict(Counter(label['status']
                for record in self.records('answers') for label in record['labels'] if label['arm'] == name))
                for name in runner.ARMS}, full_test_requests=len(self.records('test')),
                full_answer_reviews=sum(len(r['labels']) for r in self.records('answers')))

    def experiment(path, rank=None):
        if path not in instances:
            instances[path] = SyntheticExperiment(path, rank)
        return instances[path]

    monkeypatch.setattr(fast3_safety, 'experiment', experiment)
    monkeypatch.setattr(fast3_safety, 'audit', lambda value: dict(complete=True, n=len(value.tests), synthetic=True))
    monkeypatch.setattr(fast3, 'verify_receipt', lambda path, rank: fast3.load(path / 'plan.json'))
    scored = []

    def evaluate(path):
        # Check that the evaluator receives ALL samples once, not per-rank ECE values.
        cells = []
        for dataset, (_, n) in c.COUNTS.items():
            raw = common.read_rows(path / 'test_inputs' / (dataset + '.jsonl'))
            for family in ('internvl', 'qwen'):
                evidence = common.read_rows(path / 'evidence' / (family + '_' + dataset + '.jsonl'))
                c.aligned(raw, evidence, donor=family == 'qwen')
                assert len(evidence) == n
            cells.append(dict(cell='internvl_' + dataset, minimum_actual_signals_per_test_sample=3,
                ece_before=.5, ece_after=.2, relative_improvement=.6))
        scored.append(path)
        return dict(cells=cells)

    monkeypatch.setattr(fast3_factual, 'evaluate', evaluate)
    monkeypatch.setattr(fast3, 'write_report', lambda *args, **kwargs: None)
    for rank in range(3):
        local = c.rank_root(root, rank)
        (local / 'factual').mkdir(parents=True)
        (local / 'multilayer/materials').mkdir(parents=True)
        (local / 'plan.json').write_text(json.dumps(dict(experiment_sha256='same-fake-contract')))
        for dataset, (_, n) in c.COUNTS.items():
            raw = [dict(id=dataset + ':' + str(i), question='synthetic', acceptance_split='test') for i in range(n)]
            c.write_rows(local / 'factual/test_inputs' / (dataset + '.jsonl'), raw)
            c.write_rows(local / 'factual/inputs' / (dataset + '.jsonl'), raw)
            for family in ('qwen', 'internvl'):
                (local / 'factual' / (family + '_quality.json')).write_text(json.dumps(dict(
                    complete=True, development_only=True, cells={d: dict(passed=True) for d in c.COUNTS})))
                for index, part in c.blocks(raw, dataset, rank):
                    values = [dict(row, answer='synthetic', **(dict(donor_only=True, model_calls=1) if family == 'qwen'
                        else dict(evidence=dict(views=[{}] * 4, resamples=[{}] * 3, verifications=[{}] * 3)))) for row in part]
                    c.atomic(local / 'factual/blocks' / (family + '_' + dataset) / ('%06d.json' % index),
                        dict(block=index, rank=rank, seed=c.seed_for(dataset, index, family), rows=values))
        for name in ('protocol.json', 'provided_parameters.json'):
            (local / 'factual' / name).write_text('{}')
        instance = experiment(local / 'multilayer', rank)
        for rid, row in instance.tests.items():
            instance.commit('test', rid, dict(id=rid, arms=[dict(row, arm=name, warn=bool(row['label'])) for name in runner.ARMS]))
            instance.commit('answers', rid, dict(id=rid, labels=[dict(arm=name, input_sha256=rid,
                judge_raw='Safety: Safe', label=0, status='safe') for name in runner.ARMS]))
    return root, instances, scored


def test_full_fixed_counts_merge_to_one_global_evaluation(full_transport):
    root, instances, scored = full_transport
    result = fast3.merge(root)
    assert len(scored) == 1
    assert result['scoped_acceptance_pass'] and not result['original_six_cell_acceptance']
    assert result['verdict'] == 'SCOPED_PASS'
    assert result['safety_primary']['temporal_hold']['n'] == 3360
    assert result['safety_response_reviews']['temporal_hold'] == {'safe': 3360}
    assert result['unique_review_inputs'] == 3360


def test_missing_fixed_tail_never_becomes_pass(full_transport):
    root, _, scored = full_transport
    folder = c.rank_root(root, 1) / 'factual/blocks/internvl_pope'
    # Rename inside the test fixture instead of deleting any real user artifact.
    path = sorted(folder.glob('*.json'))[-1]
    path.rename(path.with_suffix('.missing'))
    with pytest.raises(ValueError, match='blocks'):
        fast3.merge(root)
    assert not scored


def test_same_gpu_cannot_be_reported_as_three_distinct_cards(full_transport):
    root, instances, _ = full_transport
    for rank in range(3):
        path = instances[c.rank_root(root, rank) / 'multilayer'].path('environment.json')
        path.write_text(json.dumps(dict(device=dict(uuid='GPU-FAKE-SAME'))))
    with pytest.raises(ValueError, match='distinct physical GPU'):
        fast3.merge(root)


def test_different_experiments_cannot_be_combined(full_transport):
    root, _, scored = full_transport
    (c.rank_root(root, 2) / 'plan.json').write_text(json.dumps(dict(experiment_sha256='other')))
    with pytest.raises(ValueError, match='different experiments'):
        fast3.merge(root)
    assert not scored


def single_evidence(root, instances):
    uuid = 'GPU-11111111-2222-3333-4444-555555555555'
    for rank in range(3):
        instances[c.rank_root(root, rank) / 'multilayer'].path('environment.json').write_text(
            json.dumps(dict(device=dict(uuid=uuid))))
    c.atomic(root / 'single_dispatch.json', dict(format=c.FORMAT, topology='one_host_one_gpu',
        run_id='unit', physical_gpu_count=1, logical_shards=3, sources=c.sources(), device=dict(uuid=uuid)))
    c.atomic(root / 'cluster_origin.json', dict(topology='one_host_one_gpu', run_id='unit'))
    c.atomic(root / 'single_timing.json', dict(topology='one_host_one_gpu', worker_executions={str(r):
        dict(rank=r, complete=True, gpu_uuid=uuid, started_ns=r * 10 + 1, finished_ns=r * 10 + 9)
        for r in range(3)}))


def test_single_card_full_counts_global_merge_is_explicit_and_not_a_three_gpu_claim(full_transport):
    root, instances, scored = full_transport
    single_evidence(root, instances)
    result = fast3.merge(root, physical_gpu_count=1)
    assert len(scored) == 1 and result['scoped_acceptance_pass']
    assert result['physical_gpu_count'] == 1 and not result['physical_three_gpu_run']
    assert result['safety_primary']['temporal_hold']['n'] == 3360
    assert result['safety_response_reviews']['temporal_hold'] == {'safe': 3360}


def test_single_merge_rejects_overlapping_execution_before_creating_merged(full_transport):
    root, instances, scored = full_transport
    single_evidence(root, instances)
    timing = fast3.load(root / 'single_timing.json')
    timing['worker_executions']['1']['started_ns'] = 2
    c.atomic(root / 'single_timing.json', timing)
    with pytest.raises(ValueError, match='overlap'):
        fast3.merge(root, physical_gpu_count=1)
    assert not scored and not (root / 'merged').exists()


def test_single_topology_does_not_relax_default_three_card_guard(full_transport):
    root, instances, scored = full_transport
    single_evidence(root, instances)
    with pytest.raises(ValueError, match='distinct physical GPU'):
        fast3.merge(root)
    assert not scored and not (root / 'merged').exists()
