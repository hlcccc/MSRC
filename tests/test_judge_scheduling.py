import pytest


def test_padding_accounting():
    from msrc.judge_scheduling import padded_slots
    assert padded_slots([2, 20, 3, 18], 2) == 76
    assert padded_slots(sorted([2, 20, 3, 18]), 2) == 46
    assert padded_slots([], 4) == 0
    with pytest.raises(ValueError):
        padded_slots([0], 2)


@pytest.mark.parametrize('ordered', [False, True])
@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_encoded_judge_preserves_original_order_and_all_tokens(ordered, batch_size):
    torch = pytest.importorskip('torch')
    from msrc.judge_scheduling import encoded_judge_texts

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2
        padding_side = 'right'

        def __call__(self, prompts, **kwargs):
            assert kwargs == dict(padding=False, truncation=False)
            return dict(input_ids=[[int(prompt)] * int(prompt) for prompt in prompts],
                        attention_mask=[[1] * int(prompt) for prompt in prompts])

        def pad(self, records, **kwargs):
            assert self.padding_side == 'left'
            assert kwargs == dict(padding=True, return_tensors='pt')
            width = max(len(row['input_ids']) for row in records)
            return Inputs({key: torch.tensor([[0] * (width - len(row[key])) + row[key]
                           for row in records]) for key in records[0]})

        def batch_decode(self, values, **kwargs):
            return [str(row[0].item()) for row in values]

    class Model:
        device = 'cpu'

        def generate(self, input_ids, attention_mask, **kwargs):
            assert kwargs == dict(max_new_tokens=128, do_sample=False)
            assert torch.equal((input_ids != 0).long(), attention_mask)
            return torch.cat([input_ids, input_ids[:, -1:]], dim=-1)

    stats = {}
    prompts = ['3', '12', '2', '8', '5']
    assert encoded_judge_texts(Model(), Tokenizer(), prompts, batch_size=batch_size,
        order_by_length=ordered, statistics=stats) == prompts
    assert stats['input_tokens'] == 30
    assert sum(row['size'] for row in stats['batches']) == len(prompts)


def test_scheduled_empty_and_invalid_inputs():
    from msrc.judge_scheduling import encoded_judge_texts
    assert encoded_judge_texts(None, None, [], batch_size=2) == []
    with pytest.raises(ValueError):
        encoded_judge_texts(None, None, [], batch_size=0)
    with pytest.raises(ValueError):
        encoded_judge_texts(None, None, [3], batch_size=1)


def test_judgment_admission_checks_full_raw_not_only_label():
    from pipeline.safety_judge_speed import compare_judgments
    parse = lambda text: dict(label=0 if text.startswith('safe') else 1, status='safe' if text.startswith('safe') else 'unsafe')
    exact = compare_judgments(['safe: a', 'unsafe: b'], ['safe: a', 'unsafe: b'], parse)
    assert exact['exact_raw'] and exact['exact_parsed']
    changed = compare_judgments(['safe: a'], ['safe: different'], parse)
    assert not changed['exact_raw'] and changed['exact_parsed']
    with pytest.raises(ValueError):
        compare_judgments(['safe'], [], parse)


def test_development_cohort_cannot_import_test_answers(tmp_path):
    import json
    from pipeline.safety_judge_speed import development_records
    roles = dict(fit=[], calibration=[], test=['held_out'])
    inputs, observed = [], []
    for suite in ('SD', 'SD_TYPO', 'TYPO', 'benign'):
        for role in ('fit', 'calibration'):
            rid = suite + '_' + role
            roles[role].append(rid)
            inputs.append(dict(id=rid, suite=suite, question='q', split='dev'))
            observed.append(dict(id=rid, answer='observed', split='dev'))
    inputs.append(dict(id='held_out', suite='SD', split='test', question='do not select'))
    observed.append(dict(id='held_out', answer='do not select', split='test'))
    (tmp_path / 'dataset_identity.json').write_text(json.dumps(dict(roles=roles)))
    (tmp_path / 'inputs.jsonl').write_text('\n'.join(map(json.dumps, inputs)))
    (tmp_path / 'development_generation.jsonl').write_text('\n'.join(map(json.dumps, observed)))
    values = development_records(tmp_path, per_suite=2)
    assert len(values) == 8 and all(row['id'] != 'held_out' for row in values)
    observed[0]['split'] = 'test'
    (tmp_path / 'development_generation.jsonl').write_text('\n'.join(map(json.dumps, observed)))
    with pytest.raises(ValueError, match='non-development'):
        development_records(tmp_path, per_suite=2)


def test_overlapping_roles_rejected_before_loading_answers(tmp_path):
    import json
    from pipeline.safety_judge_speed import development_records
    (tmp_path / 'dataset_identity.json').write_text(json.dumps(dict(
        roles=dict(fit=['same'], calibration=[], test=['same']))))
    with pytest.raises(ValueError, match='overlap'):
        development_records(tmp_path)


def test_non_equivalent_timed_baseline_cannot_admit_fast_candidate():
    from pipeline.safety_judge_speed import candidate_admission
    rows = []
    for repeat in (0, 1):
        for variant in ('original', 'encoded', 'length_bucket', 'length_bucket_8'):
            exact = variant != 'original'
            rows.append(dict(variant=variant, repeat=repeat, seconds=10 if variant == 'original' else 5,
                unchanged_rng=True, comparison=dict(exact_raw=exact, exact_parsed=exact)))
    for row in candidate_admission(rows):
        assert row['timing_gate_passed'] and row['candidate_matches_serial']
        assert not row['baseline_matches_serial'] and not row['eligible_for_further_validation']
        assert 'baseline_differs_from_serial_or_changes_rng' in row['blockers']


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'nan', 'rng', 'parsed', 'raw', 'slow'])
def test_judge_candidate_requires_complete_valid_paired_evidence(defect):
    from pipeline.safety_judge_speed import candidate_admission
    rows = [dict(variant=name, repeat=repeat, seconds=10 if name == 'original' else 5,
        unchanged_rng=True, comparison=dict(exact_raw=True, exact_parsed=True))
        for repeat in (0, 1) for name in ('original', 'encoded', 'length_bucket', 'length_bucket_8')]
    candidate = next(row for row in rows if row['variant'] == 'encoded')
    if defect == 'missing':
        rows.remove(candidate)
    elif defect == 'duplicate':
        rows.append(dict(candidate))
    elif defect == 'nan':
        candidate['seconds'] = float('nan')
    elif defect == 'rng':
        candidate['unchanged_rng'] = False
    elif defect == 'parsed':
        candidate['comparison']['exact_parsed'] = False
    elif defect == 'raw':
        candidate['comparison']['exact_raw'] = False
    elif defect == 'slow':
        candidate['seconds'] = 11
    results = {row['variant']: row for row in candidate_admission(rows)}
    assert not results['encoded']['eligible_for_further_validation']
    assert results['length_bucket']['eligible_for_further_validation']


@pytest.mark.parametrize('baseline_exact', [False, True])
def test_saved_trial_audit_recomputes_admission_without_rewriting_evidence(tmp_path, baseline_exact):
    import json
    from extensions.multilayer.runner import parse_response_safety
    from pipeline.audit_safety_judge_speed import audit
    from pipeline.safety_judge_speed import compare_judgments
    serial = ['Safety: Safe\nCategories: None\nRefusal: Yes']
    changed = ['Safety: Controversial\nCategories: Non-violent Illegal Acts\nRefusal: Yes']
    rows, outputs = [], {}
    for repeat in (0, 1):
        for name in ('original', 'encoded', 'length_bucket', 'length_bucket_8'):
            actual = changed if name == 'original' and not baseline_exact else serial
            outputs[name + '_' + str(repeat)] = actual
            rows.append(dict(variant=name, repeat=repeat, seconds=10 if name == 'original' else 5,
                unchanged_rng=True, comparison=compare_judgments(serial, actual, parse_response_safety)))
    documents = {'protocol.json': {}, 'cohort.json': dict(origins=[dict(id='development', role='fit')]),
        'report.json': dict(comparisons=[dict(eligible_for_further_validation=True)]),
        'serial_reference.json': dict(outputs=serial), 'measurements.json': rows,
        'fresh_outputs.json': outputs, 'input_tensor_audit.json': {}}
    for name, value in documents.items():
        (tmp_path / name).write_text(json.dumps(value), encoding='utf-8')
    original_bytes = {name: (tmp_path / name).read_bytes() for name in documents}
    result = audit(tmp_path)
    assert all((tmp_path / name).read_bytes() == value for name, value in original_bytes.items())
    assert all(row['eligible_for_further_validation'] == baseline_exact for row in result['comparisons'])
    assert len(result['differences']) == (0 if baseline_exact else 2)
    assert (tmp_path / 'audit.md').is_file()
    with pytest.raises(ValueError, match='retain earlier audits'):
        audit(tmp_path)
