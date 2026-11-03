import hashlib
import json

import pytest


def development_fixture(root):
    from extensions.multilayer.runner import judgment_key

    experiment = root / 'experiment'
    experiment.mkdir()
    roles = dict(fit=[], calibration=[], test=['test_only'])
    inputs = [dict(id='test_only', split='test', suite='SD', question='held out', label=1)]
    frozen_files = {}
    for suite in ('SD', 'SD_TYPO', 'TYPO', 'benign'):
        for role in ('fit', 'calibration'):
            rid = suite + '_' + role
            roles[role].append(rid)
            row = dict(id=rid, split='dev', suite=suite, group=rid, question=rid, label=0)
            inputs.append(row)
            prefixes = [dict(position=p, text=str(p),
                prefix_sha256=hashlib.sha256(str(p).encode()).hexdigest()) for p in (8, 9)]
            record = dict(row, role=role, generated_tokens=9, prefixes=prefixes)
            labels = [dict(id=rid, role=role, group=rid, position=p['position'],
                label=0, status='safe', label_source='response_prefix_judge',
                prefix_sha256=p['prefix_sha256'], input_sha256=judgment_key(rid, p['text']),
                judge_raw='Safety: Safe') for p in prefixes]
            name = hashlib.sha256(rid.encode()).hexdigest() + '.json'
            for stage, value in [('development', record), ('labels', dict(id=rid, labels=labels))]:
                path = experiment / 'records' / stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding='utf-8')
                frozen_files[path.relative_to(experiment).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, value in [('protocol.json', dict(roles=roles)), ('calibration.json', {}), ('directions.json', {})]:
        path = experiment / name
        path.write_text(json.dumps(value), encoding='utf-8')
        frozen_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (experiment / 'frozen.json').write_text(json.dumps(dict(files=frozen_files, test_used_for_selection=False)))
    input_path = root / 'inputs.jsonl'
    input_path.write_text('\n'.join(json.dumps(row) for row in inputs), encoding='utf-8')
    return experiment, input_path


def test_selects_real_frozen_prefixes_without_test_or_retokenizing(tmp_path):
    from pipeline.safety_judge_validation import select_development, build_cohort
    experiment, inputs = development_fixture(tmp_path)
    selected, records, labels, all_inputs, hashes = select_development(experiment, inputs, per_suite=2)
    assert len(selected) == 8 and len(records) == 8 and len(labels) == 16
    assert 'test_only' not in {row['id'] for row in selected}
    assert all(p['position'] in (8, 9) for record in records for p in record['prefixes'])

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize):
            assert tokenize is False
            return json.dumps(messages)

    legacy = dict(prompts=['diagnostic'], origins=[dict(id='SD_fit', role='fit')])
    cohort = build_cohort(Tokenizer(), selected, records, legacy)
    assert cohort['actual_development_prefixes'] == 16 and cohort['unique_prompts'] == 17
    assert all('test_only' not in prompt for prompt in cohort['prompts'])
    assert not cohort['new_vlm_generations'] and cohort['fresh_judge_generation']


def test_changed_frozen_development_text_rejected(tmp_path):
    from pipeline.safety_judge_validation import select_development
    experiment, inputs = development_fixture(tmp_path)
    path = next((experiment / 'records/development').glob('*.json'))
    value = json.loads(path.read_text())
    value['prefixes'][0]['text'] = 'changed'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen observed'):
        select_development(experiment, inputs, per_suite=2)


@pytest.mark.parametrize('legacy', [dict(prompts=['one'], origins=[]),
    dict(prompts=['one'], origins=[dict(role='test')])])
def test_invalid_legacy_diagnostic_not_silently_dropped(legacy):
    from pipeline.safety_judge_validation import build_cohort
    with pytest.raises(ValueError, match='development-only'):
        build_cohort(None, [], [], legacy)


def test_validation_reports_unknown_as_unknown_not_safe():
    from pipeline.safety_judge_validation import comparison_details
    result = comparison_details(['Safety: Safe'], ['Safety: Controversial'], [[dict(role='calibration')]])
    assert result['unknown'] == 1 and result['parsed_matches'] == 0
    assert result['differences'][0]['reference_parsed']['label'] == 0
    assert result['differences'][0]['actual_parsed']['label'] is None


def test_first_divergent_token_is_not_a_padding_length_difference():
    from pipeline.judge_batch_diagnostic import divergence
    assert divergence([4, 5, 6], [4, 5, 7]) == 2
    assert divergence([4, 5, 6], [4, 5, 6]) is None
    assert divergence([4, 5, 6], [4, 5, 6, 0]) is None
