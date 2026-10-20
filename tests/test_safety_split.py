"""Content identity must constrain both held-out splits and fitting roles."""
import copy
import hashlib
import json

import numpy as np
import pytest

from pipeline import plugguard
from pipeline.plugguard_report import assign_roles
from pipeline.safety_split import (balanced_subset, components, split_rows,
                                   validate_partition)

SEED = 20260920


def fixture(n=8):
    rows, hashes = [], {}
    for label in (0, 1):
        for i in range(n):
            group = '%d:%04d' % (label, i)
            for suite in (['SD', 'SD_TYPO', 'TYPO'] if label else ['benign']):
                rid = group + ':' + suite
                image = '/fixture/' + rid + '.png'
                hashes[image] = hashlib.sha256(rid.encode()).hexdigest()
                rows.append(dict(id=rid, group=group, label=label, suite=suite,
                                 question=rid, image=image))
    return rows, hashes


def alias(rows, hashes, left, right):
    source = next(r for r in rows if r['id'] == left)
    target = next(r for r in rows if r['id'] == right)
    target['question'] = source['question']
    hashes[target['image']] = hashes[source['image']]


def test_no_aliases_preserve_previous_seeded_group_split():
    rows, hashes = fixture()
    result = split_rows(rows, hashes, SEED)
    for label in (0, 1):
        groups = sorted({r['group'] for r in rows if r['label'] == label})
        order = np.random.default_rng(SEED).permutation(len(groups))
        old_dev = {groups[i] for i in order[:len(groups) // 2]}
        assert {r['group'] for r in result if r['label'] == label and r['split'] == 'dev'} == old_dev
    assert all('split' not in r for r in rows)


def test_cross_id_content_aliases_and_transitive_variants_stay_together():
    rows, hashes = fixture()
    alias(rows, hashes, '1:0000:SD', '1:0001:SD')
    alias(rows, hashes, '1:0001:TYPO', '1:0002:TYPO')
    result = split_rows(rows, hashes, SEED)
    linked = [r for r in result if r['group'] in ('1:0000', '1:0001', '1:0002')]
    assert len({r['split'] for r in linked}) == len({r['split_group'] for r in linked}) == 1
    assert len(result) == len(rows) == 32
    assert sorted(r['id'] for r in result) == sorted(r['id'] for r in rows)
    assert sum(r['split'] == 'dev' for r in result) == 16
    audit = validate_partition(result, hashes)
    assert audit['exact_request_overlap'] == audit['original_group_overlap'] == 0
    assert audit['merged_components'] == 1


def test_split_is_order_independent_and_ignores_scores_and_answers():
    rows, hashes = fixture()
    alias(rows, hashes, '1:0000:SD', '1:0001:SD')
    first = split_rows(rows, hashes, SEED)
    shuffled = [dict(r, answer='unused', score=i, role='test') for i, r in enumerate(reversed(rows))]
    second = split_rows(shuffled, hashes, SEED)
    assert {r['id']: r['split'] for r in first} == {r['id']: r['split'] for r in second}


def test_same_image_but_different_question_is_not_an_identical_request():
    rows, hashes = fixture()
    hashes[rows[1]['image']] = hashes[rows[0]['image']]
    units, _ = components(rows, hashes)
    assert len(units) == 16


def test_conflicting_identical_request_labels_fail_instead_of_silently_deduplicating():
    rows, hashes = fixture()
    alias(rows, hashes, '0:0000:benign', '1:0000:SD')
    with pytest.raises(ValueError, match='conflicting labels'):
        split_rows(rows, hashes, SEED)


def test_impossible_equal_split_fails_instead_of_breaking_a_component():
    with pytest.raises(ValueError, match='cannot balance'):
        balanced_subset({'merged': [{'group': 'a'}, {'group': 'b'}]}, 1, ['a', 'b'])


def test_hashes_and_component_ids_are_recomputed_not_trusted():
    rows, hashes = fixture()
    result = split_rows(rows, hashes, SEED)
    changed = copy.deepcopy(result)
    changed[0]['split_group'] = 'fabricated'
    with pytest.raises(ValueError, match='recomputed component'):
        validate_partition(changed, hashes)
    changed = copy.deepcopy(result)
    changed[0]['image_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='differs from image manifest'):
        validate_partition(changed, hashes)


def test_aliases_cannot_cross_fit_and_calibration_and_role_counts_are_preserved():
    rows, hashes = fixture(1680)
    first, second = np.random.default_rng(SEED).permutation(1680)[:2]
    alias(rows, hashes, '1:%04d:SD' % first, '1:%04d:SD' % second)
    result = split_rows(rows, hashes, SEED)
    roles = assign_roles(result, hashes)
    assert {k: len(v) for k, v in roles.items()} == {'fit': 2688, 'calibration': 672, 'test': 3360}
    merged = next(records for records in components(result, hashes)[0].values() if len(records) == 6)
    assert {r['split'] for r in merged} == {'dev'}
    assert len({role for role, ids in roles.items() for r in merged if r['id'] in ids}) == 1
    for records in components(result, hashes)[0].values():
        assert len({role for role, ids in roles.items() for r in records if r['id'] in ids}) == 1
    # Manufacture a fit/calibration crossing without altering the original split.
    linked = next(records for records in components(result, hashes)[0].values()
                  if records[0]['split'] == 'dev' and len(records) == 3)
    owner = next(role for role, ids in roles.items() if linked[0]['id'] in ids)
    other = 'calibration' if owner == 'fit' else 'fit'
    roles[owner].remove(linked[0]['id'])
    roles[other].add(linked[0]['id'])
    with pytest.raises(ValueError, match='component crosses fit/calibration'):
        validate_partition(result, hashes, roles)


def test_previous_no_alias_calibration_assignment_is_preserved():
    rows, hashes = fixture(1680)
    result = split_rows(rows, hashes, SEED)
    roles = assign_roles(result, hashes)
    expected = set()
    for label in (0, 1):
        groups = sorted({r['group'] for r in result if r['split'] == 'dev' and r['label'] == label},
                        key=lambda g: hashlib.sha256((str(SEED) + ':cal:' + g).encode()).hexdigest())
        expected.update(r['id'] for r in result if r['group'] in set(groups[:168]))
    assert roles['calibration'] == expected


def test_old_head_package_is_rejected_before_loading_weights_or_copying(tmp_path, monkeypatch):
    (tmp_path / 'threshold.json').write_text(json.dumps({'head_sha256': 'a' * 64}))
    (tmp_path / 'protocol.json').write_text('{}')
    monkeypatch.setattr(plugguard, 'owned_read', lambda p: p)
    with pytest.raises(ValueError, match='obsolete safety split'):
        plugguard.load_checkpoint(tmp_path, [])
