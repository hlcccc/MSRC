"""Keep original questions and identical image/question inputs in one partition."""
from collections import Counter
import hashlib
from pathlib import Path


SPLIT_VERSION = 'original-and-exact-request-v2'


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def components(rows, image_hashes=None):
    """Union source groups transitively, without reading answers or model scores."""
    parent, labels, requests, digests, ids = {}, {}, {}, {}, set()

    def find(group):
        root = group
        while parent[root] != root:
            root = parent[root]
        while parent[group] != group:
            previous = parent[group]
            parent[group] = root
            group = previous
        return root

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for row in rows:
        rid, group, label = row['id'], row['group'], row['label']
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError('missing or duplicate safety sample ID')
        ids.add(rid)
        if not isinstance(group, str) or not group or label not in (0, 1):
            raise ValueError('invalid original safety group/label')
        if not isinstance(row['question'], str):
            raise ValueError('safety question must be an exact text string')
        digest = image_hashes[row['image']] if image_hashes is not None else row.get('image_sha256')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('verified image SHA256 required for safety grouping')
        if 'image_sha256' in row and row['image_sha256'] != digest:
            raise ValueError('safety row image SHA256 differs from image manifest')
        digests[rid] = digest
        if group in labels and labels[group] != label:
            raise ValueError('original safety group has inconsistent labels')
        labels[group] = label
        parent.setdefault(group, group)
        key = (digest, row['question'])
        if key in requests:
            previous = requests[key]
            if previous['label'] != label:
                raise ValueError('identical safety request has conflicting labels')
            union(group, previous['group'])
        else:
            requests[key] = row
    result = {}
    for row in rows:
        result.setdefault(find(row['group']), []).append(row)
    return result, digests


def balanced_subset(units, target, ordered_groups):
    """Select whole components; preserve the old seeded order when no aliases exist."""
    rank = {group: i for i, group in enumerate(ordered_groups)}
    members = {key: {r['group'] for r in records} for key, records in units.items()}
    if {g for values in members.values() for g in values} != set(rank):
        raise ValueError('component selection does not cover the original groups')
    ordered = sorted(members, key=lambda key: (min(rank[g] for g in members[key]), key))
    # Store the first deterministic exact subset, never a score-dependent assignment.
    paths = {0: None}
    for key in ordered:
        weight = len(members[key])
        for total in sorted(paths, reverse=True):
            new = total + weight
            if new <= target and new not in paths:
                paths[new] = (total, key)
        if target in paths:
            break
    if target not in paths:
        raise ValueError('cannot balance safety groups without splitting duplicate components')
    selected, total = set(), target
    while total:
        total, key = paths[total]
        selected.add(key)
    return selected


def split_rows(rows, image_hashes, seed):
    import numpy as np
    units, digests = components(rows, image_hashes)
    development = set()
    for label in (0, 1):
        subset = {key: records for key, records in units.items() if records[0]['label'] == label}
        groups = sorted({r['group'] for records in subset.values() for r in records})
        if not groups or len(groups) % 2:
            raise ValueError('safety original group counts must support equal halves')
        order = np.random.default_rng(seed).permutation(len(groups))
        development.update(balanced_subset(subset, len(groups) // 2, [groups[i] for i in order]))
    by_group = {r['group']: key for key, records in units.items() for r in records}
    result = [dict(row, split='dev' if by_group[row['group']] in development else 'test',
                   split_group=by_group[row['group']], image_sha256=digests[row['id']])
              for row in rows]
    validate_partition(result, image_hashes)
    return result


def validate_partition(rows, image_hashes=None, roles=None):
    units, digests = components(rows, image_hashes)
    request_sides, request_roles, original_sides = {}, {}, {}
    role_by_id = None
    if roles is not None:
        role_by_id = {}
        for role, ids in roles.items():
            for rid in ids:
                if rid in role_by_id:
                    raise ValueError('safety ID appears in multiple fitting roles')
                role_by_id[rid] = role
        if set(role_by_id) != {r['id'] for r in rows}:
            raise ValueError('safety fitting roles do not cover exactly the inputs')
    for key, records in units.items():
        if len({r['split'] for r in records}) != 1:
            raise ValueError('original/identical safety request component crosses dev/test')
        if role_by_id is not None and len({role_by_id[r['id']] for r in records}) != 1:
            raise ValueError('original/identical safety request component crosses fit/calibration/test')
        for row in records:
            if row['split'] not in ('dev', 'test'):
                raise ValueError('invalid safety split')
            if row.get('split_group') != key or row.get('image_sha256') != digests[row['id']]:
                raise ValueError('safety split group/fingerprint differs from recomputed component')
            if role_by_id is not None:
                role = role_by_id[row['id']]
                if ((role == 'test') != (row['split'] == 'test')
                        or role not in ('fit', 'calibration', 'test')):
                    raise ValueError('safety fitting role differs from dev/test partition')
            original_sides.setdefault(row['group'], set()).add(row['split'])
            request = (digests[row['id']], row['question'])
            request_sides.setdefault(request, set()).add(row['split'])
            if role_by_id is not None:
                request_roles.setdefault(request, set()).add(role_by_id[row['id']])
    counts = Counter((digests[r['id']], r['question']) for r in rows)
    return dict(split_version=SPLIT_VERSION, original_groups=len(original_sides),
                connected_groups=len(units),
                merged_components=sum(len({r['group'] for r in records}) > 1 for records in units.values()),
                duplicate_request_groups=sum(n > 1 for n in counts.values()),
                original_group_overlap=sum(len(sides) > 1 for sides in original_sides.values()),
                exact_request_overlap=sum(len(sides) > 1 for sides in request_sides.values()),
                role_component_overlap=0 if roles is not None else None,
                exact_request_role_overlap=sum(len(values) > 1 for values in request_roles.values())
                if roles is not None else None)
