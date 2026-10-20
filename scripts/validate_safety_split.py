"""Validate a new safety split from image fingerprints, without running a model.

This does not produce a performance result. Fresh preparation hashes live images;
an offline invocation can only verify the supplied fingerprint manifest.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import SEED, inside, save, sha
from pipeline.plugguard_report import assign_roles
from pipeline.safety_split import source_sha256, split_rows, validate_partition


def validate(inputs, image_manifest, out):
    # Preserve foreign-platform image path strings in archived input manifests.
    rows = [json.loads(line) for line in Path(inputs).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    manifest = json.loads(Path(image_manifest).read_text(encoding='utf-8'))
    hashes = manifest['images']
    before = defaultdict(set)
    for row in rows:
        before[(hashes[row['image']], row['question'])].add(row['split'])
    revised = split_rows(rows, hashes, SEED)
    roles = assign_roles(revised, hashes)
    audit = validate_partition(revised, hashes, roles)
    by_id = {r['id']: r for r in rows}
    changed = [r['id'] for r in revised if r['split'] != by_id[r['id']]['split']]
    report = dict(status='SPLIT_CHECK_OK_NOT_MODEL_EVALUATION', n=len(revised), seed=SEED,
                  source_inputs_sha256=sha(inputs), source_image_manifest_sha256=sha(image_manifest),
                  split_source_sha256=source_sha256(),
                  basis='supplied image SHA256 manifest; live image bytes not rehashed by this command',
                  old_exact_request_overlap=sum(len(sides) > 1 for sides in before.values()),
                  split_integrity=audit, roles={k: len(v) for k, v in roles.items()},
                  counts=[dict(split=side, suite=suite, label=label, n=n)
                          for (side, suite, label), n in sorted(Counter(
                              (r['split'], r['suite'], r['label']) for r in revised).items())],
                  moved_rows=len(changed), moved_ids=sorted(changed),
                  model_inference_performed=False, performance_result=None,
                  old_head_and_threshold_reusable=False)
    out = inside(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in revised),
                                     encoding='utf-8')
    save(out / 'input_hashes.json', manifest)
    save(out / 'roles.json', {k: sorted(v) for k, v in roles.items()})
    save(out / 'split_validation.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--image-hashes', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    validate(args.inputs, args.image_hashes, args.out)


if __name__ == '__main__':
    main()
