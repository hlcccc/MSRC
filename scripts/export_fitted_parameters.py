"""Publisher-only: export project-fitted state, never public base weights/test predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import emit, inside, read_rows, save, sha
from pipeline.factual_parameters import export as export_factual
from pipeline.plugguard_report import audit, checkpoint_identity
from pipeline.safety_split import source_sha256


def export(factual_source, safety_source, out):
    out = inside(out)
    out.mkdir(parents=True, exist_ok=False)
    export_factual(factual_source, out / 'factual.json')
    safety_source = Path(safety_source)
    result = audit(safety_source)
    if result['training_mode'] != 'development_refit' or result['old_scores_used']:
        raise ValueError('original freshly fitted safety run required')
    head = inside(out / 'plugguard')
    head.mkdir()
    rows = read_rows(safety_source / 'inputs.jsonl')
    image_hashes = json.loads((safety_source / 'input_hashes.json').read_text(encoding='utf-8'))['images']
    sanitized = [dict(r, image='image_sha256/' + image_hashes[r['image']] + '.png') for r in rows]
    new_hashes = {r['image']: image_hashes[old['image']] for r, old in zip(sanitized, rows)}
    if checkpoint_identity(rows, image_hashes) != checkpoint_identity(sanitized, new_hashes):
        raise ValueError('parameter export changed portable safety identity')
    with (head / 'training_inputs.jsonl').open('x', encoding='utf-8') as f:
        for row in sanitized:
            emit(f, row)
    save(head / 'input_hashes.json', dict(images=new_hashes))
    protocol = json.loads((safety_source / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['split_source_sha256'] != source_sha256():
        original = safety_source.parents[2] / 'pipeline/safety_split.py'
        current = Path(__file__).resolve().parents[1] / 'pipeline/safety_split.py'
        if (sha(original) != protocol['split_source_sha256']
                or original.read_bytes().replace(b'\r\n', b'\n') != current.read_bytes().replace(b'\r\n', b'\n')):
            raise ValueError('safety split source differs beyond line endings')
        protocol['export_verified_line_ending_only_split_source'] = protocol['split_source_sha256']
        protocol['split_source_sha256'] = source_sha256()
    protocol['source_inputs_sha256'] = sha(head / 'training_inputs.jsonl')
    protocol['input_hashes_sha256'] = sha(head / 'input_hashes.json')
    protocol['export_origin_protocol_sha256'] = sha(safety_source / 'protocol.json')
    protocol['training_provenance_is_summary_only'] = True
    save(head / 'protocol.json', protocol)
    for name in ('head.pt', 'threshold.json', 'dataset_identity.json', 'calibration_scores.jsonl'):
        shutil.copyfile(safety_source / name, head / name)
    # Import auditing needs development identity/labels, not the generated answer text.
    with (head / 'development_generation.jsonl').open('x', encoding='utf-8') as f:
        for row in read_rows(safety_source / 'development_generation.jsonl'):
            emit(f, dict(id=row['id'], label=row['label'], provenance_summary_only=True))
    manifest = dict(format='msrc-provided-parameters-v1',
                    contains_base_model_weights=False, contains_test_predictions=False,
                    safety_head_sha256=sha(head / 'head.pt'),
                    development_origin_sha256=sha(safety_source / 'development_generation.jsonl'),
                    files={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    save(out / 'manifest.json', manifest)
    archive = inside(out.with_suffix('.tar.gz'))
    if archive.exists():
        raise ValueError('new parameter archive required')
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(out).as_posix(), recursive=False)
    print(json.dumps(dict(status='FITTED_PARAMETERS_EXPORTED', archive=str(archive),
                          sha256=sha(archive), bytes=archive.stat().st_size,
                          head_sha256=manifest['safety_head_sha256'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factual-source', type=Path, required=True)
    parser.add_argument('--safety-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.factual_source, args.safety_source, args.out)


if __name__ == '__main__':
    main()
