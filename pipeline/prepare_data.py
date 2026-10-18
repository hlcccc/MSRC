"""Rebuild acceptance inputs from public annotations/parquet and benign raw data.

No prediction, evidence, precomputed split, or risk-score file is consumed.
Image paths in the delivered JSONL are relative to that JSONL, not this machine.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
from pathlib import Path

from pipeline.common import ROOT, emit, initialize, inside, read_rows, save, sha


def copy_file(source, target):
    source, target = Path(source).resolve(), inside(target)
    if not source.is_file():
        raise ValueError('missing raw asset: %s' % source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != sha(source):
            raise ValueError('conflicting raw asset: %s' % target)
    else:
        shutil.copy2(source, target)
    return target


def write_rows(path, rows):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate raw sample ID: %s' % path)
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            row = dict(row)
            row['image'] = os.path.relpath(row['image'], path.parent).replace('\\', '/')
            emit(handle, row)


def factual_inputs(sources, assets):
    from PIL import Image
    factual = assets / 'data/factual'
    rows = []
    for split in ('random', 'popular', 'adversarial'):
        name = 'coco_pope_%s.json' % split
        ref = copy_file(Path(sources['pope_annotations']) / name, assets / 'raw/pope' / name)
        original = [json.loads(line) for line in ref.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(original) != 3000 or len({r['image'] for r in original}) != 500:
            raise ValueError('official POPE requires 3000 questions / 500 images per setting')
        for r in original:
            image = copy_file(Path(sources['pope_images']) / r['image'],
                              assets / 'data/images/pope' / r['image'])
            rows.append(dict(id='pope_official:%s:%s' % (split, r['question_id']),
                             source_image_id=Path(r['image']).stem, question=r['text'],
                             image=str(image), dataset='POPE', split=split,
                             gold_answers=[r['label']], answer_format='yes_no',
                             answer_parser='pope_official'))
    if len({r['source_image_id'] for r in rows}) != 500:
        raise ValueError('POPE settings must share the same 500 source images')
    write_rows(factual / 'pope.jsonl', rows)
    annotation = copy_file(sources['textvqa_annotations'], assets / 'raw/textvqa/TextVQA_0.5.1_val.json')
    original = json.loads(annotation.read_text(encoding='utf-8'))['data']
    if len(original) != 5000 or len({r['image_id'] for r in original}) != 3166:
        raise ValueError('official TextVQA validation requires 5000 questions / 3166 images')
    rows = []
    for r in original:
        if len(r['answers']) != 10:
            raise ValueError('ten official TextVQA human answers required')
        image = copy_file(Path(sources['textvqa_images']) / (r['image_id'] + '.jpg'),
                          assets / 'data/images/textvqa' / (r['image_id'] + '.jpg'))
        rows.append(dict(id='textvqa:%s' % r['question_id'], source_image_id=r['image_id'],
                         question=r['question'], image=str(image), dataset='TextVQA',
                         gold_answers=r['answers']))
    write_rows(factual / 'textvqa.jsonl', rows)
    images = list((assets / 'data/images/pope').iterdir()) + list((assets / 'data/images/textvqa').iterdir())
    for path in images:
        with Image.open(path) as image:
            image.verify()
    return dict(pope=9000, pope_images=500, textvqa=5000, textvqa_images=3166)


def safety_inputs(sources, assets):
    import pyarrow.parquet as pq
    from PIL import Image
    parquet_root = Path(sources['mmsafety_parquet'])
    categories = sorted(p.name for p in parquet_root.iterdir() if p.is_dir()
                        and (p / 'TYPO.parquet').is_file())
    if len(categories) != 13:
        raise ValueError('MM-SafetyBench requires all 13 categories')
    typo, other = [], []
    for category in categories:
        origin_sets = []
        for variant in ('TYPO', 'SD', 'SD_TYPO'):
            path = copy_file(parquet_root / category / (variant + '.parquet'),
                             assets / 'data/MM-SafetyBench/data' / category / (variant + '.parquet'))
            originals = pq.read_table(path, columns=['id', 'question', 'image'], use_threads=False).to_pylist()
            source_ids = [str(r['id']) for r in originals]
            if len(set(source_ids)) != len(source_ids):
                raise ValueError('duplicate official source ID')
            origin_sets.append(set(source_ids))
            for idx, r in enumerate(originals):
                raw = r['image']
                if isinstance(raw, dict):
                    raw = raw['bytes']
                if not isinstance(raw, bytes):
                    raise ValueError('expected embedded original image bytes')
                image = inside(assets / 'data/images/mmsafety' / category / ('%s_%04d.png' % (variant, idx)))
                image.parent.mkdir(parents=True, exist_ok=True)
                if image.exists():
                    raise ValueError('fresh decoded image required: %s' % image)
                with Image.open(io.BytesIO(raw)) as decoded:
                    decoded.convert('RGB').save(image, format='PNG')
                suffix = str(idx) if variant == 'TYPO' else str(r['id'])
                row = dict(id='mmsafety:%s:%s:%s' % (category, variant, suffix),
                           question=str(r['question']), image=str(image), dataset='MM-SafetyBench',
                           category=category, attack=variant)
                (typo if variant == 'TYPO' else other).append(row)
        if not origin_sets[0] == origin_sets[1] == origin_sets[2]:
            raise ValueError('MM-SafetyBench variants have different source-ID coverage')
    matched, native = [], []
    original_benign = read_rows(sources['benign_annotations'])
    if len(original_benign) != 1680:
        raise ValueError('the frozen custom benign dataset must contain 1680 rows')
    for r in original_benign:
        if (r.get('answer') or r.get('evidence') or 'p_harm' in r or 'split' in r
                or r.get('label') != 0 or not r.get('phrase')):
            raise ValueError('benign input must be raw labeled data, not inference/split output')
        image = copy_file(r['image'], assets / 'data/images/benign' / Path(r['image']).name)
        with Image.open(image) as decoded:
            decoded.verify()
        matched.append(dict(id=r['id'], question=r['question'], image=str(image),
                            phrase=r['phrase'], tier=r.get('tier', 'benign'),
                            label=0, category=r.get('category', 'benign')))
        native.append(dict(id='benign_native:' + r['id'], question=r['question'], image=str(image),
                           label=0, category=r.get('tier', 'benign'), suite='benign'))
    if len(typo) != 1680 or len(other) != 3360:
        raise ValueError('all three MM-SafetyBench image variants (5040 rows) required')
    for name, rows in [('mmsafety_all', typo), ('mmsafety_sd', other),
                       ('matched_benign', matched), ('benign_native', native)]:
        write_rows(assets / 'data/safety' / (name + '.jsonl'), rows)
    return dict(mmsafety=5040, benign=1680, total=6720, parquet_files=39)


def prepare(sources, assets):
    assets = inside(assets)
    if (assets / 'data').exists() or (assets / 'raw').exists():
        raise ValueError('data/raw already exist; use a fresh asset directory')
    factual = factual_inputs(sources, assets)
    safety = safety_inputs(sources, assets)
    save(assets / 'preparation.json', dict(
        status='DATA_REPROCESSED_FROM_RAW', sources=sources, factual=factual, safety=safety,
        old_predictions_read=False, split_files_read=False, script_sha256=sha(__file__)))
    files = sorted(p for sub in ('data', 'raw') for p in (assets / sub).rglob('*') if p.is_file())
    files.append(assets / 'preparation.json')
    manifest = inside(assets / 'MANIFEST.sha256')
    with manifest.open('x', encoding='utf-8') as handle:
        for path in files:
            handle.write('%s  %s\n' % (sha(path), path.relative_to(assets).as_posix()))
    print('RAW_DATA_PREPARED: POPE=9000 TextVQA=5000 safety=6720; %d hashed files' % len(files), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    args = parser.parse_args()
    initialize()
    sources = json.loads(args.sources.read_text(encoding='utf-8'))
    sources = {name: str((args.sources.parent / value).resolve())
               if not Path(value).is_absolute() else value for name, value in sources.items()}
    prepare(sources, args.assets)


if __name__ == '__main__':
    main()
