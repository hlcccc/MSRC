"""Publisher-only: copy full raw evidence without changing historical results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

from fast_review.common import (CORE, DATASETS, FAMILIES, FORMAT, ROOT, core_contract,
                                emit, load, output, require, rows, save, sha)


def export(factual, experiment, safety_assets, destination):
    destination = output(destination)
    require(not destination.exists(), 'use a new export directory')
    factual, experiment, safety_assets = [Path(p).resolve(strict=True)
                                          for p in (factual, experiment, safety_assets)]
    if Path.home() == Path('/home/HLC'):
        import os
        for root in (factual, experiment, safety_assets):
            require(root.is_relative_to(Path('/mnt/data/HLC')) and root.stat().st_uid == os.getuid(),
                    'source must be HLC-owned')
    destination.mkdir(parents=True)
    origins = []

    def copy(source, name):
        source = Path(source)
        require(source.is_file() and not source.is_symlink(), 'missing source: ' + str(source))
        if Path.home() == Path('/home/HLC'):
            require(source.resolve().is_relative_to(Path('/mnt/data/HLC'))
                    and source.stat().st_uid == os.getuid(), 'source outside HLC boundary')
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(sha(source) == sha(target), 'source changed during copy')
        origins.append(dict(file=name, source=str(source), sha256=sha(source)))

    protocol = load(factual / 'protocol.json')
    require(protocol['mode'] == 'full', 'full factual source required')
    for name in CORE[:9]:
        if name in protocol['source_hashes']:
            require(core_contract()[name] == protocol['source_hashes'][name],
                    'historical factual scoring differs: ' + name)
    for name in ('protocol.json', 'report.json'):
        copy(factual / name, 'factual/' + name)
    image_hashes = {}
    demo = dict(factual={}, safety=[])
    for dataset in DATASETS:
        copy(factual / 'inputs' / (dataset + '.jsonl'), 'factual/inputs/' + dataset + '.jsonl')
        raw = rows(factual / 'inputs' / (dataset + '.jsonl'))
        for row in raw:
            path = row['image']
            if path not in image_hashes:
                image_hashes[path] = sha(Path(path).resolve(strict=True))
        strata = ('random', 'popular', 'adversarial') if dataset == 'pope' else ('val',)
        selected = []
        for stratum in strata:
            candidates = [r for r in raw if r['acceptance_split'] == 'test'
                          and (dataset != 'pope' or r['split'] == stratum)]
            candidates.sort(key=lambda r: hashlib.sha256(('fast-demo-v1:' + r['id']).encode()).hexdigest())
            selected += candidates[:1 if dataset == 'pope' else 3]
        demo['factual'][dataset] = [r['id'] for r in selected]
        for row in selected:
            copy(row['image'], 'demo/images/' + image_hashes[row['image']] + Path(row['image']).suffix.lower())
        for family in FAMILIES:
            cell = family + '_' + dataset
            for folder in ('evidence', 'calibrators', 'signal_usage'):
                ext = '.json' if folder == 'calibrators' else '.jsonl'
                copy(factual / folder / (cell + ext), 'factual/' + folder + '/' + cell + ext)
    save(destination / 'factual/image_hashes.json', image_hashes)
    for name in ('protocol.json', 'frozen.json', 'directions.json', 'calibration.json',
                 'report.json', 'live_benchmark.json', 'environment.json'):
        copy(experiment / name, 'safety/' + name)
    for name in ('inputs.jsonl', 'input_hashes.json', 'dataset_identity.json', 'threshold.json'):
        copy(safety_assets / 'snapshot' / name, 'safety/snapshot/' + name)
    copy(safety_assets / 'controller_reference.json', 'safety/controller_reference.json')
    copy(safety_assets / 'snapshot/head.pt', 'demo/head.pt')
    for stage in ('test', 'answers'):
        sources = sorted((experiment / 'records' / stage).glob('*.json'))
        require(len(sources) == 3360, 'full safety records required')
        target = destination / 'safety' / (stage + '.jsonl')
        with target.open('x', encoding='utf-8', newline='\n') as handle:
            for source in sources:
                value = load(source)
                require(source.stem == hashlib.sha256(value['id'].encode()).hexdigest(), 'bad journal ID')
                emit(handle, value)
        save(destination / 'safety' / (stage + '_source_hashes.json'),
             {p.name: sha(p) for p in sources})
    raw = rows(safety_assets / 'snapshot/inputs.jsonl')
    hashes = load(safety_assets / 'snapshot/input_hashes.json')['images']
    roles = load(safety_assets / 'snapshot/dataset_identity.json')['roles']
    for suite in ('SD', 'SD_TYPO', 'TYPO', 'benign'):
        candidates = [r for r in raw if r['id'] in set(roles['test']) and r['suite'] == suite]
        candidates.sort(key=lambda r: hashlib.sha256(('fast-demo-v1:' + r['id']).encode()).hexdigest())
        demo['safety'] += [r['id'] for r in candidates[:2]]
        for row in candidates[:2]:
            copy(row['image'], 'demo/images/' + hashes[row['image']] + Path(row['image']).suffix.lower())
    save(destination / 'demo/selection.json', dict(**demo,
         selection='lowest SHA256(fast-demo-v1:ID), by input-only dataset/source strata',
         selected_using_predictions=False, factual_per_model=6, safety_total=8))
    save(destination / 'provenance.json', dict(
        factual_run=factual.parent.name, safety_run=experiment.name,
        independent_source_runs=True, origins=origins, exported_ns=time.time_ns(),
        factual_full_model_records=42000, safety_full_test_requests=3360,
        safety_response_judgments=16800, full_fresh_inference=False,
        safety_prior_test_already_observed=True,
        note='These are two historical runs, not a fresh joint acceptance experiment.'))
    files = {p.relative_to(destination).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(destination.rglob('*')) if p.is_file()}
    manifest = dict(format=FORMAT, full_fresh_inference=False, core_contract=core_contract(), files=files)
    save(destination / 'manifest.json', manifest)
    archive = output(destination.parent / 'MSRC-fast-evidence-20261102.tar.gz')
    require(not archive.exists(), 'archive already exists')
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(destination).as_posix(), recursive=False)
    release = dict(format='msrc-fast-review-release-v1', archive=archive.name,
                   archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
                   manifest_sha256=sha(destination / 'manifest.json'))
    save(destination.parent / 'release.json', release)
    print(json.dumps(release), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('factual', 'experiment', 'safety-assets', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    export(args.factual, args.experiment, args.safety_assets, args.out)


if __name__ == '__main__':
    main()
