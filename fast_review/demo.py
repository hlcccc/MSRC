"""Fresh fixed-sample inference; no fitting, no cached answer fallback."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from fast_review.common import (DATASETS, DONORS, FAMILIES, ROOT, SIGNALS, emit,
    indexed, load, output, require, rows, save, sha, verify_bundle)
from fast_review.review import features, predict


def plan(bundle, out):
    selection = load(bundle / 'demo/selection.json')
    require(selection['selected_using_predictions'] is False, 'outcome-selected demonstration')
    out = output(out)
    require(not out.exists(), 'fresh demonstration output required')
    out.mkdir(parents=True)
    mapped = {}
    images = {p.stem: p for p in (bundle / 'demo/images').iterdir()}
    for dataset in DATASETS:
        source = indexed(rows(bundle / 'factual/inputs' / (dataset + '.jsonl')))
        hashes = load(bundle / 'factual/image_hashes.json')
        mapped[dataset] = []
        for rid in selection['factual'][dataset]:
            row = dict(source[rid])
            image = images[hashes[row['image']]]
            require(sha(image) == hashes[row['image']] and 'answer' not in row
                    and 'evidence' not in row and row['acceptance_split'] == 'test', 'invalid fresh demo input')
            row['image'] = str(image)
            mapped[dataset].append(row)
        target = out / 'factual/inputs' / (dataset + '.jsonl')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as handle:
            for row in mapped[dataset]:
                emit(handle, row)
    source = indexed(rows(bundle / 'safety/snapshot/inputs.jsonl'))
    hashes = load(bundle / 'safety/snapshot/input_hashes.json')['images']
    safety = []
    for rid in selection['safety']:
        row = dict(source[rid])
        image = images[hashes[row['image']]]
        require(sha(image) == hashes[row['image']] and row['split'] == 'test', 'bad safety demo image/split')
        row['image'], row['role'] = str(image), 'test'
        safety.append(row)
    save(out / 'plan.json', dict(selection=selection, factual=mapped, safety=safety,
        scope='fresh functionality samples only, not full-dataset quantitative acceptance',
        evidence_manifest_sha256=sha(bundle / 'manifest.json'),
        no_fitting=True, generated_answers_imported=False))
    return mapped, safety


def collect_factual(bundle, out, raw):
    from pipeline.common import model_paths
    from pipeline.preflight import wait_for_gpu
    paths = model_paths()
    for family in FAMILIES:
        for dataset in DATASETS:
            wait_for_gpu()
            target = out / 'factual/evidence' / (family + '_' + dataset + '.jsonl')
            target.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-B', '-u', str(ROOT / 'scripts/collect_evidence_fitted.py'),
                '--data', str(out / 'factual/inputs' / (dataset + '.jsonl')),
                '--out', str(target), '--provider', family, '--model-path', paths[family],
                '--device', 'cuda:0', '--dtype', 'bfloat16', '--max-new-tokens', '128',
                '--max-pixels', '1003520', '--k', '3', '--seed', '20260920', '--save-every', '1']
            print('DEMO_COLLECT ' + family + '_' + dataset, flush=True)
            with target.with_suffix('.log').open('x', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    results = []
    for dataset in DATASETS:
        evidence = {f: rows(out / 'factual/evidence' / (f + '_' + dataset + '.jsonl')) for f in FAMILIES}
        for family in FAMILIES:
            cell = family + '_' + dataset
            readings, x, available, labels = features(raw[dataset], evidence[family], evidence[DONORS[family]], dataset)
            state = load(bundle / 'factual/calibrators' / (cell + '.json'))
            scores = predict(state, x, available)
            weights = np.zeros(8)
            weights[np.asarray(state['live'], dtype=bool)] = state['coef'][1:]
            for i, score in enumerate(scores):
                used = available[i] & np.asarray(state['live']) & (abs(weights) > 1e-12)
                results.append(dict(cell=cell, id=raw[dataset][i]['id'], answer=evidence[family][i]['answer'],
                    confidence=1 - float(score), risk=float(score), n_actual_signals=int(used.sum()),
                    signals=[dict(name=n, risk=float(x[i, j]), available=bool(available[i, j]),
                        participates=bool(used[j]), weight=float(weights[j])) for j, n in enumerate(SIGNALS)]))
    require(len(results) == 18 and min(r['n_actual_signals'] for r in results) >= 3,
            'fresh factual demonstration incomplete or fewer than three actual signals')
    save(out / 'factual/demo_results.json', results)
    return results


def collect_safety(bundle, out, raw):
    from extensions.multilayer.runner import PortableExperiment, Policy, audit_events

    class DemoExperiment(PortableExperiment):
        def setup(self):
            self.policy = Policy(**load(bundle / 'safety/protocol.json')['controller'])
            self.rows, self.dev = raw, []
            self.tests = indexed(raw)

    material = out / 'safety/materials/snapshot'
    material.mkdir(parents=True)
    with (material / 'inputs.jsonl').open('x', encoding='utf-8', newline='\n') as handle:
        for row in raw:
            emit(handle, row)
    save(material / 'input_hashes.json', dict(images={r['image']: sha(r['image']) for r in raw}))
    import shutil
    shutil.copyfile(bundle / 'demo/head.pt', material / 'head.pt')
    directions = load(bundle / 'safety/directions.json')
    experiment = DemoExperiment(out / 'safety', material.parent)
    experiment.setup()
    frozen_ns = time.time_ns()
    experiment.gpu_setup()
    model, processor = experiment.pg.load_generator()
    head = experiment.head()
    results = []
    try:
        for row in raw:
            arms, tokens, _, elapsed = experiment.decode_request(
                model, processor, head, row, fitted=directions, live_arm='temporal_hold')
            primary = next(arm for arm in arms if arm['arm'] == 'temporal_hold')
            audit_events(primary, frozen_ns)
            require(primary['generated_tokens'] == len(tokens), 'generation continued after live primary stop')
            record = dict(id=row['id'], suite=row['suite'], request_risk_label=row['label'],
                selected_arm=primary, seconds=elapsed, live_stop=True)
            results.append(record)
            save(out / 'safety' / ('sample_%02d.json' % len(results)), record)
            print('DEMO_SAFETY %d/%d' % (len(results), len(raw)), flush=True)
    finally:
        del model, processor, head
        experiment.release()
    require(len(results) == 8, 'eight fresh safety requests required')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('assets/fast_review'))
    parser.add_argument('--out', type=Path, default=Path('runs/fast_001/demo'))
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    release = load(ROOT / 'resources/fast_review/release.json')
    bundle, out = args.evidence.resolve(strict=True), output(args.out)
    verify_bundle(bundle, release['manifest_sha256'])
    raw, safety = plan(bundle, out)
    if args.plan_only:
        print('DEMO_PLAN_ONLY_NOT_EXECUTED')
        return
    from pipeline.common import initialize, select_device
    initialize(out)
    select_device()
    try:
        started = time.perf_counter()
        factual_result = collect_factual(bundle, out, raw)
        safety_result = collect_safety(bundle, out, safety)
        result = dict(format='msrc-fresh-sample-demo-v1', status='FRESH_SAMPLE_DEMO_COMPLETE',
            full_fresh_inference=False, factual_model_records=len(factual_result),
            safety_requests=len(safety_result), actual_signals_min=min(r['n_actual_signals'] for r in factual_result),
            seconds=time.perf_counter() - started, no_fitting=True,
            evidence_manifest_sha256=sha(bundle / 'manifest.json'),
            cached_answers_used=False, selection=load(out / 'plan.json')['selection'],
            files={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        save(out / 'summary.json', result)
        text = '# 现场抽测报告\n\n新推理：18条事实性模型记录、8条内容安全请求。\n\n' \
            + '最少实际参与信号数：%d。\n\n本次没有拟合；小样本不替代全量性能统计。\n' % result['actual_signals_min']
        (out / 'demo_report.md').write_text(text, encoding='utf-8', newline='\n')
        print('FRESH_SAMPLE_DEMO_COMPLETE', flush=True)
    except Exception as error:
        save(out / 'failure.json', dict(status='DEMO_FAILED', error=str(error)))
        raise


if __name__ == '__main__':
    main()
