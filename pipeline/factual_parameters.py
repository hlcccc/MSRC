"""Portable fitted factual parameters; importing never estimates new parameters."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from pipeline.common import inside, read_rows, save, sha


FORMAT = 'msrc-factual-parameters-v1'
METHOD_KEYS = ('seed', 'max_new_tokens', 'k', 'views', 'probes',
               'resample_temperature', 'top_p', 'max_pixels', 'attention',
               'image_contrast', 'selected_signals', 'donors', 'baseline',
               'textvqa_labels', 'l2', 'bins', 'missing')


def text_sha256(path):
    """Platform line endings may differ; all other source bytes must be identical."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def source_fingerprints():
    from pipeline.factual import SOURCE, _source_files
    return {str(p.relative_to(SOURCE)): text_sha256(p) for p in _source_files()}


def identity(rows):
    """Ignore installation paths, but bind every input, label and split to image bytes."""
    images, values = {}, []
    for row in rows:
        image = row['image']
        if image not in images:
            images[image] = sha(image)
        value = {k: v for k, v in row.items() if k != 'image'}
        value['image_sha256'] = images[image]
        values.append(value)
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':')).encode()).hexdigest()


def features(out, family, dataset):
    from pipeline.factual import DONORS, SIGNALS, SOURCE
    from msrc.signals import Evidence, build_report
    from msrc.types import Sample
    raw = read_rows(out / 'inputs' / (dataset + '.jsonl'))
    rows = read_rows(out / 'evidence' / (family + '_' + dataset + '.jsonl'))
    donors = read_rows(out / 'evidence' / (DONORS[family] + '_' + dataset + '.jsonl'))
    if [r['id'] for r in rows] != [r['id'] for r in raw] or len(donors) != len(rows):
        raise ValueError('incomplete or reordered fresh evidence')
    official = None
    if dataset == 'textvqa':
        spec = importlib.util.spec_from_file_location('official_mmf', SOURCE / 'references/m4c_evaluator.py')
        mmf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mmf)
        official = mmf.TextVQAAccuracyEvaluator()
    readings, labels = [], []
    for row, donor in zip(rows, donors):
        if row['id'] != donor['id'] or row['question'] != donor['question']:
            raise ValueError('donor is not the same sample/question')
        evidence = Evidence.from_dict(row['evidence'])
        evidence.cross_model = [Sample(text=donor['answer'], meta={
            'fresh_donor': DONORS[family], 'source_id': donor['id']})]
        readings.append({s.name: s for s in build_report(evidence).signals})
        if official is not None:
            if len(row['gold_answers']) != 10:
                raise ValueError('ten official answers required')
            soft = official._compute_answer_scores(row['gold_answers']).get(
                official.answer_processor(row['answer']), 0.)
            labels.append(int(soft < .5))
        else:
            labels.append(int(row['label']))
    x = np.asarray([[v[n].risk for n in SIGNALS] for v in readings])
    available = np.asarray([[v[n].available for n in SIGNALS] for v in readings])
    dev = np.asarray([r['acceptance_split'] == 'dev' for r in raw])
    return raw, rows, readings, x, available, np.asarray(labels), dev


def predict(state, x, available):
    from msrc.model import RiskCalibrator
    live = np.asarray(state['live'], dtype=bool)
    values = np.where(available, x, np.asarray(state['fill']))
    model = RiskCalibrator()
    model.coef_ = np.asarray(state['coef'], dtype=float)
    model.mean_ = np.asarray(state['mean'], dtype=float)
    model.std_ = np.asarray(state['std'], dtype=float)
    return model.predict_proba(values[:, live])


def warning_threshold(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if not len(labels) or len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError('nonempty finite development scores required')
    grid = np.round(np.linspace(0, 1, 201), 4)
    counts = [int(np.count_nonzero((scores >= t) == labels)) for t in grid]
    index = max(range(len(grid)), key=lambda i: (counts[i], grid[i]))
    return dict(value=float(grid[index]), grid_points=201, tie_rule='highest',
                selected_on='development only', n_dev=len(labels),
                dev_correct=counts[index], test_used_for_selection=False)


def validate_state(state, signals, train_ids):
    live = np.asarray(state['live'], dtype=bool)
    if (state['signals'] != signals or len(live) != len(signals) or not live.any()
            or state['train_ids'] != train_ids or state.get('converged') is not True
            or state.get('fitting_scope') != 'development only'):
        raise ValueError('fitted signal order/training identity differs')
    for name, length in [('fill', len(signals)), ('mean', int(live.sum())),
                         ('std', int(live.sum())), ('coef', int(live.sum()) + 1)]:
        values = np.asarray(state[name], dtype=float)
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError('invalid fitted parameter: ' + name)
    if (np.asarray(state['std']) <= 0).any():
        raise ValueError('nonpositive fitted scale')
    threshold = state['warning_threshold']
    if (not 0 <= threshold['value'] <= 1 or threshold['grid_points'] != 201
            or threshold['tie_rule'] != 'highest'
            or threshold['selected_on'] != 'development only'
            or threshold['test_used_for_selection'] is not False):
        raise ValueError('invalid frozen factual warning threshold')


def load_package(path, out, protocol):
    from pipeline.factual import MODELS, SIGNALS, SOURCE
    package = json.loads(Path(path).read_text(encoding='utf-8'))
    if package.get('format') != FORMAT:
        raise ValueError('unknown fitted factual parameter format')
    if package['method'] != {k: protocol[k] for k in METHOD_KEYS}:
        raise ValueError('fitted parameters use different inference/calibration settings')
    if package['sources'] != source_fingerprints():
        raise ValueError('fitted parameters use different signal/model/scoring source')
    if package['models_recipe_sha256'] != text_sha256(SOURCE / 'resources/reproduction/models.json'):
        raise ValueError('fitted parameters use different public model recipe')
    if set(package['states']) != {m + '_' + d for m in MODELS for d in ('pope', 'textvqa')}:
        raise ValueError('all six fitted factual states required')
    for dataset in ('pope', 'textvqa'):
        rows = read_rows(out / 'inputs' / (dataset + '.jsonl'))
        if identity(rows) != package['datasets'][dataset]:
            raise ValueError('fitted parameters belong to different inputs/images/splits')
        ids = [r['id'] for r in rows if r['acceptance_split'] == 'dev']
        for family in MODELS:
            validate_state(package['states'][family + '_' + dataset], SIGNALS, ids)
    return package


def export(source, destination):
    """Publisher operation: retain six fitted states and derive thresholds on dev only."""
    from pipeline.factual import MODELS, SIGNALS, SOURCE
    source, destination = Path(source), inside(destination)
    if destination.exists():
        raise ValueError('new parameter file required')
    protocol = json.loads((source / 'protocol.json').read_text(encoding='utf-8'))
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    sources = source_fingerprints()
    if (protocol['mode'] != 'full' or report['status'] != 'FULL_FACTUAL_FLOW_COMPLETE'
            or set(protocol['source_hashes']) != set(sources) or report['old_evidence_used'] is not False):
        raise ValueError('compatible full source run required for parameter export')
    normalized = {}
    origin_root = source.parents[2]
    for name, digest in protocol['source_hashes'].items():
        if digest != sources[name]:
            original, current = origin_root / name, SOURCE / name
            if (not original.is_file() or sha(original) != digest
                    or original.read_bytes().replace(b'\r\n', b'\n') != current.read_bytes().replace(b'\r\n', b'\n')):
                raise ValueError('fitted source differs beyond line endings: ' + name)
            normalized[name] = dict(origin_sha256=digest, released_sha256=sources[name],
                                    difference='CRLF/LF only; identical normalized bytes')
    states = {}
    for family in MODELS:
        for dataset in ('pope', 'textvqa'):
            cell = family + '_' + dataset
            state = json.loads((source / 'calibrators' / (cell + '.json')).read_text(encoding='utf-8'))
            raw, _, _, x, available, y, dev = features(source, family, dataset)
            # Test features/labels are never passed into threshold selection.
            state['warning_threshold'] = warning_threshold(predict(state, x[dev], available[dev]), y[dev])
            validate_state(state, SIGNALS, [r['id'] for r in raw if r['acceptance_split'] == 'dev'])
            states[cell] = state
    result = dict(format=FORMAT, exported_ns=time.time_ns(), source_run=source.parent.name,
                  origin_protocol_sha256=sha(source / 'protocol.json'),
                  verified_line_ending_only_differences=normalized,
                  method={k: protocol[k] for k in METHOD_KEYS}, sources=sources,
                  models_recipe_sha256=text_sha256(SOURCE / 'resources/reproduction/models.json'),
                  datasets={d: identity(read_rows(source / 'inputs' / (d + '.jsonl')))
                            for d in ('pope', 'textvqa')}, states=states,
                  contains_test_predictions=False, test_used_for_fitting_or_selection=False)
    save(destination, result)
    print('FACTUAL_PARAMETERS_EXPORTED ' + str(destination), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.out)


if __name__ == '__main__':
    main()
