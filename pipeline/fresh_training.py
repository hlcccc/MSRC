"""Reject imported fitted state in the public from-scratch acceptance flow.

This check certifies the recorded training provenance, not metric attainment.
The normal report still independently checks sample coverage and predictions.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pipeline.common import inside, read_rows, runs_dir, save, sha


CELLS = {model + '_' + dataset for model in ('llava', 'qwen', 'internvl')
         for dataset in ('pope', 'textvqa')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(run):
    run = Path(run)
    factual, safety = run / 'factual', run / 'safety'
    report = load(factual / 'report.json')
    protocol = load(factual / 'protocol.json')
    require(report.get('status') == 'FULL_FACTUAL_FLOW_COMPLETE'
            and report.get('mode') == protocol.get('mode') == 'full'
            and report.get('old_evidence_used') is False,
            'fresh full factual collection and fitting are required')
    cells = report['cells']
    require(len(cells) == 6 and {row['cell'] for row in cells} == CELLS,
            'all six newly fitted factual cells are required')
    states = {}
    for cell in sorted(CELLS):
        dataset = cell.partition('_')[2]
        raw = read_rows(factual / 'inputs' / (dataset + '.jsonl'))
        train_ids = [row['id'] for row in raw if row['acceptance_split'] == 'dev']
        test_ids = {row['id'] for row in raw if row['acceptance_split'] == 'test'}
        state_path = factual / 'calibrators' / (cell + '.json')
        state = load(state_path)
        require(train_ids and len(set(train_ids)) == len(train_ids)
                and not set(train_ids) & test_ids, 'invalid factual development/test IDs')
        require(state.get('fitting_scope') == 'development only'
                and state.get('converged') is True and state.get('train_ids') == train_ids,
                'calibrator did not use this run\'s development partition: ' + cell)
        states[cell] = dict(sha256=sha(state_path), n_train=len(train_ids))

    safety_protocol = load(safety / 'protocol.json')
    threshold = load(safety / 'threshold.json')
    safety_report = load(safety / 'report.json')
    require(safety_protocol.get('backend') == 'plugguard'
            and safety_protocol.get('execution_kind') == 'fresh_inference'
            and safety_protocol.get('training_mode') == 'development_refit'
            and safety_protocol.get('development_generated_this_run') is True,
            'a previously trained/imported safety head is not from-scratch reproduction')
    require(not threshold.get('checkpoint_reused')
            and 'origin_frozen_before_test_ns' not in threshold,
            'a frozen head/threshold package was imported')
    require(safety_protocol.get('fit_n') == 2688
            and safety_protocol.get('calibration_n') == 672
            and safety_protocol.get('test_n') == 3360,
            'safety fit/calibration/test sizes differ')
    require(threshold.get('test_used_for_fitting_or_selection') is False
            and threshold.get('frozen_before_test_ns', 0) >= safety_protocol.get('created_ns', 1),
            'the safety operating point must be frozen in this run, without test selection')
    require(safety_report.get('status') == 'FULL_SAFETY_FLOW_COMPLETE'
            and safety_report.get('old_scores_used') is False
            and safety_report.get('development_generated_this_run') is True
            and safety_report.get('training_mode') == 'development_refit',
            'safety report does not describe fresh development generation and head fitting')
    head_path = safety / 'head.pt'
    require(sha(head_path) == threshold.get('head_sha256'), 'trained head hash changed')
    return dict(scope='training provenance only; the final report checks metrics and completeness',
                factual_calibrators=states, safety_training_mode='development_refit',
                safety_protocol_sha256=sha(safety / 'protocol.json'), head_sha256=sha(head_path),
                threshold_sha256=sha(safety / 'threshold.json'),
                imported_project_training_state=False, test_used_for_fitting_or_selection=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_id):
        parser.error('invalid run ID')
    run = inside(runs_dir() / args.run_id)
    result = audit(run)
    save(run / 'FRESH_TRAINING_VERIFIED.json', result)
    print('FRESH_TRAINING_SCOPE_OK ' + args.run_id)


if __name__ == '__main__':
    main()
