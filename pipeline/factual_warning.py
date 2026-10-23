"""Factual answer-error warnings, after signal collection and before local release.

This is not a harmful-content classifier and not a streaming-token guard.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pipeline.common import emit, inside, read_rows, save, sha


CELLS = {m + '_' + d for m in ('llava', 'qwen', 'internvl') for d in ('pope', 'textvqa')}


def metrics(rows):
    counts = dict(TP=0, TN=0, FP=0, FN=0)
    for row in rows:
        counts[('TP' if row['warn'] else 'FN') if row['label'] else ('FP' if row['warn'] else 'TN')] += 1
    return dict(counts, n=len(rows), accuracy=(counts['TP'] + counts['TN']) / len(rows),
                all_alarm=(counts['TP'] + counts['FN']) / len(rows),
                never_alarm=(counts['TN'] + counts['FP']) / len(rows))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(root, factual):
    root, factual = Path(root), Path(factual)
    protocol = load(root / 'protocol.json')
    if (protocol.get('domain') != 'factual_answer_error'
            or protocol['factual_report_sha256'] != sha(factual / 'report.json')
            or protocol['source_sha256'] != sha(__file__)):
        raise ValueError('factual warning protocol/source changed')
    results = []
    for cell in sorted(CELLS):
        details = read_rows(factual / 'signal_usage' / (cell + '.jsonl'))
        state_path = factual / 'calibrators' / (cell + '.json')
        if sha(state_path) != protocol['states'][cell]:
            raise ValueError('factual warning used changed fitted parameters')
        state = load(state_path)
        threshold = state['warning_threshold']
        if (threshold['selected_on'] != 'development only' or threshold['test_used_for_selection'] is not False
                or threshold['tie_rule'] != 'highest' or threshold['grid_points'] != 201):
            raise ValueError('factual warning threshold is not development-frozen')
        decisions = read_rows(root / (cell + '.jsonl'))
        delivered = read_rows(root / (cell + '_release.jsonl'))
        if [r['id'] for r in decisions] != [r['id'] for r in details] or len(delivered) != len(details):
            raise ValueError('incomplete factual held-out warning coverage')
        for detail, row, release in zip(details, decisions, delivered):
            if (row['risk'] != detail['risk'] or row['label'] != detail['label']
                    or row['threshold'] != threshold['value'] or type(row['warn']) is not bool
                    or row['warn'] != (detail['risk'] >= threshold['value'])
                    or not protocol['created_ns'] <= row['decision_done_ns'] <= release['release_start_ns']
                    or release['id'] != row['id'] or release['warn'] != row['warn']
                    or release['answer'] != ('' if row['warn'] else detail['answer'])):
                raise ValueError('factual decision/release differs from frozen score')
        results.append(dict(cell=cell, threshold=threshold['value'], **metrics(decisions)))
    return dict(status='FULL_FACTUAL_WARNING_COMPLETE', domain='factual_answer_error',
                timing_scope='after generation and evidence extraction, before controlled local release',
                not_streaming_content_safety=True, cells=results,
                minimum_accuracy=min(r['accuracy'] for r in results),
                old_test_predictions_used=False, all_test_rows_present=True)


def run(factual, out):
    factual, out = Path(factual), inside(out)
    from pipeline.acceptance_check import factual_check
    factual_check(factual, load(factual / 'report.json'))
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'protocol.json', dict(domain='factual_answer_error', created_ns=time.time_ns(),
         factual_report_sha256=sha(factual / 'report.json'), source_sha256=sha(__file__),
         states={c: sha(factual / 'calibrators' / (c + '.json')) for c in sorted(CELLS)}))
    for cell in sorted(CELLS):
        threshold = load(factual / 'calibrators' / (cell + '.json'))['warning_threshold']['value']
        with (out / (cell + '.jsonl')).open('x', encoding='utf-8') as decisions, \
                (out / (cell + '_release.jsonl')).open('x', encoding='utf-8') as releases:
            for detail in read_rows(factual / 'signal_usage' / (cell + '.jsonl')):
                warn = detail['risk'] >= threshold
                emit(decisions, dict(id=detail['id'], risk=detail['risk'], label=detail['label'],
                     threshold=threshold, warn=warn, decision_done_ns=time.time_ns()))
                emit(releases, dict(id=detail['id'], warn=warn, release_start_ns=time.time_ns(),
                                   answer='' if warn else detail['answer']))
    report = audit(out, factual)
    save(out / 'report.json', report)
    print('FACTUAL_WARNING_COMPLETED ' + out.parent.name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factual', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.factual, args.out)


if __name__ == '__main__':
    main()
