"""Audit completed scheduling evidence without changing the measured artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

from pipeline.safety_judge_speed import candidate_admission, compare_judgments


def audit(directory):
    from extensions.multilayer.runner import parse_response_safety

    directory = Path(directory).resolve(strict=True)
    destinations = [directory / 'audited_admission.json', directory / 'audit.md']
    if any(path.exists() for path in destinations):
        raise ValueError('retain earlier audits; select a fresh evidence directory')
    files = ['protocol.json', 'cohort.json', 'report.json', 'serial_reference.json',
             'measurements.json', 'fresh_outputs.json', 'input_tensor_audit.json']
    contents = {name: (directory / name).read_bytes() for name in files}
    documents = {name: json.loads(value) for name, value in contents.items()}
    reference = documents['serial_reference.json']['outputs']
    fresh = documents['fresh_outputs.json']
    cohort = documents['cohort.json']
    rows, differences = [], []
    for source in documents['measurements.json']:
        item = dict(source)
        if 'seconds' not in item:
            rows.append(item)
            continue
        name = item['variant'] + '_' + str(item['repeat'])
        actual = fresh[name]
        item['comparison'] = compare_judgments(reference, actual, parse_response_safety)
        if item['comparison'] != source['comparison']:
            raise ValueError('saved comparison does not match raw judge outputs')
        baseline = fresh['original_' + str(item['repeat'])]
        item['comparison_to_paired_baseline'] = compare_judgments(baseline, actual, parse_response_safety)
        rows.append(item)
        for i, (left, right) in enumerate(zip(reference, actual)):
            if left != right:
                differences.append(dict(variant=item['variant'], repeat=item['repeat'], index=i,
                    origin=cohort['origins'][i], serial_output=left, actual_output=right,
                    serial_parsed=parse_response_safety(left), actual_parsed=parse_response_safety(right)))
    comparisons = candidate_admission(rows)
    eligible = any(row['eligible_for_further_validation'] for row in comparisons)
    result = dict(verdict=('ELIGIBLE_FOR_FURTHER_VALIDATION_NOT_ACCEPTANCE' if eligible
        else 'NOT_ADMITTED_AS_OUTPUT_EQUIVALENT_DROP_IN'),
        fresh_judge_only=True, primary_warning_accuracy_retested=False,
        formal_pipeline_modified=False, full_acceptance=False,
        source_sha256={name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparisons=comparisons, differences=differences, measurements=rows)
    lines = ['# Audited single-GPU safety judge trial', '',
        'This is a development-only A100 judge-stage timing trial, not a full acceptance test.',
        'The measured original files are retained without changes. This audit supersedes their admission flags.',
        'Both the timed baseline and each candidate must preserve fresh serial outputs before admission.', '',
        '| Variant | Round | Seconds | Raw matches serial | Raw matches batch-4 baseline |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        if 'seconds' in row:
            lines.append('| %s | %d | %.3f | %d/%d | %d/%d |' % (row['variant'], row['repeat'],
                row['seconds'], row['comparison']['raw_matches'], len(reference),
                row['comparison_to_paired_baseline']['raw_matches'], len(reference)))
    lines += ['', '## Decision', '', result['verdict'], '',
        '%d output differences versus fresh serial inference are retained in audited_admission.json.' % len(differences),
        'When the timed baseline differs from serial, a candidate cannot be admitted as an output-equivalent replacement.',
        'No formal or cloud pipeline was changed. No candidate is activated automatically.',
        'The primary request-warning accuracy was not rerun; this does not establish a threshold failure.', '',
        '```json', json.dumps(result['comparisons'], indent=2), '```', '']
    destinations[0].write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    destinations[1].write_text('\n'.join(lines), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_directory', type=Path)
    args = parser.parse_args()
    result = audit(args.result_directory)
    print(result['verdict'])


if __name__ == '__main__':
    main()
