"""Maintainer: export only fitted development state from an audited experiment.

No base weights, images, test predictions or answer reviews enter the package.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.multilayer import runner
from pipeline.plugguard_report import checkpoint_identity


def export(experiment, snapshot, output):
    experiment = runner.owned_read(experiment)
    snapshot = runner.owned_read(snapshot)
    output = Path(output).absolute()
    if Path.home() == Path('/home/HLC'):
        runner.require(output.resolve().is_relative_to(Path('/mnt/data/HLC'))
                       and output.resolve() == output, 'HLC export must stay in its data tree')
    runner.require(not output.exists(), 'new export directory required')
    frozen = runner.load(experiment / 'frozen.json')
    runner.require(frozen['test_used_for_selection'] is False
                   and all(frozen['new_detection_enabled'].get(k) for k in ('semantic', 'temporal')),
                   'both added detectors must have development-frozen evidence')
    for name, digest in frozen['files'].items():
        path = runner.owned_read(experiment / name)
        runner.require(path.is_relative_to(experiment) and runner.sha(path) == digest,
                       'frozen training artifact changed: ' + name)
    rows, roles, audit = runner.load_partition(snapshot)
    protocol = runner.load(experiment / 'protocol.json')
    runner.require(protocol['settings'] == runner.SETTINGS and
                   all(set(protocol['roles'][k]) == v for k, v in roles.items()),
                   'historical experiment uses different settings or roles')
    from safety_lab.prefix_data import parse_response_safety
    records = lambda stage: [runner.load(p) for p in sorted((experiment / 'records' / stage).glob('*.json'))]
    dev = [dict(r, role=next(k for k, ids in roles.items() if r['id'] in ids))
           for r in rows.values() if r['id'] not in roles['test']]
    labels = [r for item in records('labels') for r in item['labels']]
    runner.validate_prefixes(records('development'), dev, labels, complete=True)
    report = runner.load(experiment / 'report.json')
    runner.require(report['status'] == 'COMPLETE' and report['full_test_requests'] == 3360
                   and report['full_answer_reviews'] == 16800,
                   'source experiment was not a complete five-arm experiment')
    hashes = runner.load(snapshot / 'input_hashes.json')['images']
    controller = protocol['controller']
    threshold = runner.load(snapshot / 'threshold.json')
    runner.require(controller['hard_threshold'] == threshold['threshold']
                   and runner.sha(snapshot / 'head.pt') == threshold['head_sha256'], 'source baseline differs')
    origin = dict(format='msrc-multilayer-fit-provenance-v1',
        source_experiment=experiment.parent.parent.name,
        source_frozen_sha256=runner.sha(experiment / 'frozen.json'),
        source_report_sha256=runner.sha(experiment / 'report.json'),
        original_frozen_ns=frozen['frozen_ns'], settings=runner.SETTINGS,
        core_files=runner.verify_core(), controller=controller,
        baseline_head_sha256=threshold['head_sha256'],
        baseline_threshold_sha256=runner.sha(snapshot / 'threshold.json'),
        input_identity_sha256=checkpoint_identity(list(rows.values()), hashes),
        roles={k: sorted(v) for k, v in roles.items()}, split_integrity=audit,
        test_used_for_fitting_or_selection=False,
        development_prefix_requests=len(dev), development_prefix_labels=len(labels),
        frozen_training_artifacts_verified=len(frozen['files']),
        source_full_test_verified_as_historical_only=True,
        exported_test_predictions=False, base_model_weights_included=False)
    output.mkdir(parents=True)
    for name in ('directions.json', 'calibration.json'):
        (output / name).write_bytes((experiment / name).read_bytes())
    (output / 'origin.json').write_text(json.dumps(origin, indent=2) + '\n', encoding='utf-8', newline='\n')
    files = {p.name: runner.sha(p) for p in sorted(output.iterdir())}
    (output / 'manifest.json').write_text(json.dumps(dict(
        format='msrc-multilayer-supplied-state-v1', files=files,
        contains_test_predictions=False, contains_base_model_weights=False), indent=2) + '\n',
        encoding='utf-8', newline='\n')
    print(json.dumps(dict(exported=str(output), files=files,
                          frozen_training_artifacts_verified=len(frozen['files']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.experiment, args.snapshot, args.out)
