"""Recover development-only GPU resource failures without changing the method.

The failed directory is read-only. All completed development evidence is copied
and verified in a fresh continuation directory; head fitting, threshold
selection, test inference and answer judging are always executed again.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from pipeline import plugguard as pg
from pipeline.acceptance_check import safety_check
from pipeline.common import read_rows
from pipeline.safety import prepare


RESOURCE_FAILURE = 'Shared GPU contains an unauthorized process owner'
FORBIDDEN = ('head.pt', 'threshold.json', 'calibration_scores.jsonl',
             'test_delivery.jsonl', 'answer_judgments.jsonl', 'primary_metric.json',
             'report.json')
ORIGIN_FILES = ('inputs.jsonl', 'input_hashes.json', 'protocol.json',
                'dataset_identity.json', 'environment.json', 'execution_claim.json',
                'status.json', 'inference_source.py', 'development_generation.jsonl')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record_sha(row):
    text = json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def safe_read(path, root):
    path, root = Path(path), Path(root).resolve(strict=True)
    resolved = path.resolve(strict=True)
    owned = not hasattr(os, 'getuid') or resolved.stat().st_uid == os.getuid()
    require(resolved.is_relative_to(root) and owned,
            'Recovery source escapes its owned directory: ' + str(path))
    require(not path.is_symlink(), 'Recovery evidence cannot be a symlink: ' + str(path))
    return resolved


def validate_source(source, current_source):
    source = pg.owned_read(source)
    require(source.is_dir(), 'Recovery source must be a failed run directory')
    for name in FORBIDDEN:
        require(not (source / name).exists() and not (source / name).is_symlink(),
                'Recovery only accepts development before head/threshold/test: ' + name)
    for name in ORIGIN_FILES:
        safe_read(source / name, source)
    state, claim = load(source / 'status.json'), load(source / 'execution_claim.json')
    require(state.get('state') == 'FAILED' and RESOURCE_FAILURE in state.get('error', ''),
            'Only the disclosed foreign-owner GPU resource failure can be recovered')
    require(state.get('pid') == claim.get('pid') and type(state.get('failed_ns')) is int,
            'Failed worker identity or timestamp differs')
    boot = Path('/proc/sys/kernel/random/boot_id')
    if boot.exists() and boot.read_text().strip() == claim.get('boot_id'):
        found = subprocess.check_output(['ps', '-p', str(claim['pid']), '-o', 'pid=,args='],
                                        text=True).strip() if Path('/proc/%s' % claim['pid']).exists() else ''
        require(not found, 'Failed worker PID is still present; reconcile it before recovery')
    protocol = load(source / 'protocol.json')
    require(protocol.get('backend') == 'plugguard'
            and protocol.get('execution_kind') == 'fresh_inference'
            and protocol.get('training_mode') == 'development_refit',
            'Archived or loaded-head runs are not development recovery sources')
    require(pg.sha(source / 'inference_source.py') == protocol['source_sha256']
            == pg.sha(current_source), 'The frozen inference code must be byte-identical')
    require(pg.sha(source / 'inputs.jsonl') == protocol['source_inputs_sha256']
            and pg.sha(source / 'input_hashes.json') == protocol['input_hashes_sha256'],
            'Source input or image manifest changed')
    generation = read_rows(source / 'development_generation.jsonl')
    inputs = read_rows(source / 'inputs.jsonl')
    expected = [r for r in inputs if r['split'] == 'dev']
    require(0 < len(generation) <= len(expected) == 3360, 'Invalid development recovery count')
    require([r['id'] for r in generation] == [r['id'] for r in expected[:len(generation)]],
            'Only an ordered development prefix can be recovered; no held-out IDs')
    for record, original in zip(generation, expected):
        require(all(record[k] == original[k] for k in ('id', 'label', 'group', 'suite', 'split')),
                'Development record identity or label differs')
        require(record['threshold'] is None and record['warn'] is False and not record['events']
                and record['max_score'] == 0 and record['trigger_token'] is None,
                'Only unscored development generation may be recovered')
        require(type(record['generated_tokens']) is int and 0 < record['generated_tokens'] <= 128
                and record['released_tokens'] == record['generated_tokens']
                and protocol['created_ns'] <= record['start_ns'] <= record['end_ns'] <= state['failed_ns'],
                'Incomplete development response or invalid origin timestamps')
    return protocol, generation, state


def copy_features(source, records, input_hash, loader):
    features = []
    for i, row in enumerate(records, 1):
        name = hashlib.sha256(row['id'].encode()).hexdigest() + '.pt'
        path = safe_read(source / 'features' / name, source)
        feat = loader(path)
        require(feat['id'] == row['id'] and feat['label'] == row['label']
                and feat['input_sha256'] == input_hash, 'Cached feature identity differs')
        prefix, response = feat['prefix'], feat['response']
        require(len(prefix.shape) == len(response.shape) == 3
                and prefix.shape[0] == response.shape[0] == 1
                and prefix.shape[-1] == response.shape[-1] == 3584
                and prefix.shape[1] > 0 and response.shape[1] == row['generated_tokens'],
                'Incomplete or differently shaped feature tensors')
        del feat, prefix, response
        digest = pg.sha(path)
        target = pg.destination(pg.ROOT / 'features' / name)
        require(not target.exists(), 'Refusing to overwrite a recovery feature')
        shutil.copyfile(path, target)
        require(pg.sha(target) == digest, 'Feature copy digest differs')
        features.append(dict(id=row['id'], filename=name, sha256=digest,
                             bytes=target.stat().st_size, record_sha256=record_sha(row)))
        pg.append('development_generation.jsonl', row)
        if i % 100 == 0 or i == len(records):
            pg.status('VERIFY_AND_COPY_DEVELOPMENT_CACHE', completed=i, total=len(records))
    return features


def load_feature(path):
    feat = pg.torch.load(path, map_location='cpu', weights_only=True)
    require(all(feat[key].dtype == pg.torch.bfloat16 and bool(pg.torch.isfinite(feat[key]).all())
                for key in ('prefix', 'response')), 'Non-finite or differently typed feature tensors')
    return feat


def audit_recovery(root):
    protocol = load(root / 'protocol.json')
    require(pg.sha(root / 'development_recovery.json') == protocol['development_recovery_sha256'],
            'Development recovery manifest changed')
    manifest = load(root / 'development_recovery.json')
    require(pg.sha(root / 'recovery_source.py') == manifest['recovery_source_sha256']
            == pg.sha(__file__), 'Recovery implementation changed')
    origin = root / 'recovery_origin'
    for name, digest in manifest['origin_files_sha256'].items():
        require(pg.sha(origin / name) == digest, 'Frozen recovery origin changed: ' + name)
    old = load(origin / 'protocol.json')
    require(old['source_sha256'] == protocol['source_sha256']
            and old['source_inputs_sha256'] == protocol['source_inputs_sha256']
            and old['input_hashes_sha256'] == protocol['input_hashes_sha256'],
            'Recovered and new evidence use different methods or inputs')
    original = read_rows(origin / 'development_generation.jsonl')
    current = read_rows(root / 'development_generation.jsonl')
    n = manifest['recovered_development_rows']
    require(n == len(original) == len(manifest['features']) and current[:n] == original,
            'Recovered development prefix was modified')
    for feature, row in zip(manifest['features'], original):
        require(feature['id'] == row['id'] and feature['record_sha256'] == record_sha(row)
                and pg.sha(root / 'features' / feature['filename']) == feature['sha256'],
                'Recovered feature/record bytes changed')
    require(all(r['start_ns'] >= protocol['created_ns'] for r in current[n:]),
            'New development rows predate this continuation')
    return dict(recovered_development_rows=n, newly_generated_development_rows=len(current)-n,
                recovered_head=False, recovered_threshold=False, recovered_test_scores=False,
                failed_source_run=manifest['failed_source_run'],
                interrupted_experiment_not_an_uninterrupted_fresh_attempt=True)


def recover(source, out):
    source, out = Path(source).resolve(strict=True), pg.inside(out)
    require(not out.exists() and not out.is_symlink(), 'Fresh recovery output required')
    require(not out.is_relative_to(source) and not source.is_relative_to(out),
            'Source and continuation must be separate directories')
    origin_protocol, records, failure = validate_source(source, pg.__file__)
    require(shutil.disk_usage(out.parent.parent).free > 250 * 1024**3,
            'Less than 250 GiB free before recovery')
    pg.ROOT = out
    pg.initialize()
    out.mkdir(parents=True)
    try:
        prepare(out, backend='plugguard')
        shutil.copyfile(out / 'protocol.json', out / 'preprocessing_protocol.json')
        require(pg.sha(out / 'inputs.jsonl') == origin_protocol['source_inputs_sha256']
                and pg.sha(out / 'input_hashes.json') == origin_protocol['input_hashes_sha256'],
                'Re-prepared inputs differ; recovery cannot reuse these features')
        pg.setup()
        with pg.destination(out / 'execution_claim.json').open('x') as handle:
            json.dump(dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                           start_ns=time.time_ns(), gpu=os.environ['CUDA_VISIBLE_DEVICES']), handle)
        rows = pg.dataset()
        require(load(source / 'dataset_identity.json')['roles'] == load(out / 'dataset_identity.json')['roles'],
                'Recovery role assignments differ')
        require(load(source / 'environment.json')['verified_model_identities']
                == load(out / 'environment.json')['verified_model_identities'], 'Base model identities differ')
        require(origin_protocol['native_generator_config_sha256'] == pg.sha(pg.MODEL / 'generation_config.json')
                and origin_protocol['split_source_sha256'] == pg.source_sha256()
                and origin_protocol['shared_gpu_policy_sha256'] == pg.sha(out / 'shared_gpu_policy.json'),
                'Frozen generation, split, or GPU sharing policy differs')
        copied = out / 'recovery_origin'
        copied.mkdir()
        origin_hashes = {}
        for name in ORIGIN_FILES:
            path = safe_read(source / name, source)
            origin_hashes[name] = pg.sha(path)
            shutil.copyfile(path, pg.destination(copied / name))
            require(pg.sha(copied / name) == origin_hashes[name], 'Origin copy changed')
        feature_manifest = copy_features(source, records, origin_protocol['source_inputs_sha256'], load_feature)
        manifest = dict(failed_source_run=str(source), failure_ns=failure['failed_ns'],
                        recovery_source_sha256=pg.sha(__file__), origin_files_sha256=origin_hashes,
                        recovered_development_rows=len(records), features=feature_manifest,
                        recovered_head=False, recovered_threshold=False, recovered_test_scores=False)
        pg.save('development_recovery.json', manifest)
        protocol = dict(origin_protocol, created_ns=time.time_ns(),
                        preprocessing_protocol_sha256=pg.sha(out / 'preprocessing_protocol.json'),
                        development_recovery_sha256=pg.sha(out / 'development_recovery.json'),
                        continuation_of_same_experiment=True,
                        development_generated_this_run=True,
                        development_generation_scope='Full current experiment, including verified pre-interruption prefix',
                        recovered_development_rows=len(records), newly_generated_development_rows=3360-len(records))
        pg.save('protocol.json', protocol)
        shutil.copyfile(pg.__file__, pg.destination(out / 'inference_source.py'))
        shutil.copyfile(__file__, pg.destination(out / 'recovery_source.py'))
        with pg.budget.work('causality_smoke'):
            pg.causal_test()
        dev = [r for r in rows if r['split'] == 'dev']
        pg.extract(dev, next(r for r in dev if r['label'] == 0))
        head = pg.fit(rows)
        threshold = pg.calibrate(head, rows)
        delivered = pg.evaluate(head, threshold, rows)
        del head
        gc.collect()
        pg.torch.cuda.empty_cache()
        pg.judge_answers(delivered, rows)
        require(pg.sha(out / 'inputs.jsonl') == pg.INPUT_HASH, 'Input manifest changed')
        pg.save('runtime_statistics.json', dict(
            peak_allocated_bytes=pg.torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=pg.torch.cuda.max_memory_reserved(),
            head_latency_ms_per_scored_token=sum(r['head_ns'] for r in delivered)
                / max(1, sum(r['generated_tokens'] for r in delivered)) / 1e6,
            latency_includes_prefix_initialization_and_synchronization=True,
            latency_excludes_voluntary_sharing_idle=True,
            all_test_rows_present=True, test_used_for_selection=False, resource_control=pg.budget.snapshot()))
        recovery = audit_recovery(out)
        pg.save('recovery_audit.json', recovery)
        report = pg.write_report(out)
        report['development_recovery'] = recovery
        report['newly_inferred_in_this_continuation'] = 3360 + recovery['newly_generated_development_rows']
        report['n_inferred_scope'] = 'Whole experiment including the verified development prefix before interruption'
        pg.save('report.json', report)
        safety_check(out, report)
        pg.save('recovery_result.json', dict(primary=report['primary'], recovery=recovery,
            milestone_75_percent=report['primary']['accuracy'] >= .75,
            completion_85_percent=report['primary']['accuracy'] >= .85,
            independent_acceptance_check_passed=True))
        pg.save('status.json', dict(state='COMPLETE', stage='RECOVERY_REPORT_AND_AUDIT',
                                   pid=os.getpid(), completed_ns=time.time_ns()))
        print('RECOVERED_FULL_SAFETY_ACCEPTANCE_CHECK_OK', flush=True)
    except BaseException as exc:
        pg.save('status.json', dict(state='FAILED', error=str(exc), traceback=traceback.format_exc(),
                                   pid=os.getpid(), failed_ns=time.time_ns()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    recover(args.source, args.out)


if __name__ == '__main__':
    main()
