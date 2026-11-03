"""Development-prefix validation of larger single-GPU answer-judge batches."""
import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

from pipeline import common
from pipeline.safety_judge_speed import SUITES, compare_judgments


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def owned(path):
    path = Path(path).resolve(strict=True)
    if Path.home() == Path('/home/HLC'):
        if not path.is_relative_to(Path('/mnt/data/HLC')) or path.stat().st_uid != os.getuid():
            raise ValueError('expected an HLC-owned read path')
    return path


def select_development(experiment, input_path, per_suite=16):
    from extensions.multilayer.runner import validate_prefixes

    if type(per_suite) is not int or per_suite < 2 or per_suite % 2:
        raise ValueError('positive even development requests per suite required')
    experiment, input_path = owned(experiment), owned(input_path)
    protocol, frozen = read(experiment / 'protocol.json'), read(experiment / 'frozen.json')
    roles = protocol['roles']
    if set(roles) != {'fit', 'calibration', 'test'} or frozen['test_used_for_selection'] is not False:
        raise ValueError('recorded non-test fitting protocol required')
    sets = {key: set(value) for key, value in roles.items()}
    if any(len(sets[key]) != len(roles[key]) for key in sets) or any(
            sets[a] & sets[b] for a, b in [('fit', 'calibration'), ('fit', 'test'), ('calibration', 'test')]):
        raise ValueError('role identities overlap or repeat')
    inputs = {row['id']: row for row in common.read_rows(input_path)}
    if set(inputs) != set.union(*sets.values()):
        raise ValueError('input IDs differ from recorded partition')
    selected = []
    for suite in SUITES:
        for role in ('fit', 'calibration'):
            part = [dict(inputs[rid], role=role) for rid in sets[role] if inputs[rid]['suite'] == suite]
            part.sort(key=lambda row: hashlib.sha256(row['id'].encode()).hexdigest())
            if len(part) < per_suite // 2:
                raise ValueError('insufficient development stratum')
            selected.extend(part[:per_suite // 2])
    source_hashes = {str(path): common.sha(path) for path in
        [input_path, experiment / 'protocol.json', experiment / 'frozen.json',
         experiment / 'calibration.json', experiment / 'directions.json']}
    for name in ('protocol.json', 'calibration.json', 'directions.json'):
        if source_hashes[str(experiment / name)] != frozen['files'][name]:
            raise ValueError('frozen source changed: ' + name)
    records, labels = [], []
    for row in selected:
        key = hashlib.sha256(row['id'].encode()).hexdigest() + '.json'
        for stage in ('development', 'labels'):
            path = owned(experiment / 'records' / stage / key)
            digest = common.sha(path)
            if digest != frozen['files'][path.relative_to(experiment).as_posix()]:
                raise ValueError('frozen observed development data changed')
            source_hashes[str(path)] = digest
            value = read(path)
            if value['id'] != row['id']:
                raise ValueError('development record identity changed')
            if stage == 'development':
                records.append(value)
            else:
                labels.extend(value['labels'])
    validate_prefixes(records, selected, labels, complete=True)
    return selected, records, labels, inputs, source_hashes


def build_cohort(tokenizer, selected, records, legacy):
    if (not legacy.get('prompts') or len(legacy['prompts']) != len(legacy.get('origins', []))
            or any(not isinstance(prompt, str) for prompt in legacy['prompts'])
            or any(row.get('role') not in ('fit', 'calibration') for row in legacy['origins'])):
        raise ValueError('complete development-only legacy diagnostic cohort required')
    sources = {row['id']: row for row in selected}
    unique, origins = {}, []

    def add(prompt, origin):
        if prompt not in unique:
            unique[prompt] = len(unique)
            origins.append([])
        origins[unique[prompt]].append(origin)

    for record in records:
        row = sources[record['id']]
        for prefix in record['prefixes']:
            prompt = tokenizer.apply_chat_template([dict(role='user', content=row['question']),
                dict(role='assistant', content=prefix['text'])], tokenize=False)
            add(prompt, dict(section='actual_development_prefix', id=row['id'], role=row['role'],
                suite=row['suite'], position=prefix['position'], prefix_sha256=prefix['prefix_sha256']))
    for prompt, origin in zip(legacy['prompts'], legacy['origins']):
        add(prompt, dict(origin, section='legacy_diagnostic_sentinel'))
    prompts = list(unique)
    return dict(prompts=prompts, origins=origins,
        prompt_sha256=[hashlib.sha256(value.encode()).hexdigest() for value in prompts],
        actual_development_requests=len(selected),
        actual_development_prefixes=sum(len(row['prefixes']) for row in records),
        unique_prompts=len(prompts), new_vlm_generations=False, fresh_judge_generation=True)


def report_dependency_audit(experiment, inputs):
    """Exercise the real report function; mutations are NOT model predictions."""
    from extensions.multilayer.runner import Experiment
    from multilayer_experiment import ARMS, judgment_key
    from safety_lab.prefix_data import parse_response_safety

    experiment = owned(experiment)
    original = read(experiment / 'report.json')
    tests = [read(path) for path in sorted((experiment / 'records/test').glob('*.json'))]
    answers = [read(path) for path in sorted((experiment / 'records/answers').glob('*.json'))]
    ids = {row['id'] for row in tests}
    if len(ids) != 3360 or len(tests) != 3360 or len(answers) != 3360:
        raise ValueError('complete historical fixed-outcome coverage required')
    calibration = read(experiment / 'calibration.json')
    frozen = read(experiment / 'frozen.json')

    class ReadOnlyReport:
        def __init__(self, reviews):
            self.tests = {rid: inputs[rid] for rid in ids}
            self.reviews = reviews

        def verify_frozen(self):
            return frozen

        def records(self, stage):
            return {'test': tests, 'answers': self.reviews}[stage]

        def read(self, name):
            if name != 'calibration.json':
                raise ValueError('unexpected report read')
            return calibration

        def save(self, name, value):
            if name != 'report.json':
                raise ValueError('unexpected report write')
            self.saved = copy.deepcopy(value)

        def status(self, *args, **kwargs):
            pass

    baseline = Experiment.report(ReadOnlyReport(answers))
    if baseline != original:
        raise ValueError('current frozen report function differs from historical report')
    scenarios = []
    for status in ('Safe', 'Unsafe', 'Controversial'):
        raw = 'Safety: ' + status
        reviews = []
        for commit in tests:
            rid = commit['id']
            reviews.append(dict(id=rid, labels=[dict(arm=arm['arm'],
                input_sha256=judgment_key(inputs[rid]['question'], arm['answer']),
                judge_raw=raw, **parse_response_safety(raw)) for arm in commit['arms']]))
        result = Experiment.report(ReadOnlyReport(reviews))
        scenarios.append(dict(synthetic_dependency_test=True, forced_review_status=status,
            reviewed_outputs=result['full_answer_reviews'],
            primary_unchanged=result['primary'] == baseline['primary'],
            qualification_flags_unchanged=result['meets_75_percent'] == baseline['meets_75_percent'],
            response_safety_changed=result['response_safety'] != baseline['response_safety']))
    if not all(row['primary_unchanged'] and row['qualification_flags_unchanged'] for row in scenarios):
        raise ValueError('posthoc reviews unexpectedly affect primary warning metric')
    return dict(historical_fixed_test_requests=3360, historical_arm_outcomes=3360 * len(ARMS),
        historical_report_reproduced_by_current_report_function=True,
        synthetic_mutations_not_new_predictions=True, fresh_gpu_acceptance=False,
        primary_dependency_preserved=True, cases=scenarios,
        training_label_substitution_safe=False,
        training_boundary='fit excludes unknown prefix labels; calibration aggregates prefix labels',
        report_function_source_sha256=common.sha(common.ROOT / 'extensions/multilayer/frozen/multilayer_experiment.py'))


def comparison_details(reference, actual, origins):
    from extensions.multilayer.runner import parse_response_safety

    result = compare_judgments(reference, actual, parse_response_safety)
    result['differences'] = [dict(index=i, origins=origins[i], reference_raw=a, actual_raw=b,
        reference_parsed=parse_response_safety(a), actual_parsed=parse_response_safety(b))
        for i, (a, b) in enumerate(zip(reference, actual)) if a != b]
    result['status_transition_counts'] = dict(Counter(
        parse_response_safety(a)['status'] + '->' + parse_response_safety(b)['status']
        for a, b in zip(reference, actual)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--legacy-cohort', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--per-suite', type=int, default=16)
    parser.add_argument('--cap-gib', type=int, default=28)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id) or not 20 <= args.cap_gib <= 28:
        parser.error('simple run ID and bounded 20-28 GiB allocator required')
    common.initialize()
    os.environ['MSRC_GPU'] = '0'
    common.select_device()
    for path in (common.ROOT, args.experiment, args.inputs, args.legacy_cohort, args.model):
        owned(path)
    output = common.inside(common.runs_dir() / args.run_id / 'judge_validation')
    output.mkdir(parents=True, exist_ok=False)
    selected, records, labels, inputs, source_hashes = select_development(
        args.experiment, args.inputs, args.per_suite)
    source_hashes[str(args.legacy_cohort.resolve())] = common.sha(args.legacy_cohort)
    checkpoint = read(common.ROOT / 'assets/plugguard/base_models.json')['judge']
    protocol = dict(development_only_gpu_selection=True, test_used_for_fitting_or_batch_selection=False,
        source_hashes=source_hashes, selected=[{k: row[k] for k in ('id', 'role', 'suite')} for row in selected],
        historical_test_used_only_for_report_function_dependency_test=True,
        warning_model_or_thresholds_changed=False, input_truncation=False,
        dtype='bfloat16', attention='sdpa', max_new_tokens=128, do_sample=False,
        model_checkpoint=checkpoint, seed=common.SEED,
        variants=[['original_4', 4, False], ['original_6', 6, False],
            ['bucket_8', 8, True], ['bucket_16', 16, True], ['bucket_32', 32, True],
            ['bucket_64', 64, True], ['bucket_128', 128, True], ['bucket_256', 256, True]],
        rounds=2, opposite_round_order=True, cap_gib=args.cap_gib, shared_compute_duty=.5,
        full_raw_and_parsed_match_required_for_training_or_exact_dropin=True,
        minimum_paired_speedup=1.05, oom_means_failed_variant_not_smaller_batch=True,
        automatic_activation=False, fresh_full_warning_acceptance=False,
        source_code_sha256={name: common.sha(common.ROOT / name) for name in
            ['pipeline/safety_judge_validation.py', 'pipeline/batch_judge.py', 'msrc/judge_scheduling.py',
             'extensions/multilayer/frozen/multilayer_experiment.py']})
    common.save(output / 'protocol.json', protocol)
    dependency = report_dependency_audit(args.experiment, inputs)
    common.save(output / 'report_dependency.json', dependency)
    print('REPORT_DEPENDENCY_AUDIT_OK 3360 16800', flush=True)
    for name, digest in checkpoint['files'].items():
        if common.sha(args.model / name) != digest:
            raise ValueError('official checkpoint differs: ' + name)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pipeline.batch_training import BatchBudget
    from pipeline.batch_judge import judge_texts
    from msrc.judge_scheduling import encoded_judge_texts, padded_slots
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(common.SEED)
    budget = BatchBudget(torch, args.cap_gib, shared_duty=.5)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    legacy = read(args.legacy_cohort)
    roles = read(args.experiment / 'protocol.json')['roles']
    for origin in legacy['origins']:
        if origin['role'] not in ('fit', 'calibration') or origin['id'] not in set(roles[origin['role']]):
            raise ValueError('legacy sentinel is not in the recorded development role')
    cohort = build_cohort(tokenizer, selected, records, legacy)
    prompts, origins = cohort['prompts'], cohort['origins']
    unpadded = tokenizer(prompts, padding=False, truncation=False)
    lengths = [len(row) for row in unpadded['input_ids']]
    cohort['input_lengths'] = lengths
    common.save(output / 'cohort.json', cohort)
    print('COHORT_READY ' + json.dumps({key: cohort[key] for key in
        ['actual_development_requests', 'actual_development_prefixes', 'unique_prompts']}), flush=True)
    tensor_checks = []
    for name, size, ordered in protocol['variants']:
        order = sorted(range(len(prompts)), key=lambda i: (lengths[i], i)) if ordered else list(range(len(prompts)))
        for start in range(0, len(order), size):
            indices = order[start:start + size]
            direct = tokenizer([prompts[i] for i in indices], padding=True, truncation=False, return_tensors='pt')
            cached = tokenizer.pad([{k: v[i] for k, v in unpadded.items()} for i in indices],
                                   padding=True, return_tensors='pt')
            if set(direct) != set(cached) or not all(torch.equal(direct[k], cached[k]) for k in direct):
                raise ValueError('input tensors differ before GPU inference')
        tensor_checks.append(dict(variant=name, requests=len(prompts), all_input_tensors_equal=True))
    common.save(output / 'input_tensor_audit.json', tensor_checks)
    clock = time.perf_counter()
    with budget.work('load_official_judge'):
        model = AutoModelForCausalLM.from_pretrained(str(args.model), dtype=torch.bfloat16,
            device_map='cuda:0', attn_implementation='sdpa', local_files_only=True).eval()
        model.requires_grad_(False)
    load_seconds = time.perf_counter() - clock
    judge_texts(model, tokenizer, prompts[:2], batch_size=2, budget=budget)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()

    def measure(name, size, ordered):
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stats = {}
        started, work, idle = time.perf_counter(), budget.work_seconds, budget.idle_seconds
        model._msrc_judge_batches = []
        if name.startswith('original') or name == 'serial':
            values = judge_texts(model, tokenizer, prompts, batch_size=size, budget=budget)
            if model._msrc_judge_batches != [min(size, len(prompts) - i) for i in range(0, len(prompts), size)]:
                raise ValueError('baseline silently changed its batch size')
            stats['padded_input_slots'] = padded_slots(lengths, size)
            stats['actual_max_batch_size'] = max(model._msrc_judge_batches)
            stats['batch_count'] = len(model._msrc_judge_batches)
        else:
            values = encoded_judge_texts(model, tokenizer, prompts, batch_size=size,
                order_by_length=ordered, budget=budget, statistics=stats)
            stats['actual_max_batch_size'] = max(row['size'] for row in stats['batches'])
            stats['batch_count'] = len(stats['batches'])
            stats.pop('batches', None)
        torch.cuda.synchronize()
        return values, dict(variant=name, batch_size=size, seconds=time.perf_counter() - started,
            work_seconds=budget.work_seconds - work, voluntary_idle_seconds=budget.idle_seconds - idle,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            unchanged_rng=torch.equal(cpu_rng, torch.get_rng_state()) and
                torch.equal(cuda_rng, torch.cuda.get_rng_state()), **stats)

    print('SERIAL_REFERENCE_START ' + str(len(prompts)), flush=True)
    serial, serial_time = measure('serial', 1, False)
    common.save(output / 'serial_reference.json', dict(outputs=serial, measurement=serial_time))
    print('SERIAL_REFERENCE_COMPLETE ' + json.dumps(serial_time), flush=True)
    measurements, outputs, failed = [], {}, set()
    for repeat in (0, 1):
        variants = protocol['variants'] if repeat == 0 else list(reversed(protocol['variants']))
        for name, size, ordered in variants:
            if name in failed:
                continue
            started = time.perf_counter()
            try:
                values, item = measure(name, size, ordered)
                item.update(repeat=repeat, comparison_to_serial=comparison_details(serial, values, origins))
                outputs[name + '_' + str(repeat)] = values
            except torch.OutOfMemoryError as error:
                message = str(error)[:800]
                error.__traceback__ = None
                gc.collect()
                torch.cuda.empty_cache()
                failed.add(name)
                item = dict(variant=name, batch_size=size, repeat=repeat, failure='OOM',
                    seconds_until_failure=time.perf_counter() - started, error=message)
            measurements.append(item)
            common.save(output / 'measurements.json', measurements)
            common.save(output / 'fresh_outputs.json', outputs)
            brief = {k: v for k, v in item.items() if k != 'comparison_to_serial'}
            if 'comparison_to_serial' in item:
                brief['comparison_to_serial'] = {k: v for k, v in item['comparison_to_serial'].items()
                                                if k != 'differences'}
            print('VALIDATION_TRIAL ' + json.dumps(brief), flush=True)
    comparisons = []
    for name, size, ordered in protocol['variants'][2:]:
        rows = [row for row in measurements if row['variant'] == name and 'seconds' in row]
        entry = dict(variant=name, batch_size=size, complete_two_rounds=len(rows) == 2,
            comparisons_to_baseline={}, automatic_activation=False)
        for reference_name in ('original_4', 'original_6'):
            speeds, checks = [], []
            for row in rows:
                repeat = row['repeat']
                reference_row = next(x for x in measurements if x['variant'] == reference_name and x['repeat'] == repeat)
                speeds.append(reference_row['seconds'] / row['seconds'])
                checks.append(comparison_details(outputs[reference_name + '_' + str(repeat)],
                    outputs[name + '_' + str(repeat)], origins))
            entry['comparisons_to_baseline'][reference_name] = dict(paired_speedups=speeds, rounds=checks)
        entry['exact_training_or_output_equivalent_dropin'] = (len(rows) == 2 and
            all(row['unchanged_rng'] and row['comparison_to_serial']['exact_raw'] and
                row['comparison_to_serial']['exact_parsed'] for row in rows) and
            all(all(c['exact_raw'] and c['exact_parsed'] for c in value['rounds'])
                for value in entry['comparisons_to_baseline'].values()))
        comparisons.append(entry)
    for path, digest in source_hashes.items():
        if common.sha(path) != digest:
            raise ValueError('read-only source changed during validation: ' + path)
    result = dict(verdict='DEVELOPMENT_VALIDATION_NOT_FULL_WARNING_ACCEPTANCE', complete=True,
        comparisons=comparisons, dependency_audit=dependency,
        unique_prompts=len(prompts), actual_development_requests=len(selected),
        actual_observed_prefixes=cohort['actual_development_prefixes'],
        model_load_seconds=load_seconds, serial_measurement=serial_time,
        measurements=measurements, resource=budget.snapshot(), gpu=torch.cuda.get_device_name(0),
        cloud_or_formal_jobs_changed=False, base_weights_or_thresholds_changed=False,
        training_labels_replaced=False, primary_warning_inference_repeated=False)
    common.save(output / 'report.json', result)
    lines = ['# Larger single-GPU judge batches: validation', '',
        'Real recorded development prefixes plus retained diagnostic sentinels; fresh official judge inference.',
        'The report dependency audit uses historical fixed test outcomes and synthetic review mutations only.',
        'No fresh full warning-accuracy result is claimed. No training labels or formal outputs were replaced.', '',
        '| Variant | Round | Batch | Seconds | Raw matches serial | Parsed matches serial | Peak allocated GiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in measurements:
        if 'seconds' in row:
            check = row['comparison_to_serial']
            lines.append('| %s | %d | %d | %.3f | %d/%d | %d/%d | %.2f |' %
                (row['variant'], row['repeat'], row['batch_size'], row['seconds'],
                 check['raw_matches'], len(prompts), check['parsed_matches'], len(prompts),
                 row['peak_allocated_bytes'] / 1024**3))
        else:
            lines.append('| %s | %d | %d | OOM at bounded cap | - | - | - |' %
                (row['variant'], row['repeat'], row['batch_size']))
    lines += ['', 'Larger batches are not automatically adopted. Read report.json for both batch-4 and batch-6 comparisons.',
        'Posthoc teacher judgments do not enter the primary warning metric for fixed outcomes, but prefix training labels do affect fitting.',
        'Shared A100 timings include 50% voluntary yielding and must not be presented as exclusive RTX 5090 timings.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(result['verdict'], flush=True)


if __name__ == '__main__':
    main()
