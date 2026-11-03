"""Development-only, fresh GPU comparisons of single-card judge scheduling."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from pipeline import common

SUITES = ('SD', 'SD_TYPO', 'TYPO', 'benign')


def development_records(source, per_suite=4):
    identity = json.loads((source / 'dataset_identity.json').read_text(encoding='utf-8'))
    roles = identity['roles']
    if set(roles) != {'fit', 'calibration', 'test'}:
        raise ValueError('recorded fit/calibration/test roles required')
    dev = set(roles['fit']) | set(roles['calibration'])
    if (len(dev) != len(roles['fit']) + len(roles['calibration'])
            or dev & set(roles['test'])):
        raise ValueError('development/test roles overlap')
    inputs = {row['id']: row for row in common.read_rows(source / 'inputs.jsonl') if row['id'] in dev}
    selected = []
    for suite in SUITES:
        for role in ('fit', 'calibration'):
            part = [row for row in inputs.values() if row['id'] in set(roles[role]) and row['suite'] == suite]
            part.sort(key=lambda row: hashlib.sha256(row['id'].encode()).hexdigest())
            count = per_suite // 2
            if len(part) < count:
                raise ValueError('insufficient development stratum')
            selected.extend(dict(row, role=role) for row in part[:count])
    generated = {}
    wanted = {row['id'] for row in selected}
    with (source / 'development_generation.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['id'] in wanted:
                if row['id'] in generated or row.get('split') != 'dev' or not isinstance(row.get('answer'), str):
                    raise ValueError('duplicate or non-development observed answer')
                generated[row['id']] = row
    if set(generated) != wanted:
        raise ValueError('missing previously observed development answer')
    return [dict(row, answer=generated[row['id']]['answer']) for row in selected]


def prompts_from_records(tokenizer, records):
    unique = {}
    for row in records:
        ids = tokenizer.encode(row['answer'], add_special_tokens=False)
        for position in (16, 64, None):
            answer = row['answer'] if position is None else tokenizer.decode(ids[:position], skip_special_tokens=True)
            prompt = tokenizer.apply_chat_template([dict(role='user', content=row['question']),
                dict(role='assistant', content=answer)], tokenize=False)
            unique.setdefault(prompt, dict(id=row['id'], suite=row['suite'], role=row['role'],
                text_prefix_tokens=position, observed_development_text=True))
    return list(unique), list(unique.values())


def compare_judgments(reference, actual, parse):
    if len(reference) != len(actual) or not reference:
        raise ValueError('complete corresponding judgment outputs required')
    left, right = [parse(value) for value in reference], [parse(value) for value in actual]
    return dict(raw_matches=sum(a == b for a, b in zip(reference, actual)),
        parsed_matches=sum(a == b for a, b in zip(left, right)),
        unparsed=sum(value['status'] == 'unparsed' for value in right),
        unknown=sum(value['label'] is None for value in right), count=len(actual),
        exact_raw=reference == actual, exact_parsed=left == right)


def candidate_admission(measurements):
    """Require the timed baseline and candidate to preserve the serial outputs."""
    variants = [('encoded', 4), ('length_bucket', 4), ('length_bucket_8', 8)]

    def complete_rows(name):
        rows = [item for item in measurements if item['variant'] == name]
        complete = (len(rows) == 2 and {row['repeat'] for row in rows} == {0, 1}
            and all(isinstance(row.get('seconds'), (int, float))
                and math.isfinite(row['seconds']) and row['seconds'] > 0 for row in rows))
        return {row['repeat']: row for row in rows}, complete

    def equivalent(row):
        return (row.get('comparison', {}).get('exact_raw') is True
            and row.get('comparison', {}).get('exact_parsed') is True
            and row.get('unchanged_rng') is True)

    baseline, baseline_complete = complete_rows('original')
    baseline_equivalent = baseline_complete and all(equivalent(row) for row in baseline.values())
    comparisons = []
    for name, size in variants:
        rows, complete = complete_rows(name)
        speeds = ([baseline[i]['seconds'] / rows[i]['seconds'] for i in (0, 1)]
            if baseline_complete and complete else [])
        candidate_equivalent = complete and all(equivalent(row) for row in rows.values())
        speed_gate = len(speeds) == 2 and all(speed >= 1.05 for speed in speeds)
        blockers = []
        if not baseline_complete or not complete:
            blockers.append('incomplete_paired_rounds')
        if baseline_complete and not baseline_equivalent:
            blockers.append('baseline_differs_from_serial_or_changes_rng')
        if complete and not candidate_equivalent:
            blockers.append('candidate_differs_from_serial_or_changes_rng')
        if not speed_gate:
            blockers.append('paired_speedup_below_5_percent_or_unavailable')
        comparisons.append(dict(variant=name, batch_size=size, paired_speedups=speeds,
            baseline_matches_serial=bool(baseline_equivalent),
            candidate_matches_serial=bool(candidate_equivalent),
            timing_gate_passed=bool(speed_gate),
            eligible_for_further_validation=bool(not blockers), blockers=blockers,
            automatically_activated=False))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cap-gib', type=int, default=22)
    parser.add_argument('--per-suite', type=int, default=4)
    args = parser.parse_args()
    if re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id) is None or args.per_suite < 2 or args.per_suite % 2:
        parser.error('simple run ID and positive even samples per suite required')
    common.initialize()
    os.environ['MSRC_GPU'] = args.gpu
    common.select_device()
    source, model_path = args.source.resolve(strict=True), args.model.resolve(strict=True)
    if Path.home() == Path('/home/HLC'):
        for path in (source, model_path, common.ROOT):
            if not path.is_relative_to(Path('/mnt/data/HLC')) or path.stat().st_uid != os.getuid():
                raise ValueError('HLC-owned source, checkpoint and isolated workspace required')
    output = common.inside(common.runs_dir() / args.run_id / 'judge_speed')
    output.mkdir(parents=True, exist_ok=False)
    selected = development_records(source, args.per_suite)
    source_files = ('inputs.jsonl', 'dataset_identity.json', 'development_generation.jsonl')
    frozen_source = {name: common.sha(source / name) for name in source_files}
    checkpoint = json.loads((common.ROOT / 'assets/plugguard/base_models.json').read_text())['judge']
    for name, digest in checkpoint['files'].items():
        if common.sha(model_path / name) != digest:
            raise ValueError('official judge checkpoint differs: ' + name)
    common.save(output / 'protocol.json', dict(development_only=True, test_used_for_selection=False,
        selected=[{key: row[key] for key in ('id', 'suite', 'role')} for row in selected],
        source=str(source), source_hashes=frozen_source, model=str(model_path), checkpoint=checkpoint,
        base_model_weights_frozen=True, no_fitting=True, quantization=False,
        max_new_tokens=128, do_sample=False, input_truncation=False,
        original_historical_answers_used_only_as_fixed_judge_inputs=True,
        fresh_judge_generation=True, cache_of_old_judge_answers=False,
        primary_warning_accuracy_retested=False, full_acceptance=False,
        shared_A100_not_a_5090_benchmark=True, model_loading_included_in_report_not_arm_timing=True,
        minimum_speedup_both_rounds=1.05, strict_raw_and_parsed_equivalence=True,
        execution_sources={name: common.sha(common.ROOT / name) for name in
            ('pipeline/safety_judge_speed.py', 'msrc/judge_scheduling.py', 'pipeline/batch_judge.py')}))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from extensions.multilayer.runner import parse_response_safety
    from msrc.judge_scheduling import encoded_judge_texts, padded_slots
    from pipeline.batch_judge import judge_texts
    from pipeline.batch_training import BatchBudget
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(common.SEED)
    budget = BatchBudget(torch, args.cap_gib, shared_duty=.5)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts, origins = prompts_from_records(tokenizer, selected)
    lengths = [len(value) for value in tokenizer(prompts, padding=False, truncation=False)['input_ids']]
    common.save(output / 'cohort.json', dict(prompts=prompts, origins=origins, lengths=lengths,
        prompt_sha256=[hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts],
        unique_prompts=len(prompts), original_generation_repeated=False))
    start = time.perf_counter()
    with budget.work('load_judge'):
        model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.bfloat16,
            device_map='cuda:0', attn_implementation='sdpa', local_files_only=True).eval()
        model.requires_grad_(False)
    model_load_seconds = time.perf_counter() - start
    judge_texts(model, tokenizer, prompts[:2], batch_size=2, budget=budget)
    initial_cpu, initial_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()

    def reset_rng():
        torch.set_rng_state(initial_cpu)
        torch.cuda.set_rng_state(initial_cuda)

    def measure(name, size, order=False):
        reset_rng()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stats = {}
        before_work, before_idle = budget.work_seconds, budget.idle_seconds
        clock = time.perf_counter()
        if name in ('serial', 'original'):
            values = judge_texts(model, tokenizer, prompts, batch_size=size, budget=budget)
            stats['padded_input_slots'] = padded_slots(lengths, size)
        else:
            values = encoded_judge_texts(model, tokenizer, prompts, batch_size=size,
                order_by_length=order, budget=budget, statistics=stats)
        torch.cuda.synchronize()
        item = dict(variant=name, batch_size=size, seconds=time.perf_counter() - clock,
            prompts=len(prompts), work_seconds=budget.work_seconds - before_work,
            voluntary_idle_seconds=budget.idle_seconds - before_idle,
            unchanged_rng=torch.equal(initial_cpu, torch.get_rng_state()) and
                torch.equal(initial_cuda, torch.cuda.get_rng_state()),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), **stats)
        return values, item

    reference, reference_measurement = measure('serial', 1)
    common.save(output / 'serial_reference.json', dict(outputs=reference, measurement=reference_measurement))
    variants = [('original', 4, False), ('encoded', 4, False),
                ('length_bucket', 4, True), ('length_bucket_8', 8, True)]
    measurements, all_outputs = [], {}
    failed = set()
    for repeat in (0, 1):
        for name, size, order in (variants if repeat == 0 else list(reversed(variants))):
            if name in failed:
                continue
            try:
                values, item = measure(name, size, order)
                item.update(repeat=repeat, comparison=compare_judgments(reference, values, parse_response_safety))
                all_outputs[name + '_' + str(repeat)] = values
            except torch.OutOfMemoryError as error:
                gc.collect()
                torch.cuda.empty_cache()
                failed.add(name)
                item = dict(variant=name, repeat=repeat, batch_size=size, failure='OOM', error=str(error)[:600])
            measurements.append(item)
            common.save(output / 'measurements.json', measurements)
            common.save(output / 'fresh_outputs.json', all_outputs)
            print('SAFETY_JUDGE_TRIAL ' + json.dumps(item), flush=True)
    comparisons = candidate_admission(measurements)
    if any(common.sha(source / name) != digest for name, digest in frozen_source.items()):
        raise ValueError('read-only historical input changed during the trial')
    result = dict(verdict='DEVELOPMENT_JUDGE_SCHEDULING_TRIAL_NOT_ACCEPTANCE',
        complete=True, comparisons=comparisons, selected_development_requests=len(selected),
        unique_prompts=len(prompts), model_load_wall_seconds=model_load_seconds,
        serial_reference=reference_measurement, primary_warning_accuracy_retested=False,
        original_jobs_or_cloud_pipelines_modified=False, full_acceptance=False,
        resource=budget.snapshot(), gpu=torch.cuda.get_device_name(0), measurements=measurements)
    common.save(output / 'report.json', result)
    lines = ['# Single-GPU safety judge scheduling trial', '',
        'Development-only fresh judge inference; not a full safety acceptance result.',
        'No warning model, thresholds, data split, maximum length or formal output was changed.',
        'Historical development answers are fixed judge inputs, not new VLM generations.',
        'Model load wall time: %.2f seconds. Arm timings include tokenization, padding, transfers, judge generation and voluntary shared-GPU yielding.' % model_load_seconds,
        '', '| Variant | Round | Batch | Seconds | Exact raw | Exact parsed | Padded slots |',
        '|---|---:|---:|---:|---|---|---:|']
    for item in measurements:
        if 'seconds' in item:
            lines.append('| %s | %d | %d | %.3f | %s | %s | %d |' % (item['variant'], item['repeat'],
                item['batch_size'], item['seconds'], item['comparison']['exact_raw'],
                item['comparison']['exact_parsed'], item['padded_input_slots']))
        else:
            lines.append('| %s | %d | %d | OOM | not checked | not checked | - |' % (
                item['variant'], item['repeat'], item['batch_size']))
    lines += ['', '## Admission', '', '```json', json.dumps(comparisons, indent=2), '```', '',
        'No candidate is activated automatically. These timings are shared A100 measurements, not 5090 or full-pipeline speed guarantees.',
        'Raw judge outputs, parsing and RNG are checked against fresh serial inference. No test label is used for scheduling selection.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(result['verdict'] + ' ' + str(output / 'report.md'), flush=True)


if __name__ == '__main__':
    main()
