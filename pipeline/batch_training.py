"""Fresh InternVL-only training run and throughput profiling, separate from release.

Only scheduling and unused auxiliary-score retention change. The original
prompt set, generation budget, eight features, grouped split and dev-only fit
are reused. Qwen supplies fresh primary answers, not another evaluated cell.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pipeline import common


class BatchBudget:
    """Bound memory and yield compute without stopping another user's process."""
    def __init__(self, torch, cap_gib=24, reserve_gib=8, shared_duty=.5):
        self.torch, self.cap_gib, self.reserve_gib = torch, cap_gib, reserve_gib
        self.shared_duty = shared_duty
        self.work_seconds = self.idle_seconds = 0.
        self.snapshots = []
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(cap_gib * 1024**3 / total, 0)

    def before(self):
        index = str(os.environ.get('MSRC_GPU', '0'))
        while True:
            line = subprocess.check_output(['nvidia-smi', '--id=' + index,
                '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            uuid, free, util = [part.strip() for part in line.split(',')]
            own = self.torch.cuda.memory_reserved(0) / 1024**2
            if int(free) + own >= (self.cap_gib + self.reserve_gib) * 1024:
                break
            print('BATCH_MEMORY_WAIT ' + line.strip(), flush=True)
            time.sleep(10)
        snapshot = dict(gpu_uuid=uuid, free_mib=int(free), utilization=int(util), time_ns=time.time_ns())
        self.snapshots.append(snapshot)
        self.snapshots = self.snapshots[-32:]

    @contextmanager
    def work(self, stage):
        self.before()
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        yield
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.work_seconds += elapsed
        idle = elapsed * (1 / self.shared_duty - 1)
        if idle:
            time.sleep(idle)
            self.idle_seconds += idle

    def snapshot(self):
        return dict(allocator_cap_gib=self.cap_gib, reserve_gib=self.reserve_gib,
            work_seconds=self.work_seconds, voluntary_idle_seconds=self.idle_seconds,
            duty=self.shared_duty, peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),
            peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
            shared_hardware_not_exclusive=True, device_snapshots=self.snapshots)


def scoped_factual():
    from pipeline import factual
    factual.MODELS = {'internvl': common.model_paths()['internvl']}
    factual.DONORS = {'internvl': 'qwen'}
    return factual


def extra_sources():
    names = ['pipeline/batch_training.py', 'pipeline/batch_judge.py',
             'extensions/multilayer/batched.py', 'pipeline/acceptance_check.py']
    return {name: common.sha(common.ROOT / name) for name in names}


def prepare(root):
    factual = scoped_factual()
    root.mkdir(parents=True, exist_ok=False)
    factual.prepare(root, 'full')
    protocol = json.loads((root / 'protocol.json').read_text())
    protocol.update(scope='InternVL-only two-cell full-data experiment; NOT original six-cell acceptance',
        evaluated_models=['internvl'], auxiliary_models=['qwen'],
        expected_cells=['internvl_pope', 'internvl_textvqa'], batch_sources=extra_sources(),
        batch_sampling_protocol='batch-dependent global RNG; seed reset before each fresh cell',
        oom_policy='halve batch on OOM; restore RNG before retry; never truncate or drop samples',
        auxiliary_collection='fresh primary answers only; no auxiliary fitting',
        extra_generation_scores_omitted=True, base_model_weights_frozen=True,
        image_cache='bounded 1 GiB CPU cache of preprocessing and frozen InternVL visual embeddings only',
        cached_answers=False, greedy_length_grouping=True, sampled_request_order_preserved=True)
    common.save(root / 'protocol.json', protocol)


def verify(root):
    factual = scoped_factual()
    protocol = json.loads((root / 'protocol.json').read_text())
    if (protocol.get('evaluated_models') != ['internvl'] or protocol['parameter_mode'] != 'fresh_fit'
            or protocol.get('batch_sources') != extra_sources()):
        raise ValueError('scoped fresh-training source contract changed')
    for name, digest in protocol['source_hashes'].items():
        if common.sha(common.ROOT / name) != digest:
            raise ValueError('source changed after run preparation: ' + name)
    for dataset, state in protocol['inputs'].items():
        if common.sha(root / 'inputs' / (dataset + '.jsonl')) != state['sha256']:
            raise ValueError('input changed')
    return factual, protocol


def make_provider(family, budget):
    from msrc.providers import HFInternVLProvider, HFQwenVLProvider, HFOCRProvider
    provider = {'internvl': HFInternVLProvider, 'qwen': HFQwenVLProvider}[family](
        model_path=common.model_paths()[family], device='cuda:0', dtype='bfloat16',
        seed=common.SEED, max_new_tokens=128, max_pixels=1003520,
        ocr_provider=HFOCRProvider() if family == 'internvl' else None)
    with budget.work('load_' + family):
        provider.load()
    provider._model.requires_grad_(False)
    provider._batch_measurements = {}
    if family == 'internvl':
        from msrc.providers.internvl_cache import InternVLInputCache
        provider._batch_input_cache = InternVLInputCache(provider)
        provider._batch_sort_greedy = True
    return provider


def attention_backends(provider):
    config = provider._model.config
    result = {'model': getattr(config, '_attn_implementation', None)}
    for name in ('vision_config', 'text_config'):
        result[name] = getattr(getattr(config, name, None), '_attn_implementation', None)
    return result


def optimization_check(root, args):
    import math
    from msrc.batch_collection import collect_chunk
    from msrc.signals import Evidence, build_report
    verify(root)
    import torch
    torch.set_num_threads(2)
    budget = BatchBudget(torch, args.cap_gib, shared_duty=args.duty)
    provider = make_provider('internvl', budget)
    provider._batch_sort_greedy = False
    result = dict(complete=False, development_only=True, tolerance=dict(abs=1e-6, rel=1e-5),
        changed_prompts=False, changed_image_resolution=False, changed_token_budget=False,
        attention_backends=attention_backends(provider), cases=[])
    path = root / 'optimization_check.json'
    if path.exists():
        raise ValueError('optimization audit already exists; preserve its results')
    def payload(row):
        evidence = Evidence.from_dict(row['evidence'])
        values = [evidence.primary, *evidence.views, *evidence.resamples, *evidence.verifications]
        return dict(texts=[sample.text for sample in values],
            readings=[[sample.sequence_confidence, sample.sequence_entropy, sample.first_token_margin]
                      for sample in values], ocr=evidence.ocr_texts,
            signals=[dict(name=s.name, risk=s.risk, available=s.available)
                     for s in build_report(evidence).signals])
    def compare(left, right, where='root'):
        if isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                raise ValueError('cache field mismatch: ' + where)
            for key in left:
                compare(left[key], right[key], where + '.' + key)
        elif isinstance(left, list):
            if not isinstance(right, list) or len(left) != len(right):
                raise ValueError('cache list mismatch: ' + where)
            for i, (a, b) in enumerate(zip(left, right)):
                compare(a, b, where + '.' + str(i))
        elif isinstance(left, float):
            if not math.isclose(left, right, abs_tol=1e-6, rel_tol=1e-5):
                raise ValueError('cache numeric mismatch: ' + where + ': ' + repr((left, right)))
        elif left != right:
            raise ValueError('cache value mismatch: ' + where + ': ' + repr((left, right)))
    try:
        for dataset in ('pope', 'textvqa'):
            dev = [row for row in common.read_rows(root / 'inputs' / (dataset + '.jsonl'))
                   if row['acceptance_split'] == 'dev']
            selected = [dev[0], dev[len(dev)//2]]
            case = dict(dataset=dataset, ids=[row['id'] for row in selected], batch_size=1)
            result['cases'].append(case)
            reference = None
            from msrc.providers.internvl_cache import InternVLInputCache
            cache = InternVLInputCache(provider)
            for mode in ('uncached', 'cached_cold', 'cached_warm'):
                provider._batch_input_cache = None if mode == 'uncached' else cache
                provider._batch_measurements = {}
                torch.manual_seed(common.SEED)
                before_work, before_idle = budget.work_seconds, budget.idle_seconds
                started = time.perf_counter()
                rows = collect_chunk(provider, selected, batch_size=1, budget=budget, adaptive=False)
                current = [payload(row) for row in rows]
                case[mode] = dict(elapsed_seconds=time.perf_counter()-started,
                    work_seconds=budget.work_seconds-before_work,
                    voluntary_idle_seconds=budget.idle_seconds-before_idle,
                    stages=provider._batch_measurements, payload=current,
                    cache=cache.snapshot() if mode != 'uncached' else None)
                if reference is None:
                    reference = current
                else:
                    compare(reference, current, dataset + '.' + mode)
                common.save(path, result)
                print('OPTIMIZATION_CHECK ' + dataset + ' ' + mode + ' OK', flush=True)
            case['cold_speedup'] = case['uncached']['elapsed_seconds'] / case['cached_cold']['elapsed_seconds']
            case['warm_speedup'] = case['uncached']['elapsed_seconds'] / case['cached_warm']['elapsed_seconds']
            from msrc.batch_collection import primary_prompt
            from msrc.providers.batched import generate_batch
            batch_reference = None
            cache = InternVLInputCache(provider)
            case['batch2'] = {}
            for mode in ('uncached', 'cached_cold', 'cached_warm'):
                provider._batch_input_cache = None if mode == 'uncached' else cache
                torch.manual_seed(common.SEED)
                with budget.work('audit_batch2'):
                    values = generate_batch(provider, [row['image'] for row in selected],
                        [primary_prompt(row) for row in selected])
                current = [dict(text=value.text, confidence=value.sequence_confidence,
                    entropy=value.sequence_entropy, margin=value.first_token_margin) for value in values]
                if batch_reference is None:
                    batch_reference = current
                else:
                    compare(batch_reference, current, dataset + '.batch2.' + mode)
                case['batch2'][mode] = current
                common.save(path, result)
        result.update(complete=True, verdict='OPTIMIZATION_CHECK_OK', resource=budget.snapshot())
        common.save(path, result)
        print('OPTIMIZATION_CHECK_OK', flush=True)
    except BaseException as error:
        result.update(error=repr(error), resource=budget.snapshot())
        common.save(path, result)
        raise


def profile(root, args):
    from msrc.batch_collection import collect_chunk
    verify(root)
    audit = json.loads((root / 'optimization_check.json').read_text())
    if not audit.get('complete') or audit.get('verdict') != 'OPTIMIZATION_CHECK_OK':
        raise ValueError('cache correctness audit must pass before performance selection')
    import torch
    torch.set_num_threads(2)
    budget = BatchBudget(torch, args.cap_gib, shared_duty=args.duty)
    profile_path = root / 'batch_profile.json'
    if profile_path.exists():
        raise ValueError('profile exists; preserve the measurements or choose a new run')
    results = dict(created_ns=time.time_ns(), gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__, candidates=args.batch_sizes, measurements=[], selections={},
        estimator='sample-throughput extrapolation, not a completed full run',
        sample_source='development only; selection uses runtime, not labels or metric gains',
        optimization_check_sha256=common.sha(root / 'optimization_check.json'))
    for family in ('qwen', 'internvl'):
        provider = make_provider(family, budget)
        for dataset, count in [('pope', 9000), ('textvqa', 5000)]:
            dev = [row for row in common.read_rows(root / 'inputs' / (dataset + '.jsonl'))
                   if row['acceptance_split'] == 'dev']
            # Evenly spaced development inputs; include longest questions without looking at outputs.
            cohort = max(args.profile_items, 16 if family == 'internvl' else max(args.batch_sizes))
            indices = {round(i * (len(dev) - 1) / max(cohort - 1, 1))
                       for i in range(cohort)}
            indices.add(max(range(len(dev)), key=lambda i: len(dev[i]['question'])))
            reference = {}
            for size in args.batch_sizes:
                if family == 'internvl' and size > 16:
                    break
                selected_indices = indices
                selected = [dev[i] for i in sorted(selected_indices)]
                torch.manual_seed(common.SEED)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                provider._batch_measurements = {}
                work_before, idle_before = budget.work_seconds, budget.idle_seconds
                started = time.perf_counter()
                item = dict(family=family, dataset=dataset, batch_size=size, samples=len(selected),
                    attention_backends=attention_backends(provider), same_cohort_for_all_batch_sizes=True)
                try:
                    values = collect_chunk(provider, selected, batch_size=size,
                                           donor_only=family == 'qwen', budget=budget, adaptive=False)
                    elapsed = time.perf_counter() - started
                    answers = {row['id']: row['answer'] for row in values}
                    if not reference:
                        reference = answers
                    compared = set(answers) & set(reference)
                    prediction_path = root / 'profile_predictions' / (family + '_' + dataset + '_batch' + str(size) + '.json')
                    common.save(prediction_path, dict(development_only=True, requested_batch_size=size,
                        selection_uses_labels=False,
                        predictions=[dict(id=row['id'], answer=row['answer']) for row in values]))
                    item.update(status='OK', elapsed_seconds=elapsed, samples_per_second=len(values) / elapsed,
                        projected_full_seconds=elapsed / len(values) * count,
                        primary_answer_comparisons=len(compared),
                        primary_answer_matches_batch1=sum(answers[key] == reference[key] for key in compared),
                        predictions=str(prediction_path.relative_to(root)),
                        work_seconds=budget.work_seconds - work_before,
                        voluntary_idle_seconds=budget.idle_seconds - idle_before,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), stages=provider._batch_measurements,
                        image_cache=getattr(provider, '_batch_input_cache', None).snapshot()
                            if family == 'internvl' else None)
                except torch.OutOfMemoryError as error:
                    item.update(status='OOM', error=str(error)[:500])
                    gc.collect()
                    torch.cuda.empty_cache()
                results['measurements'].append(item)
                common.save(profile_path, results)
                print(json.dumps(item), flush=True)
                if item['status'] == 'OOM':
                    break
            valid = [item for item in results['measurements']
                     if item['family'] == family and item['dataset'] == dataset and item['status'] == 'OK']
            if not valid:
                raise ValueError('even batch 1 did not fit within the resource limit')
            chosen = max(valid, key=lambda item: item['samples_per_second'])
            results['selections'][family + '_' + dataset] = chosen
            common.save(profile_path, results)
        del provider
        gc.collect()
        torch.cuda.empty_cache()
    results.update(complete=True, resource=budget.snapshot(),
        projected_factual_seconds=sum(v['projected_full_seconds'] for v in results['selections'].values()))
    common.save(profile_path, results)
    print('BATCH_PROFILE_COMPLETED', flush=True)


def collect(root, args):
    from msrc.batch_collection import collect_chunk
    verify(root)
    profile_data = json.loads((root / 'batch_profile.json').read_text())
    if not profile_data.get('complete'):
        raise ValueError('finish resource profiling before fresh full collection')
    import torch
    torch.set_num_threads(2)
    budget = BatchBudget(torch, args.cap_gib, shared_duty=args.duty)
    for family in ('qwen', 'internvl'):
        provider = make_provider(family, budget)
        for dataset in ('pope', 'textvqa'):
            rows = common.read_rows(root / 'inputs' / (dataset + '.jsonl'))
            key = family + '_' + dataset
            output = common.inside(root / 'evidence' / (key + '.jsonl'))
            output.parent.mkdir(parents=True, exist_ok=True)
            size = profile_data['selections'][key]['batch_size']
            if output.exists():
                raise ValueError('fresh output required; no old or partial predictions are imported')
            torch.manual_seed(common.SEED)
            provider._batch_measurements = {}
            started_ns, started = time.time_ns(), time.perf_counter()
            with output.open('x', encoding='utf-8') as handle:
                for start in range(0, len(rows), size):
                    part = rows[start:start + size]
                    values = collect_chunk(provider, part, batch_size=size,
                                           donor_only=family == 'qwen', budget=budget)
                    for value in values:
                        value['collected_ns'] = time.time_ns()
                        common.emit(handle, value)
                    progress = dict(cell=key, completed=start + len(part), total=len(rows), batch_size=size,
                        elapsed_seconds=time.perf_counter() - started, updated_ns=time.time_ns(),
                        stages=provider._batch_measurements,
                        image_cache=getattr(provider, '_batch_input_cache', None).snapshot()
                            if family == 'internvl' else None)
                    common.save(root / 'progress.json', progress)
                    if (start // size) % 10 == 0:
                        print(json.dumps(progress), flush=True)
            common.save(output.with_suffix('.resources.json'), dict(budget.snapshot(), complete=True,
                started_ns=started_ns, ended_ns=time.time_ns(), rows=len(rows), batch_size=size,
                oom_retried_batch_sizes=getattr(provider, '_batch_retries', []),
                stages=provider._batch_measurements, attention_backends=attention_backends(provider),
                image_cache=getattr(provider, '_batch_input_cache', None).snapshot()
                    if family == 'internvl' else None))
        del provider
        gc.collect()
        torch.cuda.empty_cache()
    print('BATCH_FACTUAL_COLLECTED', flush=True)


def evaluate(root):
    factual, _ = verify(root)
    factual.evaluate(root)
    from pipeline.acceptance_check import factual_check
    report = json.loads((root / 'report.json').read_text())
    factual_check(root, report, expected_cells={'internvl_pope', 'internvl_textvqa'})
    common.save(root / 'scope_audit.json', dict(complete=True, evaluated_models=['internvl'],
        original_six_cell_coverage=False, dataset_rows={'pope': 9000, 'textvqa': 5000},
        fitted_on_development_only=True, report_sha256=common.sha(root / 'report.json')))
    print('INTERNVL_TWO_CELLS_CHECKED', flush=True)


def joint(root, run_id):
    import html
    import math
    from extensions.multilayer import variants
    from extensions.multilayer.batched import BatchedExperiment
    from extensions.multilayer.report_artifacts import page, table, markdown_table
    verify(root)
    from pipeline.acceptance_check import factual_check
    factual = json.loads((root / 'report.json').read_text())
    factual_check(root, factual, expected_cells={'internvl_pope', 'internvl_textvqa'})
    multilayer = root.parent / 'multilayer'
    experiment = BatchedExperiment(multilayer, multilayer / 'materials')
    experiment.setup()
    safety = variants.audit_safety(experiment, 'no_fit')
    cells = factual['cells']
    gain = math.fsum(c['relative_improvement'] for c in cells) / 2
    primary = safety['primary']['temporal_hold']
    passed = all(c['signals_pass'] and c['calibration_pass'] for c in cells) and primary['accuracy'] >= .75
    output = multilayer / 'joint'
    output.mkdir(exist_ok=True)
    tables = dict(
        signal_report=(('Dataset', 'Minimum actual signals', 'Requirement'),
                       [(c['cell'], c['minimum_actual_signals_per_test_sample'], 3) for c in cells]),
        calibration_report=(('Dataset', 'Development', 'Test', 'ECE before', 'ECE after', 'Relative gain'),
            [(c['cell'], c['n_dev'], c['n_test'], '%.8f' % c['ece_before'], '%.8f' % c['ece_after'],
              '%.2f%%' % (100 * c['relative_improvement'])) for c in cells]
            + [('Two-cell mean', '-', '-', '-', '-', '%.2f%%' % (100 * gain))]),
        warning_report=(('Scope', 'Test n', 'TP', 'TN', 'FP', 'FN', 'Accuracy'),
            [(name, v['n'], v['TP'], v['TN'], v['FP'], v['FN'], '%.2f%%' % (v['accuracy'] * 100))
             for name, v in [('overall', primary), *sorted(safety['by_source']['temporal_hold'].items())]]))
    context = 'InternVL-only two-cell fresh-training experiment. NOT original three-model acceptance.'
    all_markdown, all_html = ['# Scoped results', '', context, ''], ['<p>' + context + '</p>']
    for name, (headers, values) in tables.items():
        text = '# ' + name + '\n\n' + context + '\n\n' + markdown_table(headers, values) + '\n'
        common.inside(output / (name + '.md')).write_text(text, encoding='utf-8')
        rendered = table(headers, [[html.escape(str(v)) for v in row] for row in values])
        common.inside(output / (name + '.html')).write_text(page(name, '<p>' + context + '</p>' + rendered), encoding='utf-8')
        all_markdown += [text]
        all_html += ['<h2>' + name + '</h2>', rendered]
    result = dict(complete=True, verdict='SCOPED_PASS' if passed else 'SCOPED_FAIL',
        original_three_model_acceptance=False, mode='onsite_fresh_training',
        evaluated_models=['internvl'], factual_cells=cells, enhanced_safety=safety,
        mean_relative_improvement=gain, report_ns=time.time_ns(), run_id=run_id)
    common.save(output / 'summary.json', result)
    common.inside(output / 'report.md').write_text('\n'.join(all_markdown), encoding='utf-8')
    common.inside(output / 'report.html').write_text(page(result['verdict'], ''.join(all_html)), encoding='utf-8')
    print(result['verdict'], str(output / 'report.html'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--phase', choices=['prepare', 'optimization_check', 'profile', 'collect', 'evaluate', 'report'], required=True)
    parser.add_argument('--batch-sizes', type=lambda value: [int(v) for v in value.split(',')], default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--profile-items', type=int, default=8)
    parser.add_argument('--cap-gib', type=int, default=24)
    parser.add_argument('--duty', type=float, default=.5)
    args = parser.parse_args()
    if (not args.run_id.replace('_', '').replace('-', '').isalnum() or args.profile_items < 2
            or not args.batch_sizes or args.batch_sizes[0] != 1
            or args.batch_sizes != sorted(set(args.batch_sizes)) or min(args.batch_sizes) < 1
            or not 0 < args.duty <= 1 or not 18 <= args.cap_gib <= 28):
        parser.error('invalid run ID, batch grid, memory or duty')
    common.initialize()
    common.select_device()
    root = common.inside(common.runs_dir() / args.run_id / 'factual')
    if args.phase == 'prepare':
        prepare(root)
    elif args.phase == 'optimization_check':
        optimization_check(root, args)
    elif args.phase == 'profile':
        profile(root, args)
    elif args.phase == 'collect':
        collect(root, args)
    elif args.phase == 'evaluate':
        evaluate(root)
    else:
        joint(root, args.run_id)


if __name__ == '__main__':
    main()
