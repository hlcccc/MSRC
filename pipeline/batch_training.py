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
    def __init__(self, torch, cap_gib=30, reserve_gib=1, shared_duty=1.):
        self.torch, self.cap_gib, self.reserve_gib = torch, cap_gib, reserve_gib
        self.requested_cap_gib = cap_gib
        self.memory_adjustments = []
        self.shared_duty = shared_duty
        self.work_seconds = self.idle_seconds = 0.
        self.snapshots = []
        total = torch.cuda.get_device_properties(0).total_memory
        if (cap_gib + reserve_gib) * 1024**3 >= total:
            raise ValueError('allocator cap and reserve exceed this device')
        torch.cuda.set_per_process_memory_fraction(cap_gib * 1024**3 / total, 0)

    def before(self):
        index = str(os.environ.get('MSRC_GPU', '0'))
        waiting_since = time.monotonic()
        while True:
            line = subprocess.check_output(['nvidia-smi', '--id=' + index,
                '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            uuid, free, util = [part.strip() for part in line.split(',')]
            own = self.torch.cuda.memory_reserved(0) / 1024**2
            if int(free) + own >= (self.cap_gib + self.reserve_gib) * 1024:
                break
            apps = subprocess.check_output(['nvidia-smi',
                '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
            other = [int(pid) for gpu, pid in (line.split(',') for line in apps.splitlines() if line.strip())
                     if gpu.strip() == uuid and int(pid) != os.getpid()]
            if self.shared_duty == 1. and not other:
                # Driver/context memory is not included in PyTorch's reserved bytes.
                cap = min(self.cap_gib, int((int(free) + own - self.reserve_gib * 1024) // 512) / 2)
                if cap < 18:
                    raise RuntimeError('insufficient usable exclusive GPU memory after CUDA initialization')
                previous = self.cap_gib
                total = self.torch.cuda.get_device_properties(0).total_memory
                self.torch.cuda.set_per_process_memory_fraction(cap * 1024**3 / total, 0)
                self.cap_gib = cap
                adjustment = dict(previous_cap_gib=previous, effective_cap_gib=cap,
                    usable_mib=int(free) + own, reserve_gib=self.reserve_gib,
                    reason='exclusive CUDA context overhead outside allocator', time_ns=time.time_ns())
                self.memory_adjustments.append(adjustment)
                print('EXCLUSIVE_ALLOCATOR_ADJUSTED ' + json.dumps(adjustment), flush=True)
                break
            if time.monotonic() - waiting_since >= 120:
                raise RuntimeError('GPU memory pressure persisted for 120 seconds; foreign PIDs: ' + str(other))
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
            requested_allocator_cap_gib=self.requested_cap_gib,
            context_memory_adjustments=self.memory_adjustments,
            work_seconds=self.work_seconds, voluntary_idle_seconds=self.idle_seconds,
            duty=self.shared_duty, peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),
            peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
            shared_hardware_not_exclusive=self.shared_duty < 1,
            device_snapshots=self.snapshots)


def scoped_factual():
    from pipeline import factual
    factual.MODELS = {'internvl': common.model_paths()['internvl']}
    factual.DONORS = {'internvl': 'qwen'}
    return factual


def extra_sources():
    names = ['pipeline/batch_training.py', 'pipeline/batch_judge.py',
             'extensions/multilayer/batched.py', 'pipeline/acceptance_check.py',
             'pipeline/cloud_quality_5090.py', 'pipeline/cloud_judge_5090.py',
             'extensions/multilayer/cloud_5090.py', 'scripts/cloud_run_5090.py']
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
        extra_generation_scores_omitted=True, base_model_weights_frozen=True)
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
    return provider


def profile(root, args):
    from pipeline.batched_collection import collect_chunk
    verify(root)
    import torch
    torch.set_num_threads(2)
    budget = BatchBudget(torch, args.cap_gib, shared_duty=args.duty)
    profile_path = root / 'batch_profile.json'
    if profile_path.exists():
        raise ValueError('profile exists; preserve the measurements or choose a new run')
    results = dict(created_ns=time.time_ns(), gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__, candidates=args.batch_sizes, measurements=[], selections={},
        estimator='sample-throughput extrapolation, not a completed full run',
        sample_source='development only; selection uses runtime, not labels or metric gains')
    for family in ('qwen', 'internvl'):
        provider = make_provider(family, budget)
        for dataset, count in [('pope', 9000), ('textvqa', 5000)]:
            dev = [row for row in common.read_rows(root / 'inputs' / (dataset + '.jsonl'))
                   if row['acceptance_split'] == 'dev']
            # Evenly spaced development inputs; include longest questions without looking at outputs.
            indices = {round(i * (len(dev) - 1) / max(args.profile_items - 1, 1))
                       for i in range(args.profile_items)}
            indices.add(max(range(len(dev)), key=lambda i: len(dev[i]['question'])))
            reference = {}
            for size in args.batch_sizes:
                selected_indices = indices | {round(i * (len(dev) - 1) / max(size - 1, 1))
                                              for i in range(size)}
                selected = [dev[i] for i in sorted(selected_indices)]
                torch.manual_seed(common.SEED)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                work_before, idle_before = budget.work_seconds, budget.idle_seconds
                started = time.perf_counter()
                item = dict(family=family, dataset=dataset, batch_size=size, samples=len(selected))
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
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
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
    from pipeline.batched_collection import collect_chunk
    verify(root)
    profile_data = json.loads((root / 'batch_profile.json').read_text())
    if not profile_data.get('complete'):
        raise ValueError('finish resource profiling before fresh full collection')
    selection = json.loads((root / 'batch_selection_frozen.json').read_text())
    if (selection['profile_sha256'] != common.sha(root / 'batch_profile.json')
            or selection['test_used_for_selection'] is not False):
        raise ValueError('development-only batch selection changed')
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
            started_ns, started = time.time_ns(), time.perf_counter()
            with output.open('x', encoding='utf-8') as handle:
                for start in range(0, len(rows), size):
                    part = rows[start:start + size]
                    if profile_data['selections'][key]['collection_mode'] == 'serial':
                        from pipeline.cloud_quality_5090 import collect_serial
                        values = collect_serial(provider, part, donor_only=family == 'qwen', budget=budget)
                    else:
                        values = collect_chunk(provider, part, batch_size=size,
                                               donor_only=family == 'qwen', budget=budget)
                    for value in values:
                        value['collected_ns'] = time.time_ns()
                        common.emit(handle, value)
                    progress = dict(cell=key, completed=start + len(part), total=len(rows), batch_size=size,
                        elapsed_seconds=time.perf_counter() - started, updated_ns=time.time_ns())
                    common.save(root / 'progress.json', progress)
                    if (start // size) % 10 == 0:
                        print(json.dumps(progress), flush=True)
            common.save(output.with_suffix('.resources.json'), dict(budget.snapshot(), complete=True,
                started_ns=started_ns, ended_ns=time.time_ns(), rows=len(rows), batch_size=size,
                oom_retried_batch_sizes=getattr(provider, '_batch_retries', [])))
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
    from extensions.multilayer.cloud_5090 import CloudExperiment
    from extensions.multilayer.report_artifacts import page, table, markdown_table
    verify(root)
    from pipeline.acceptance_check import factual_check
    factual = json.loads((root / 'report.json').read_text())
    factual_check(root, factual, expected_cells={'internvl_pope', 'internvl_textvqa'})
    multilayer = root.parent / 'multilayer'
    os.environ['MSRC_JUDGE_BATCH_SIZE'] = '32'
    experiment = CloudExperiment(multilayer, multilayer / 'materials')
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
    parser.add_argument('--phase', choices=['prepare', 'profile', 'collect', 'evaluate', 'report'], required=True)
    parser.add_argument('--batch-sizes', type=lambda value: [int(v) for v in value.split(',')], default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--profile-items', type=int, default=8)
    parser.add_argument('--cap-gib', type=int, default=30)
    parser.add_argument('--duty', type=float, default=1.)
    args = parser.parse_args()
    if (not args.run_id.replace('_', '').replace('-', '').isalnum() or args.profile_items < 2
            or not args.batch_sizes or args.batch_sizes[0] != 1
            or args.batch_sizes != sorted(set(args.batch_sizes)) or min(args.batch_sizes) < 1
            or not 0 < args.duty <= 1 or not 18 <= args.cap_gib <= 30):
        parser.error('invalid run ID, batch grid, memory or duty')
    common.initialize()
    common.select_device()
    root = common.inside(common.runs_dir() / args.run_id / 'factual')
    if args.phase == 'prepare':
        prepare(root)
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
