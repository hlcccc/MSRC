"""Development-only precision trial. Never an acceptance or calibration result."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import time

from pipeline import common
from pipeline.batch_training import BatchBudget, attention_backends


def cohort(rows, count, *, longest=True):
    dev = [row for row in rows if row.get('acceptance_split') == 'dev']
    if count < 2 or len(dev) < count:
        raise ValueError('a sufficient, development-only cohort is required')
    indices = {round(i * (len(dev) - 1) / (count - 1)) for i in range(count)}
    if longest:
        indices.add(max(range(len(dev)), key=lambda i: len(dev[i]['question'])))
    selected = [dev[i] for i in sorted(indices)]
    if (len({row['id'] for row in selected}) != len(selected)
            or any(row.get('answer') or row.get('evidence') for row in selected)):
        raise ValueError('unique, raw development inputs are required')
    return selected


def provider(family, precision, budget):
    from msrc.providers import HFInternVLProvider, HFQwenVLProvider, HFOCRProvider
    cls = {'internvl': HFInternVLProvider, 'qwen': HFQwenVLProvider}[family]
    kwargs = dict(model_path=common.model_paths()[family], dtype='bfloat16',
        device='cuda:0', max_new_tokens=128, seed=common.SEED,
        quantization=None if precision == 'bf16' else 'bnb_nf4',
        ocr_provider=HFOCRProvider() if family == 'internvl' else None)
    if family == 'qwen':
        kwargs['max_pixels'] = 1003520
    result = cls(**kwargs)
    with budget.work('load_' + family):
        result.load()
    result._model.requires_grad_(False)
    result._batch_measurements = {}
    if family == 'internvl':
        from msrc.providers.internvl_cache import InternVLInputCache
        result._batch_input_cache = InternVLInputCache(result)
        result._batch_sort_greedy = True
    return result


def fingerprint(rows):
    return [dict(id=row['id'], image_sha256=common.sha(row['image']),
                 question=row['question']) for row in rows]


def run(args):
    import torch
    from msrc.batch_collection import collect_chunk, primary_prompt
    from msrc.providers.batched import generate_many
    from msrc.providers.internvl_cache import InternVLInputCache
    from msrc.signals import Evidence, build_report
    from msrc.types import Sample
    from pipeline.factual import SIGNALS
    torch.set_num_threads(2)
    output = common.inside(common.runs_dir() / args.run_id / args.precision)
    output.mkdir(parents=True, exist_ok=False)
    budget = BatchBudget(torch, args.cap_gib, shared_duty=.5)
    sources = {name: args.inputs / (name + '.jsonl') for name in ('pope', 'textvqa')}
    rows = {name: common.read_rows(path) for name, path in sources.items()}
    selected = {name: cohort(value, args.full_items) for name, value in rows.items()}
    result = dict(complete=False, acceptance_test=False, calibrator_fitted=False,
        development_only=True, precision=args.precision, seed=common.SEED,
        full_signal_candidates=SIGNALS, full_calls_per_sample=11,
        max_new_tokens=128, prompts_or_image_preprocessing_changed=False,
        gpu=torch.cuda.get_device_name(0), physical_gpu=__import__('os').environ['MSRC_GPU'],
        torch=torch.__version__, started_ns=time.time_ns(),
        source_hashes={name: common.sha(path) for name, path in sources.items()},
        full_cohorts={name: fingerprint(value) for name, value in selected.items()},
        full_measurements=[], primary_measurements=[], donors={})
    path = output / 'probe.json'
    common.save(path, result)
    model = None
    try:
        # Independent fresh donor answers; never borrowed from a fitted report.
        model = provider('qwen', args.precision, budget)
        for name, part in selected.items():
            generated = generate_many(model, [row['image'] for row in part],
                [primary_prompt(row) for row in part], batch_size=2, collect_scores=False,
                budget=budget, stage='donor')
            result['donors'][name] = {row['id']: sample.text for row, sample in zip(part, generated)}
        result['donor_quantization'] = model.quantization_audit
        common.save(path, result)
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()

        model = provider('internvl', args.precision, budget)
        result['quantization'] = model.quantization_audit
        result['model_footprint_bytes'] = model._model.get_memory_footprint()
        result['attention_backends'] = attention_backends(model)
        # Warm-up is recorded separately and never counted as measured samples.
        first = selected['pope'][0]
        start = time.perf_counter()
        generate_many(model, [first['image']], [primary_prompt(first)], batch_size=1,
            budget=budget, stage='warmup')
        result['warmup_seconds'] = time.perf_counter() - start
        common.save(path, result)
        for name, part in selected.items():
            for size in args.batch_sizes:
                torch.manual_seed(common.SEED)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model._batch_input_cache = InternVLInputCache(model)
                model._batch_measurements = {}
                work, idle = budget.work_seconds, budget.idle_seconds
                start = time.perf_counter()
                item = dict(dataset=name, batch_size=size, samples=len(part))
                try:
                    values = collect_chunk(model, part, batch_size=size, budget=budget, adaptive=False)
                    signal_rows = []
                    for row in values:
                        evidence = Evidence.from_dict(row['evidence'])
                        evidence.cross_model = [Sample(text=result['donors'][name][row['id']])]
                        readings = [signal.to_dict() for signal in build_report(evidence).signals
                                    if signal.name in SIGNALS]
                        if {value['name'] for value in readings} != set(SIGNALS):
                            raise ValueError('signal candidate set changed')
                        if any(not math.isfinite(value['risk']) for value in readings):
                            raise ValueError('non-finite signal reading')
                        signal_rows.append(dict(id=row['id'], answer=row['answer'], signals=readings,
                            warnings=row['collection_warnings'], evidence=evidence.to_dict()))
                    item.update(status='OK', seconds=time.perf_counter()-start,
                        work_seconds=budget.work_seconds-work, voluntary_idle_seconds=budget.idle_seconds-idle,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), stages=model._batch_measurements,
                        availability_counts=dict(Counter(value['name'] for row in signal_rows
                            for value in row['signals'] if value['available'])),
                        image_cache=model._batch_input_cache.snapshot())
                    common.save(output / (name + '_full_batch' + str(size) + '.json'), signal_rows)
                except torch.OutOfMemoryError as error:
                    item.update(status='OOM', seconds=time.perf_counter()-start, error=str(error)[:1000])
                    gc.collect()
                    torch.cuda.empty_cache()
                result['full_measurements'].append(item)
                common.save(path, result)
                print(json.dumps(item), flush=True)
                if item['status'] == 'OOM':
                    break
            valid = [item for item in result['full_measurements'] if item['dataset'] == name and item['status'] == 'OK']
            if not valid:
                raise ValueError('full signal collection did not fit even at batch 1')
            size = max(valid, key=lambda item: item['samples']/item['seconds'])['batch_size']
            primary_rows = cohort(rows[name], args.primary_items, longest=False)
            model._batch_input_cache = InternVLInputCache(model)
            model._batch_measurements = {}
            work, idle = budget.work_seconds, budget.idle_seconds
            start = time.perf_counter()
            values = generate_many(model, [row['image'] for row in primary_rows],
                [primary_prompt(row) for row in primary_rows], batch_size=size,
                collect_scores=True, budget=budget, stage='primary_100')
            if any(value.sequence_confidence is None or value.sequence_entropy is None for value in values):
                raise ValueError('primary distribution readings disappeared')
            item = dict(dataset=name, samples=len(values), requested_batch_size=size,
                seconds=time.perf_counter()-start, work_seconds=budget.work_seconds-work,
                voluntary_idle_seconds=budget.idle_seconds-idle, stages=model._batch_measurements,
                predictions=[dict(id=row['id'], answer=value.text, confidence=value.sequence_confidence,
                                  entropy=value.sequence_entropy) for row, value in zip(primary_rows, values)],
                oom_retries=getattr(model, '_batch_retries', []), input_identity=fingerprint(primary_rows))
            result['primary_measurements'].append(item)
            common.save(path, result)
        result.update(complete=True, verdict='PROBE_COMPLETED_NOT_ACCEPTANCE', ended_ns=time.time_ns())
        print(result['verdict'], flush=True)
    except BaseException as error:
        result.update(error=repr(error), ended_ns=time.time_ns())
        raise
    finally:
        result['resource'] = budget.snapshot()
        common.save(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--precision', choices=['bf16', 'bnb_nf4'], required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--cap-gib', type=int, default=8)
    parser.add_argument('--full-items', type=int, default=16)
    parser.add_argument('--primary-items', type=int, default=100)
    parser.add_argument('--batch-sizes', type=lambda value: [int(v) for v in value.split(',')], default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    if (not args.run_id.replace('_', '').replace('-', '').isalnum()
            or not 6 <= args.cap_gib <= 24 or args.full_items < 2 or args.primary_items < 2
            or not args.batch_sizes or args.batch_sizes[0] != 1
            or args.batch_sizes != sorted(set(args.batch_sizes))):
        parser.error('invalid run ID, sample sizes, memory cap or batch grid')
    common.initialize()
    common.select_device()
    run(args)


if __name__ == '__main__':
    main()
