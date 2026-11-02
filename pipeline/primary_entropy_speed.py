"""Matched fresh inference timing: original 11 vs primary-statistics 8 generations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

from pipeline import common


def cohort(source,items):
    result = {}
    if items<1:
        raise ValueError('positive timing cohort size required')
    for dataset in ('pope','textvqa'):
        dev = [r for r in common.read_rows(source/'inputs'/(dataset+'.jsonl'))
               if r['acceptance_split']=='dev']
        if len(dev)<items+1:
            raise ValueError('enough development examples and a separate warmup are required')
        indices = [round(i*(len(dev)-1)/max(items-1,1)) for i in range(items)]
        chosen = [dev[i] for i in indices]
        warmup = next(row for i,row in enumerate(dev) if i not in indices)
        result[dataset] = dict(rows=chosen,warmup=warmup)
    return result


def summarize(measurements):
    result = {}
    for dataset in ('pope','textvqa'):
        arms = {name:[m for m in measurements if m['dataset']==dataset and m['variant']==name]
                for name in ('observed','primary_stats')}
        if not arms['observed'] or len(arms['observed'])!=len(arms['primary_stats']):
            raise ValueError('paired complete timing arms required')
        medians = {name:statistics.median(v['elapsed_seconds'] for v in values)
                   for name,values in arms.items()}
        work = {name:statistics.median(v['work_seconds'] for v in values)
                for name,values in arms.items()}
        if min(medians.values())<=0 or min(work.values())<=0:
            raise ValueError('positive measured durations required')
        result[dataset] = dict(original_seconds=medians['observed'],fast_seconds=medians['primary_stats'],
            wall_speedup=medians['observed']/medians['primary_stats'],
            wall_time_reduction=1-medians['primary_stats']/medians['observed'],
            generation_work_speedup=work['observed']/work['primary_stats'],
            pairs=len(arms['observed']),items_per_arm=arms['observed'][0]['items'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--parameters',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--items-per-dataset',type=int,default=4)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--gpu',default='0')
    p.add_argument('--cap-gib',type=int,default=19)
    args = p.parse_args()
    if args.repeats<1 or args.batch_size<1:
        p.error('positive repeats and batch size required')
    common.initialize();os.environ['MSRC_GPU']=args.gpu;common.select_device()
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    source = args.source.resolve(strict=True);parameters = args.parameters.resolve(strict=True)
    output = common.inside(common.runs_dir()/args.run_id/'primary_entropy_speed')
    output.mkdir(parents=True,exist_ok=False)
    from msrc.primary_entropy_estimator import PrimaryStatisticsEntropyEstimator
    model = PrimaryStatisticsEntropyEstimator.load(parameters/'estimator.npz')
    contract = model.metadata['contract']
    for name,digest in contract['source_sha256'].items():
        if common.sha(source/name)!=digest:
            raise ValueError('the timing source differs from the frozen training evidence')
    protocol = json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if (protocol['max_new_tokens']!=128 or protocol['k']!=3 or protocol['views']!=4
            or protocol['probes']!=3 or protocol['dtype']!='bfloat16'
            or Path(protocol['models']['internvl']).resolve()!=Path(common.model_paths()['internvl']).resolve()):
        raise ValueError('the original frozen BF16 collection settings are required')
    frozen = {name:common.sha(parameters/name) for name in
              ('estimator.npz','estimator.json','matched_fusion_parameters.json')}
    selected = cohort(source,args.items_per_dataset)
    common.save(output/'protocol.json',dict(source=str(source),parameters=str(parameters),frozen=frozen,
        seed=common.SEED,max_new_tokens=128,k_original=3,k_fast=0,items_per_dataset=args.items_per_dataset,
        repeats=args.repeats,batch_size=args.batch_size,development_only=True,
        original_model_weights_and_precision=True,fresh_answers=True,
        scope='fresh InternVL evidence collection and feature matrix; not complete system acceptance',
        independent_donor_excluded_from_both_timing_arms=True,content_safety_not_measured=True,
        estimated_signal_not_observed_entropy=True,includes_ocr=True,cold_image_cache_each_arm=True,
        model_loading_excluded=True,alternating_arm_order=True,
        cohorts={d:[r['id'] for r in c['rows']] for d,c in selected.items()},
        sources={name:common.sha(common.ROOT/name) for name in ('msrc/primary_entropy_collection.py',
            'msrc/primary_entropy_estimator.py','pipeline/primary_entropy_speed.py',
            'msrc/batch_collection.py','msrc/providers/batched.py')},created_ns=time.time_ns()))
    import torch
    from pipeline.batch_training import BatchBudget,make_provider
    from msrc.batch_collection import collect_chunk
    from msrc.primary_entropy_collection import collect_fast_chunk,signal_matrix
    from msrc.providers.internvl_cache import InternVLInputCache
    from msrc.signals import Evidence,build_report
    torch.set_num_threads(2)
    budget = BatchBudget(torch,args.cap_gib,shared_duty=.5)
    provider = make_provider('internvl',budget)
    provider._batch_sort_greedy = True
    for dataset,part in selected.items():
        collect_fast_chunk(provider,[part['warmup']],dataset=dataset,estimator=model,contract=contract,
                           batch_size=1,budget=budget)
    measurements = []
    for repeat in range(args.repeats):
        order = ('observed','primary_stats') if repeat%2==0 else ('primary_stats','observed')
        for dataset,part in selected.items():
            for variant in order:
                provider._batch_input_cache = InternVLInputCache(provider)
                provider._batch_measurements = {};provider._batch_retries = []
                torch.manual_seed(common.SEED)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                work,idle,calls = budget.work_seconds,budget.idle_seconds,provider.calls
                start = time.perf_counter()
                if variant=='primary_stats':
                    rows = collect_fast_chunk(provider,part['rows'],dataset=dataset,estimator=model,
                        contract=contract,batch_size=args.batch_size,budget=budget)
                    signal_matrix(rows)
                else:
                    rows = collect_chunk(provider,part['rows'],batch_size=args.batch_size,k=3,budget=budget)
                    for row in rows:build_report(Evidence.from_dict(row['evidence']))
                torch.cuda.synchronize()
                item = dict(variant=variant,dataset=dataset,repeat=repeat,items=len(rows),
                    elapsed_seconds=time.perf_counter()-start,work_seconds=budget.work_seconds-work,
                    voluntary_idle_seconds=budget.idle_seconds-idle,
                    measured_generation_requests=provider.calls-calls,
                    expected_generation_requests=len(rows)*(11 if variant=='observed' else 8),
                    stages=provider._batch_measurements,adaptive_batch_retries=provider._batch_retries,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),fresh_answers=True)
                if item['measured_generation_requests']!=item['expected_generation_requests']:
                    raise ValueError('actual generation request count differs from the compared protocol')
                with (output/(dataset+'_'+variant+'_'+str(repeat)+'.jsonl')).open('x',encoding='utf-8') as f:
                    for row in rows:common.emit(f,row)
                measurements.append(item)
                common.save(output/'progress.json',dict(completed_arms=len(measurements),
                    total_arms=args.repeats*4,last=item,time_ns=time.time_ns()))
                common.save(output/'measurements.json',measurements)
                print('FRESH_SPEED_ARM '+json.dumps(item),flush=True)
    if any(common.sha(parameters/name)!=digest for name,digest in frozen.items()):
        raise ValueError('frozen estimator/fusion files changed during timing')
    report = dict(verdict='FRESH_SPEED_BENCHMARK_NOT_ACCEPTANCE',datasets=summarize(measurements),
        device=torch.cuda.get_device_name(0),shared_hardware_not_exclusive=True,
        fresh_model_generation=True,full_system_speedup_measured=False,
        independent_donor_and_content_safety_not_timed=True,resource=budget.snapshot())
    common.save(output/'report.json',report)
    lines = ['# Fresh entropy acceleration benchmark','',
        'Same BF16 model, questions, token budget and batch size; original 11 vs fast 8 generations.',
        'Includes OCR and feature construction. Excludes model loading, independent Qwen and content safety.',
        'Shared GPU with 50% voluntary duty; not exclusive RTX 5090 or complete system timing.','',
        '| Dataset | Items per arm | Original seconds | Fast seconds | Speedup | Time reduction |',
        '|---|---:|---:|---:|---:|---:|']
    for dataset,item in report['datasets'].items():
        lines.append(f"| {dataset} | {item['items_per_arm']} | {item['original_seconds']:.2f} | {item['fast_seconds']:.2f} | {item['wall_speedup']:.3f}x | {item['wall_time_reduction']:.2%} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(report['verdict']+' '+str(output/'report.md'),flush=True)


if __name__=='__main__':main()
