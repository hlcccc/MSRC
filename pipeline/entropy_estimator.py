"""Isolated, group-held-out InternVL entropy-surrogate pilot; not acceptance."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pipeline import common
from msrc.entropy_estimator import (FEATURE_RULE, PromptStateCapture,
    ResampleEntropyEstimator, check_records, development_folds, feature_contract)


def cohort(rows, count, seed):
    if count < 1:
        raise ValueError('positive cohort size required')
    groups = {r.get('acceptance_image_group') for r in rows}
    if None in groups or not groups:
        raise ValueError('frozen image-group identities required')
    groups = sorted(groups)
    groups.sort(key=lambda group: hashlib.sha256((str(seed) + group).encode()).hexdigest())
    selected, wanted = [], set()
    for group in groups:
        wanted.add(group)
        if sum(r['acceptance_image_group'] in wanted for r in rows) >= count:
            break
    selected = [r for r in rows if r['acceptance_image_group'] in wanted]
    if len(selected) < count:
        raise ValueError('insufficient data for the preregistered cohort')
    return selected


def prepare(source, output, n_dev=256, n_test=128):
    from pipeline.factual import SIGNALS
    from pipeline.factual_parameters import features
    from msrc.signals import Evidence, ObjectSupport
    output.mkdir(parents=True, exist_ok=False)
    selected, origins = [], {}
    image_hashes = {}
    for dataset in ('pope', 'textvqa'):
        raw, rows, readings, x, available, y, dev = features(source, 'internvl', dataset)
        all_groups = {}
        for row in raw:
            group, split = row['acceptance_image_group'], row['acceptance_split']
            if group in all_groups and all_groups[group] != split:
                raise ValueError('source image group crosses development/test')
            all_groups[group] = split
        chosen = cohort([r for r in raw if r['acceptance_split']=='dev'], n_dev, common.SEED)
        chosen += cohort([r for r in raw if r['acceptance_split']=='test'], n_test, common.SEED)
        ids = {r['id'] for r in chosen}
        for index, (row, record) in enumerate(zip(rows, raw)):
            if record['id'] not in ids:
                continue
            ev = Evidence.from_dict(row['evidence'])
            if (len(ev.resamples) != 3 or len(ev.views) != 4 or len(ev.verifications) != 3
                    or not readings[index]['resample_consistency'].available):
                raise ValueError('the pilot teacher requires K=3, four views and three probes')
            view_support = ObjectSupport().compute(replace(ev, resamples=[]))
            image = record['image']
            if image not in image_hashes:
                image_hashes[image] = common.sha(image)
            selected.append(dict(record, image_group=image_hashes[image], dataset=dataset,
                target=float(readings[index]['resample_consistency'].risk), label=int(y[index]),
                teacher_risks=x[index].tolist(), available=available[index].tolist(),
                object_support_views_risk=view_support.risk,
                object_support_views_available=view_support.available,
                historical_primary_answer=row['answer']))
        origins[dataset] = {name: common.sha(source / name) for name in
            ('inputs/'+dataset+'.jsonl', 'evidence/internvl_'+dataset+'.jsonl',
             'evidence/qwen_'+dataset+'.jsonl')}
    check_records(selected)
    for split in ('dev', 'test'):
        with (output / (split+'.jsonl')).open('x', encoding='utf-8') as handle:
            for row in selected:
                if row['acceptance_split'] == split:
                    common.emit(handle, row)
    protocol = dict(seed=common.SEED, source=str(source), origin_hashes=origins,
        source_protocol_sha256=common.sha(source/'protocol.json'), requested_dev_per_dataset=n_dev,
        requested_test_per_dataset=n_test, selected_counts={split:sum(r['acceptance_split']==split
            for r in selected) for split in ('dev','test')}, projection_dimension=64, ridge_alpha=10.,
        selection='whole image groups ranked by seeded hash; no label/metric selection',
        teacher='historical K=3 normalized string-cluster entropy; not entailment semantic entropy',
        ablation_control='observed_views retains observed entropy but uses views-only object support',
        feature_rule=FEATURE_RULE, surrogate_signal='resample_entropy_estimate',
        object_support_change='object_support_views uses four views only; original uses views+resamples',
        original_signals=SIGNALS, training_scope='development only; grouped OOF inputs to fusion',
        base_model='InternVL3-8B-hf', dtype='bfloat16', max_new_tokens=128,
        image_groups_cross_split=0, historical_evidence_analysis=True,
        full_acceptance=False, created_ns=time.time_ns())
    protocol['cohort_hashes'] = {split:common.sha(output/(split+'.jsonl')) for split in ('dev','test')}
    protocol['sources'] = {name:common.sha(common.ROOT/name) for name in
        ('msrc/entropy_estimator.py','pipeline/entropy_estimator.py','msrc/batch_collection.py',
         'msrc/providers/batched.py','msrc/providers/internvl_cache.py')}
    common.save(output/'protocol.json', protocol)
    print('ENTROPY_COHORT_PREPARED '+json.dumps(protocol['selected_counts']), flush=True)


def extract(provider, rows, output, split, budget, batch_size):
    from msrc.batch_collection import primary_prompt
    tokenizer = provider._processor.tokenizer
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    values, ids, started = [], [], time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset+batch_size]
        texts = [provider._processor.apply_chat_template([{'role':'user','content':[
            {'type':'image'}, {'type':'text','text':primary_prompt(row)}]}],
            add_generation_prompt=True) for row in batch]
        with budget.work('extract_'+split):
            inputs = provider._batch_input_cache.prepare([r['image'] for r in batch], texts)
            # generate() removes IDs at prefill; a direct HF forward does not.
            if 'inputs_embeds' in inputs:
                inputs.pop('input_ids', None)
            with PromptStateCapture(provider) as capture, provider._torch.inference_mode():
                result = provider._model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            values.append(capture.features())
            del result, inputs
        ids += [r['id'] for r in batch]
        common.save(output/'progress.json', dict(phase='extract_'+split, completed=len(ids),
            total=len(rows), elapsed_seconds=time.perf_counter()-started, time_ns=time.time_ns()))
        if len(ids) % 16 == 0 or len(ids) == len(rows):
            print('EXTRACT '+split+' '+str(len(ids))+'/'+str(len(rows)), flush=True)
    values = np.concatenate(values)
    np.savez_compressed(output/(split+'_features.npz'), features=values, ids=np.asarray(ids))
    return values


def train(output, dev, x, contract):
    y = np.asarray([r['target'] for r in dev])
    folds = development_folds(dev)
    oof = np.zeros(len(dev))
    audits = []
    for fold in range(4):
        fit, validation = folds != fold, folds == fold
        model = ResampleEntropyEstimator().fit(x[fit], y[fit], [r for r,b in zip(dev,fit) if b])
        oof[validation] = model.predict(x[validation])
        fit_groups = {r['image_group'] for r,b in zip(dev,fit) if b}
        val_groups = {r['image_group'] for r,b in zip(dev,validation) if b}
        if fit_groups & val_groups:
            raise ValueError('OOF image group leakage')
        audits.append(dict(fold=fold, train_groups=sorted(fit_groups), validation_groups=sorted(val_groups)))
    model = ResampleEntropyEstimator().fit(x,y,dev)
    model.save(output/'estimator.npz', contract)
    np.savez_compressed(output/'development_oof.npz', predictions=oof, folds=folds)
    common.save(output/'crossfit_audit.json', dict(folds=audits, test_used=False,
        training_ids=[r['id'] for r in dev], train_groups=model.metadata['train_groups']))
    print('ENTROPY_ESTIMATOR_TRAINED '+str(output/'estimator.npz'), flush=True)
    return model, oof


def matrices(rows, surrogate=None, *, views_only=False):
    x = np.asarray([r['teacher_risks'] for r in rows], dtype=float)
    a = np.asarray([r['available'] for r in rows], dtype=bool)
    if surrogate is not None:
        x[:,1], a[:,1] = surrogate, True
    if surrogate is not None or views_only:
        x[:,5] = [r['object_support_views_risk'] for r in rows]
        a[:,5] = [r['object_support_views_available'] for r in rows]
    return x,a


def fit_fusions(output, dev, oof):
    from msrc.model import RiskCalibrator
    check_records(dev, development=True)
    oof = np.asarray(oof)
    if oof.shape != (len(dev),) or not np.isfinite(oof).all():
        raise ValueError('aligned finite development OOF estimates required')
    result = {}
    for dataset in ('pope','textvqa'):
        selected = np.asarray([r['dataset']==dataset for r in dev])
        y = np.asarray([r['label'] for r in dev])[selected]
        if len(np.unique(y)) < 2:
            raise ValueError('both correctness classes required in the development pilot')
        for variant, predictions in [('observed',None),('observed_views',None),('surrogate',oof)]:
            x,a = matrices(dev,predictions,views_only=variant!='observed');x,a=x[selected],a[selected]
            fill = np.asarray([x[a[:,j],j].mean() if a[:,j].any() else 0. for j in range(x.shape[1])])
            live = np.asarray([a[:,j].any() and np.ptp(x[a[:,j],j])>1e-12 for j in range(x.shape[1])])
            z = np.where(a,x,fill)
            model = RiskCalibrator(l2=.05).fit(z[:,live],y)
            if not model.success_:
                raise ValueError('development fusion failed to converge')
            result[dataset+'_'+variant] = dict(fill=fill.tolist(),live=live.tolist(),
                coef=model.coef_.tolist(),mean=model.mean_.tolist(),std=model.std_.tolist(),
                train_ids=[r['id'] for r in dev if r['dataset']==dataset],
                train_scope='development only; surrogate input is group-OOF',
                signals=['self_consistency','resample_entropy_estimate' if variant=='surrogate' else 'resample_consistency',
                    'typed_verification','cross_model_agreement','grounding_check',
                    'object_support_views' if variant!='observed' else 'object_support',
                    'sequence_confidence','output_entropy'])
    common.save(output/'matched_fusion_parameters.json', result)
    return result


def evaluate(output, dev, test, predictions, states, estimator):
    from pipeline.factual import ece
    from pipeline.factual_parameters import predict
    report = dict(complete=False, full_acceptance=False, test_used_for_training_or_selection=False,
        estimator_sha256=common.sha(output/'estimator.npz'),
        matched_fusion_parameters_sha256=common.sha(output/'matched_fusion_parameters.json'),
        scope='historical answer evidence with newly extracted hidden states; not fresh full inference',
        warning='entropy is estimated; object support uses views only; old frozen fusion must not be silently reused',
        datasets={})
    check_records(dev, development=True)
    check_records(dev+test)
    predictions = np.asarray(predictions)
    if (not test or any(r['acceptance_split'] != 'test' for r in test)
            or predictions.shape != (len(test),) or not np.isfinite(predictions).all()
            or (predictions<0).any() or (predictions>1).any()):
        raise ValueError('aligned held-out test estimates in [0,1] required')
    if any(set(state['train_ids']) & {r['id'] for r in test} for state in states.values()):
        raise ValueError('test IDs entered fusion fitting')
    for dataset in ('pope','textvqa'):
        mask = np.asarray([r['dataset']==dataset for r in test])
        rows = [r for r in test if r['dataset']==dataset]
        target = np.asarray([r['target'] for r in rows])
        pred = predictions[mask]
        constant = np.mean([r['target'] for r in dev if r['dataset']==dataset])
        mae = float(np.abs(target-pred).mean())
        entry = dict(n_test=len(rows), n_dev=sum(r['dataset']==dataset for r in dev),
            mae=mae, rmse=float(np.sqrt(np.mean((target-pred)**2))),
            constant_dev_mean=float(constant), constant_mae=float(np.abs(target-constant).mean()),
            target_distribution={str(float(t)):int((target==t).sum()) for t in np.unique(target)},
            estimated_entropy_min=float(pred.min()), estimated_entropy_max=float(pred.max()), variants={})
        y = np.asarray([r['label'] for r in rows])
        for variant, values in [('observed',None),('observed_views',None),('surrogate',pred)]:
            x,a=matrices(rows,values,views_only=variant!='observed')
            state=states[dataset+'_'+variant]
            scores=predict(state,x,a)
            before=ece(y,x[:,6]);after=ece(y,scores)
            # Coefficients cover live columns only, not the original eight columns.
            weights=np.zeros(8);weights[np.asarray(state['live'],dtype=bool)]=np.asarray(state['coef'])[1:]
            count=(a & (np.abs(weights)>1e-12)).sum(1)
            measured = a.copy()
            if variant=='surrogate': measured[:,1]=False
            entry['variants'][variant]=dict(ece_before=before,ece_after=after,
                relative_gain=(before-after)/before if before>0 else None,
                minimum_actual_signals=int(count.min()),
                minimum_observed_signals=int((measured & (np.abs(weights)>1e-12)).sum(1).min()),
                numerical_gain_at_least_10_percent=bool(before>0 and (before-after)/before>=.1),
                numerical_signal_count_at_least_3=bool(count.min()>=3))
        report['datasets'][dataset]=entry
    with (output/'test_predictions.jsonl').open('x',encoding='utf-8') as handle:
        for row,pred in zip(test,predictions):
            common.emit(handle,dict(id=row['id'],image_group=row['image_group'],dataset=row['dataset'],
                teacher_entropy=row['target'],estimated_entropy=float(pred)))
    report.update(complete=True, verdict='PILOT_COMPLETED_NOT_ACCEPTANCE')
    common.save(output/'report.json',report)
    lines=['# Entropy surrogate pilot','',report['scope'],'',report['warning'],'',
        '| Dataset | Dev | Test | MAE | Constant MAE | Original gain | Views-only control gain | Surrogate gain | Minimum signals (observed) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def gain(value):
        return 'undefined (zero baseline ECE)' if value is None else f'{value:.2%}'
    for name,item in report['datasets'].items():
        old,control,new=[item['variants'][v] for v in ('observed','observed_views','surrogate')]
        lines.append(f"| {name} | {item['n_dev']} | {item['n_test']} | {item['mae']:.4f} | {item['constant_mae']:.4f} | {gain(old['relative_gain'])} | {gain(control['relative_gain'])} | {gain(new['relative_gain'])} | {new['minimum_actual_signals']} ({new['minimum_observed_signals']}) |")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def benchmark(provider, dev, estimator, output, budget):
    from msrc.batch_collection import collect_chunk
    from msrc.providers.internvl_cache import InternVLInputCache
    selected=[]
    for dataset in ('pope','textvqa'):
        selected += [r for r in dev if r['dataset']==dataset][:4]
    results=[]
    for variant,k in [('observed',3),('surrogate',0)]:
        provider._batch_input_cache=InternVLInputCache(provider)
        work,idle=budget.work_seconds,budget.idle_seconds
        answers=[];start=time.perf_counter()
        for row in selected:
            raw={key:row[key] for key in ('id','image','question','gold_answers','answer_format','answer_parser') if key in row}
            if variant=='surrogate':
                with PromptStateCapture(provider) as capture:
                    result=collect_chunk(provider,[raw],batch_size=1,k=k,budget=budget)[0]
                value=float(estimator.predict(capture.features())[0])
                result['estimated_entropy']=value
            else:
                result=collect_chunk(provider,[raw],batch_size=1,k=k,budget=budget)[0]
            answers.append(result)
        item=dict(variant=variant,items=len(selected),generations_per_item=11 if k else 8,
            elapsed_seconds=time.perf_counter()-start,work_seconds=budget.work_seconds-work,
            voluntary_idle_seconds=budget.idle_seconds-idle,image_cache=provider._batch_input_cache.snapshot(),
            fresh_answers=True,auxiliary_donor_and_model_load_not_timed=True,shared_A100_not_5090=True)
        common.save(output/('benchmark_'+variant+'.json'),dict(timing=item,records=answers))
        results.append(item)
    common.save(output/'benchmark.json',dict(arms=results,
        wall_speedup=results[0]['elapsed_seconds']/results[1]['elapsed_seconds'],
        work_speedup=results[0]['work_seconds']/results[1]['work_seconds'],
        all_other_jobs_untouched=True,small_development_benchmark_only=True,
        cross_model_signal_missing_in_both_timing_arms=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--phase',choices=['prepare','run'],default='run')
    p.add_argument('--dev-per-dataset',type=int,default=256)
    p.add_argument('--test-per-dataset',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--cap-gib',type=int,default=19)
    args=p.parse_args();common.initialize();common.select_device()
    if args.batch_size<1:raise ValueError('positive batch size required')
    output=common.inside(common.runs_dir()/args.run_id/'entropy_estimator')
    if args.phase=='prepare':
        prepare(args.source.resolve(strict=True),output,args.dev_per_dataset,args.test_per_dataset);return
    protocol=json.loads((output/'protocol.json').read_text())
    for name,digest in protocol['sources'].items():
        if common.sha(common.ROOT/name)!=digest:raise ValueError('source changed after preparation: '+name)
    for split,digest in protocol['cohort_hashes'].items():
        if common.sha(output/(split+'.jsonl'))!=digest:raise ValueError('cohort changed')
    dev=common.read_rows(output/'dev.jsonl');test=common.read_rows(output/'test.jsonl')
    check_records(dev+test)
    import torch
    from pipeline.batch_training import BatchBudget,make_provider
    torch.set_num_threads(2)
    budget=BatchBudget(torch,args.cap_gib,shared_duty=.5)
    provider=make_provider('internvl',budget)
    contract=feature_contract(provider)
    common.save(output/'feature_contract.json',contract)
    x=extract(provider,dev,output,'dev',budget,args.batch_size)
    estimator,oof=train(output,dev,x,contract)
    estimator=ResampleEntropyEstimator.load(output/'estimator.npz',contract=contract)
    states=fit_fusions(output,dev,oof)
    frozen={name:common.sha(output/name) for name in
        ('estimator.npz','estimator.json','matched_fusion_parameters.json')}
    common.save(output/'frozen_before_test.json',dict(frozen, frozen_ns=time.time_ns()))
    xtest=extract(provider,test,output,'test',budget,args.batch_size)
    for name,digest in frozen.items():
        if common.sha(output/name)!=digest:raise ValueError('fitted parameters changed during testing')
    report=evaluate(output,dev,test,estimator.predict(xtest),states,estimator)
    benchmark(provider,dev,estimator,output,budget)
    common.save(output/'resource_usage.json',budget.snapshot())
    print('PILOT_COMPLETED_NOT_ACCEPTANCE '+str(output/'report.md'),flush=True)


if __name__=='__main__':
    main()
