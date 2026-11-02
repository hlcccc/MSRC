"""Full historical-evidence CPU control; not fresh inference or acceptance."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from pipeline import common
from msrc.entropy_estimator import check_records, development_folds
from msrc.primary_entropy_estimator import (FEATURES, FEATURE_RULE,
    PrimaryStatisticsEntropyEstimator, primary_features)
from pipeline.entropy_estimator import fit_fusions, matrices


def prepare(source):
    from pipeline.factual_parameters import features
    from msrc.signals import Evidence, ObjectSupport
    rows, vectors, hashes = [], [], {}
    for dataset in ('pope', 'textvqa'):
        raw, evidence, readings, x, available, y, _ = features(source, 'internvl', dataset)
        for index, (record, item) in enumerate(zip(raw, evidence)):
            ev = Evidence.from_dict(item['evidence'])
            if len(ev.resamples)!=3 or len(ev.views)!=4 or len(ev.verifications)!=3:
                raise ValueError('full K=3/four-view/three-probe teacher evidence required')
            if not readings[index]['resample_consistency'].available:
                raise ValueError('an available measured entropy target is required')
            image = record['image']
            if image not in hashes: hashes[image] = common.sha(image)
            support = ObjectSupport().compute(replace(ev, resamples=[]))
            rows.append(dict(id=record['id'], image_group=hashes[image], dataset=dataset,
                acceptance_split=record['acceptance_split'], label=int(y[index]),
                target=float(readings[index]['resample_consistency'].risk),
                teacher_risks=x[index].tolist(), available=available[index].tolist(),
                object_support_views_risk=support.risk, object_support_views_available=support.available))
            vectors.append(primary_features(ev.primary, dataset))
    check_records(rows)
    return rows, np.asarray(vectors)


def train(output, records, x, contract):
    check_records(records, development=True)
    target = np.asarray([r['target'] for r in records])
    folds = development_folds(records)
    oof = np.zeros(len(records)); audit = []
    for fold in range(4):
        mask = folds!=fold
        fitted = [r for r,yes in zip(records,mask) if yes]
        validation = [r for r,yes in zip(records,mask) if not yes]
        model = PrimaryStatisticsEntropyEstimator().fit(x[mask], target[mask], fitted)
        oof[~mask] = model.predict(x[~mask])
        groups = {r['image_group'] for r in fitted}
        if groups & {r['image_group'] for r in validation}: raise ValueError('OOF leakage')
        audit.append(dict(fold=fold, train_ids=[r['id'] for r in fitted],
                          validation_ids=[r['id'] for r in validation]))
    final = PrimaryStatisticsEntropyEstimator().fit(x,target,records)
    final.save(output/'estimator.npz', contract)
    restored = PrimaryStatisticsEntropyEstimator.load(output/'estimator.npz',contract=contract)
    if not np.allclose(final.predict(x),restored.predict(x)): raise ValueError('export changed predictions')
    states = fit_fusions(output,records,oof)
    for state in states.values():
        state['signals'] = ['resample_entropy_primary_stats' if n=='resample_entropy_estimate' else n
                            for n in state['signals']]
        state['entropy_input_rule'] = FEATURE_RULE
    common.save(output/'matched_fusion_parameters.json',states)
    np.savez_compressed(output/'development_oof.npz',predictions=oof,folds=folds)
    common.save(output/'crossfit_audit.json',dict(folds=audit,test_used_for_fitting=False,
        dev_ids=[r['id'] for r in records],base_model_trained=False))
    return restored, states


def evaluate(dev,test,prediction,states):
    from pipeline.factual import ece
    from pipeline.factual_parameters import predict
    check_records(dev,development=True);check_records(dev+test)
    prediction = np.asarray(prediction,dtype=float)
    if (not test or any(r['acceptance_split']!='test' for r in test)
            or prediction.shape!=(len(test),) or not np.isfinite(prediction).all()
            or (prediction<0).any() or (prediction>1).any()):
        raise ValueError('aligned held-out test estimates in [0,1] required')
    result = {}
    for dataset in ('pope','textvqa'):
        selected = np.asarray([r['dataset']==dataset for r in test])
        rows = [r for r in test if r['dataset']==dataset]
        target = np.asarray([r['target'] for r in rows]); pred = prediction[selected]
        dev_target = np.asarray([r['target'] for r in dev if r['dataset']==dataset])
        if not len(dev_target) or not len(rows):
            raise ValueError('both datasets require development and held-out examples')
        constant = float(dev_target.mean())
        mse = float(np.mean((target-pred)**2)); constant_mse = float(np.mean((target-constant)**2))
        mae = float(np.mean(np.abs(target-pred))); constant_mae = float(np.mean(np.abs(target-constant)))
        item = dict(n_dev=len(dev_target),n_test=len(rows),mae=mae,rmse=float(np.sqrt(mse)),
            constant_dev_mean=constant,constant_mae=constant_mae,constant_rmse=float(np.sqrt(constant_mse)),
            zero_predictor_mae=float(np.abs(target).mean()),
            mse_gain_over_constant=(constant_mse-mse)/constant_mse if constant_mse>0 else None,
            better_than_constant_mae=mae<constant_mae,better_than_constant_mse=mse<constant_mse,
            teacher_counts={str(float(v)):int((target==v).sum()) for v in np.unique(target)},variants={})
        y = np.asarray([r['label'] for r in rows])
        for variant,values in [('observed',None),('observed_views',None),('surrogate',pred)]:
            x,a = matrices(rows,values,views_only=variant!='observed')
            state = states[dataset+'_'+variant]
            if set(state['train_ids']) & {r['id'] for r in test}: raise ValueError('fusion leakage')
            scores = predict(state,x,a)
            before,after = ece(y,x[:,6]),ece(y,scores)
            weights = np.zeros(8);weights[np.asarray(state['live'],dtype=bool)] = np.asarray(state['coef'])[1:]
            active = a & (np.abs(weights)>1e-12)
            measured = active.copy()
            if variant=='surrogate': measured[:,1]=False
            gain = (before-after)/before if before>0 else None
            item['variants'][variant] = dict(ece_before=before,ece_after=after,relative_gain=gain,
                minimum_signals=int(active.sum(1).min()),minimum_observed_signals=int(measured.sum(1).min()),
                numerical_gain_at_least_10_percent=bool(gain is not None and gain>=.1),
                numerical_signal_count_at_least_3=bool(active.sum(1).min()>=3))
        result[dataset] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    args = parser.parse_args();common.initialize()
    source = args.source.resolve(strict=True)
    output = common.inside(common.runs_dir()/args.run_id/'primary_entropy_control')
    output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    origins = {name:common.sha(source/name) for name in ('protocol.json','inputs/pope.jsonl',
        'inputs/textvqa.jsonl','evidence/internvl_pope.jsonl','evidence/internvl_textvqa.jsonl',
        'evidence/qwen_pope.jsonl','evidence/qwen_textvqa.jsonl')}
    rows,x = prepare(source)
    is_dev = np.asarray([r['acceptance_split']=='dev' for r in rows])
    dev,test = [r for r in rows if r['acceptance_split']=='dev'],[r for r in rows if r['acceptance_split']=='test']
    contract = dict(feature_rule=FEATURE_RULE,features=list(FEATURES),source_sha256=origins,
                    base_model='original frozen BF16 InternVL3-8B-hf',max_new_tokens=128,k_teacher=3)
    common.save(output/'protocol.json',dict(contract=contract,source=str(source),
        sources={name:common.sha(common.ROOT/name) for name in ('msrc/primary_entropy_estimator.py',
            'pipeline/primary_entropy_pilot.py','msrc/entropy_estimator.py','pipeline/entropy_estimator.py',
            'msrc/signals.py','msrc/model.py','pipeline/factual_parameters.py')},
        image_groups_cross_split=0,hyperparameters_fixed_before_test=dict(alpha=10.,projection=64,folds=4),
        test_used_for_selection=False,fresh_inference=False,full_acceptance=False,created_ns=time.time_ns()))
    model,states = train(output,dev,x[is_dev],contract)
    frozen = {name:common.sha(output/name) for name in
              ('estimator.npz','estimator.json','matched_fusion_parameters.json')}
    common.save(output/'frozen_before_test.json',dict(files=frozen,time_ns=time.time_ns()))
    prediction_start = time.perf_counter();prediction = model.predict(x[~is_dev])
    prediction_seconds = time.perf_counter()-prediction_start
    metrics = evaluate(dev,test,prediction,states)
    if any(common.sha(output/name)!=digest for name,digest in frozen.items()): raise ValueError('parameters changed')
    if any(common.sha(source/name)!=digest for name,digest in origins.items()):
        raise ValueError('historical evidence changed during the control experiment')
    with (output/'test_predictions.jsonl').open('x',encoding='utf-8') as handle:
        for row,value in zip(test,prediction):common.emit(handle,dict(row,entropy_estimate=float(value)))
    report = dict(verdict='HISTORICAL_PRIMARY_STATISTICS_CONTROL_ONLY',full_acceptance=False,
        fresh_inference=False,decoder_hidden_states_used=False,base_model_trained=False,
        scope='full historical evidence; primary-summary entropy estimator, not prompt-hidden-state student',
        source=str(source),image_groups_cross_split=0,dev_count=len(dev),test_count=len(test),datasets=metrics,
        head_prediction_seconds=prediction_seconds,entire_cpu_study_seconds=time.perf_counter()-started,
        runtime_speedup_measured=False,original_generation_count=11,proposed_generation_count=8,
        test_used_for_fitting_or_selection=False)
    common.save(output/'report.json',report)
    lines = ['# Primary-statistics entropy control','',report['scope'],
        'Historical evidence analysis only. No new VLM inference, hidden-state extraction or runtime speedup measurement.','',
        '| Dataset | Dev | Test | MAE | Constant MAE | Original ECE gain | Views-only control gain | Estimated ECE gain | Minimum signals |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def gain(value): return 'undefined' if value is None else f'{value:.2%}'
    for ds,item in metrics.items():
        old,control,new = [item['variants'][v] for v in ('observed','observed_views','surrogate')]
        lines.append(f"| {ds} | {item['n_dev']} | {item['n_test']} | {item['mae']:.5f} | {item['constant_mae']:.5f} | {gain(old['relative_gain'])} | {gain(control['relative_gain'])} | {gain(new['relative_gain'])} | {new['minimum_signals']} |")
    lines += ['', 'A passing downstream numerical gain does not establish faithful entropy prediction.',
        'The original frozen fusion is not reused. All student/fusion fits and normalization are development-only.',
        'The separate BF16 prompt-hidden-state pilot remains an independent experiment.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(report['verdict']+' '+str(output/'report.md'),flush=True)


if __name__=='__main__':main()
