"""Recollect POPE/TextVQA answers and eight signals; fit only on development."""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline.common import (inside, readable, sha, save, read_rows, emit,
                             initialize, select_device, SEED, model_paths, data_paths)

BASE = Path(__file__).resolve().parent.parent
SOURCE = BASE
sys.path.insert(0, str(BASE))
SIGNALS = ['self_consistency','resample_consistency','typed_verification',
           'cross_model_agreement','grounding_check','object_support',
           'sequence_confidence','output_entropy']
MODELS = {name: model_paths()[name] for name in ('llava', 'qwen', 'internvl')}
DONORS = {'llava':'qwen','qwen':'llava','internvl':'qwen'}
INPUTS = data_paths()['factual']


def _source_files(collector='scripts/collect_evidence.py'):
    """The files a re-run must reproduce: the package, the collection entry point and
    the official TextVQA scorer. Hashing the whole tree would also cover this file."""
    files = list((SOURCE / 'msrc').rglob('*.py')) + list((SOURCE / 'evaluation').rglob('*.py'))
    files += [SOURCE / collector,
              SOURCE / 'references' / 'm4c_evaluator.py', SOURCE / 'pipeline' / 'common.py']
    return sorted(set(files))


def ece(y, p, n=15):
    import numpy as np
    y, p = np.asarray(y), np.asarray(p)
    if not len(y) or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError('invalid ECE input')
    if not np.isin(y,[0,1]).all() or ((p<0)|(p>1)).any():
        raise ValueError('invalid binary label/probability')
    bins = np.minimum((p*n).astype(int), n-1)
    return sum(abs(float(y[bins==i].sum()-p[bins==i].sum())) for i in range(n))/len(y)


def bin_stats(y,p,n=15):
    import numpy as np
    y,p=np.asarray(y),np.asarray(p)
    assigned=np.minimum((p*n).astype(int),n-1)
    return [dict(bin=i,n=int((assigned==i).sum()),
                 mean_predicted_risk=float(p[assigned==i].mean()) if (assigned==i).any() else None,
                 observed_error_rate=float(y[assigned==i].mean()) if (assigned==i).any() else None)
            for i in range(n)]


def prepare(out, mode, parameters=None):
    import numpy as np
    from pipeline.factual_parameters import fitted_collector
    from evaluation.groups import image_group_id
    from evaluation.split import grouped_split
    collector = fitted_collector() if parameters else 'scripts/collect_evidence.py'
    protocol = dict(seed=SEED, mode=mode, max_new_tokens=128, k=3,
                    dtype='bfloat16',
                    views=4, probes=3, resample_temperature=.7, top_p=.9,
                    max_pixels=1003520, attention=False, image_contrast=False,
                    selected_signals=SIGNALS, models=MODELS, donors=DONORS,
                    donor_prompt='fresh independent model primary answer for the identical input question',
                    baseline='uncalibrated sequence_confidence risk=1-mean_token_probability',
                    textvqa_labels='official MMF/M4C soft score < 0.5',
                    l2=.05, bins=15, missing='development available-value mean',
                    scope='full acceptance' if mode=='full' else 'engineering smoke; NOT full acceptance',
                    source_hashes={p.relative_to(SOURCE).as_posix(): sha(p) for p in _source_files(collector)},
                    collector_entry=collector, collector_sha256=sha(SOURCE / collector),
                    script_sha256=sha(__file__), created_ns=time.time_ns(),
                    parameter_helper_sha256=sha(Path(__file__).with_name('factual_parameters.py')),
                    parameter_mode='provided' if parameters else 'fresh_fit')
    inputs_hashes = {}
    for dataset, expected in [('pope',9000),('textvqa',5000)]:
        rows = read_rows(INPUTS / (dataset+'.jsonl'))
        if len(rows) != expected or any(r.get('answer') or r.get('evidence') for r in rows):
            raise ValueError('complete raw inputs without answers/evidence required')
        groups = np.asarray([image_group_id(r) for r in rows])
        dev,test = grouped_split(groups, seed=SEED, dev_fraction=.5)
        if set(groups[dev]) & set(groups[test]):
            raise ValueError('image identity crosses split')
        if mode == 'smoke':
            # Selection is based solely on raw input order and prespecified split.
            indices = sorted([int(side[i]) for side in (np.flatnonzero(dev),np.flatnonzero(test))
                              for i in np.linspace(0,len(side)-1,16,dtype=int)])
        else:
            indices = list(range(len(rows)))
        chosen = []
        for i in indices:
            row = dict(rows[i], acceptance_split='dev' if dev[i] else 'test',
                       acceptance_image_group=str(groups[i]))
            if not readable(row['image']).is_file():
                raise ValueError('missing raw image')
            chosen.append(row)
        p = inside(out/'inputs'/(dataset+'.jsonl'))
        p.parent.mkdir(exist_ok=True)
        with p.open('x',encoding='utf-8') as f:
            for row in chosen:
                emit(f,row)
        inputs_hashes[dataset] = dict(sha256=sha(p), n=len(chosen), full_n=expected,
                                     dev=sum(r['acceptance_split']=='dev' for r in chosen),
                                     test=sum(r['acceptance_split']=='test' for r in chosen))
    protocol['inputs'] = inputs_hashes
    from pipeline.gpu_sharing import load_shared_policy
    sharing = load_shared_policy(workload='factual')
    if sharing and parameters:
        raise ValueError('the source-frozen fitted collector does not implement shared-GPU duty control; use an unshared GPU')
    if sharing:
        shutil.copyfile(sharing['source_path'], inside(out/'shared_gpu_policy.json'))
        protocol.update(shared_gpu_execution=True,
                        shared_gpu_policy_sha256=sha(out/'shared_gpu_policy.json'),
                        gpu_resource_source_sha256=sha(SOURCE/'pipeline/gpu_sharing.py'))
    if parameters:
        from pipeline.factual_parameters import load_package
        parameters = readable(parameters)
        load_package(parameters, out, protocol)
        shutil.copyfile(parameters, inside(out/'provided_parameters.json'))
        protocol['parameters_sha256'] = sha(out/'provided_parameters.json')
        protocol['parameters_frozen_ns'] = time.time_ns()
    save(out/'protocol.json',protocol)
    print('FACTUAL_PREPARED '+mode, flush=True)


def collect(out):
    from pipeline.factual_parameters import fitted_collector
    from pipeline.preflight import wait_for_gpu
    from pipeline.gpu_sharing import load_shared_policy
    protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
    collector = protocol.get('collector_entry', 'scripts/collect_evidence.py')
    expected = fitted_collector() if protocol.get('parameter_mode') == 'provided' else 'scripts/collect_evidence.py'
    if collector != expected or sha(SOURCE / collector) != protocol.get('collector_sha256'):
        raise ValueError('factual collector does not match the frozen run protocol')
    for family,path in MODELS.items():
        for dataset in ('pope','textvqa'):
            device = wait_for_gpu()
            output = inside(out/'evidence'/(family+'_'+dataset+'.jsonl'))
            output.parent.mkdir(exist_ok=True)
            if output.exists():
                raise ValueError('fresh output must not exist; cached evidence is refused')
            command = [sys.executable,'-B','-u',str(SOURCE/collector),
                       '--data',str(out/'inputs'/(dataset+'.jsonl')),
                       '--out',str(output),'--provider',family,'--model-path',path,
                       '--device','cuda:0','--dtype','bfloat16',
                       '--max-new-tokens','128','--max-pixels','1003520',
                       '--k','3','--seed',str(SEED),
                       '--save-every','4']
            sharing = load_shared_policy(workload='factual')
            if sharing:
                fraction = sharing['allocator_cap_gib'] * 1024 / device[0]['total_mib']
                command += ['--gpu-memory-fraction', repr(fraction)]
            log = inside(out/'logs'/(family+'_'+dataset+'.log'))
            log.parent.mkdir(exist_ok=True)
            print('COLLECT_START '+family+' '+dataset, flush=True)
            with log.open('x',encoding='utf-8') as f:
                result = subprocess.run(command, cwd=SOURCE, stdout=f, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('fresh collection failed; inspect '+str(log))
            print('COLLECT_COMPLETE '+family+' '+dataset, flush=True)


def evaluate(out):
    import numpy as np
    from msrc.model import RiskCalibrator
    from evaluation.metrics import auroc
    from pipeline.factual_parameters import features, load_package, predict, warning_threshold
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    package = None
    if protocol.get('parameter_mode') == 'provided':
        if sha(out/'provided_parameters.json') != protocol['parameters_sha256']:
            raise ValueError('provided parameters changed after preparation')
        package = load_package(out/'provided_parameters.json', out, protocol)
    results = []
    for family in MODELS:
        for dataset in ('pope','textvqa'):
            raw, rows, readings, x, a, y, dev = features(out, family, dataset)
            test = ~dev
            cell = family+'_'+dataset
            if package:
                model_state = package['states'][cell]
                fill, live = np.asarray(model_state['fill']), np.asarray(model_state['live'], dtype=bool)
                score = predict(model_state, x[test], a[test])
                x = np.where(a, x, fill)
            else:
                fill = np.asarray([x[dev & a[:,j],j].mean() if a[dev,j].any() else 0.
                                   for j in range(len(SIGNALS))])
                x = np.where(a,x,fill)
                live = np.asarray([a[dev,j].any() and
                                   np.ptp(x[dev & a[:,j],j])>1e-12 for j in range(len(SIGNALS))])
                if not live.any():
                    raise ValueError('no varying available development features')
                fit = RiskCalibrator(l2=.05).fit(x[dev][:,live],y[dev])
                if not fit.success_:
                    raise ValueError('calibrator failed to converge')
                score = fit.predict_proba(x[test][:,live])
                model_state = dict(signals=SIGNALS, live=live.tolist(),fill=fill.tolist(),
                                   coef=fit.coef_.tolist(), mean=fit.mean_.tolist(),
                                   std=fit.std_.tolist(), train_ids=[r['id'] for r,d in zip(raw,dev) if d],
                                   converged=True, iterations=fit.n_iter_,gradient=fit.grad_norm_,
                                   fitting_scope='development only',
                                   warning_threshold=warning_threshold(fit.predict_proba(x[dev][:,live]),y[dev]))
            baseline_index = SIGNALS.index('sequence_confidence')
            if not a[test,baseline_index].all():
                raise ValueError('raw token-confidence baseline unavailable')
            before = ece(y[test],x[test,baseline_index])
            after = ece(y[test],score)
            weights = np.zeros(len(SIGNALS))
            weights[live] = model_state['coef'][1:]
            save(out/'calibrators'/(cell+'.json'),model_state)
            details = inside(out/'signal_usage'/(cell+'.jsonl'))
            details.parent.mkdir(exist_ok=True)
            min_used = len(SIGNALS)
            csv_rows=[]
            with details.open('x',encoding='utf-8') as f:
                for index,pred in zip(np.flatnonzero(test),score):
                    signal_rows = []
                    for j,n in enumerate(SIGNALS):
                        signal = readings[index][n]
                        used = bool(signal.available and live[j] and abs(weights[j])>1e-12)
                        signal_rows.append(dict(name=n,risk=signal.risk,available=signal.available,
                                                participates=used,weight=float(weights[j]),detail=signal.detail,
                                                imputed_value=float(x[index,j]) if not signal.available else None))
                    count = sum(s['participates'] for s in signal_rows)
                    min_used = min(min_used,count)
                    emit(f,dict(id=rows[index]['id'],answer=rows[index]['answer'],
                                label=int(y[index]),risk=float(pred),signals=signal_rows,
                                n_actual_signals=count))
                    for s in signal_rows:
                        csv_rows.append(dict(id=rows[index]['id'],signal=s['name'],
                                             raw_risk=s['risk'],available=s['available'],
                                             participates=s['participates'],weight=s['weight'],
                                             final_fusion_risk=float(pred),reason=s['detail']))
            with details.with_suffix('.csv').open('x',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
            result = dict(cell=cell,n=len(rows),n_dev=int(dev.sum()),n_test=int(test.sum()),
                          group_overlap=0,ece_before=before,ece_after=after,
                          relative_improvement=(before-after)/before if before else None,
                          minimum_actual_signals_per_test_sample=min_used,
                          calibration_pass=bool(before and (before-after)/before>=.10),
                          signals_pass=min_used>=3,
                          auroc_before=float(auroc(y[test],x[test,baseline_index])) if len(np.unique(y[test]))>1 else None,
                          auroc_after=float(auroc(y[test],score)) if len(np.unique(y[test]))>1 else None,
                          bins_before=bin_stats(y[test],x[test,baseline_index]),
                          bins_after=bin_stats(y[test],score),
                          availability_rate={n:float(a[test,j].mean()) for j,n in enumerate(SIGNALS)},
                          ece_sensitivity={str(n):dict(before=ece(y[test],x[test,baseline_index],n),
                                                     after=ece(y[test],score,n)) for n in (5,10,15,20)},
                          new_evidence_sha256=sha(out/'evidence'/(cell+'.jsonl')))
            results.append(result)
            print(json.dumps({k:result[k] for k in ('cell','n_dev','n_test','ece_before',
                             'ece_after','relative_improvement','minimum_actual_signals_per_test_sample')}),flush=True)
    full = protocol['mode']=='full'
    report = dict(status='FULL_FACTUAL_FLOW_COMPLETE' if full else 'SMOKE_ONLY_NOT_FULL_ACCEPTANCE',
                  mode=protocol['mode'],cells=results,old_evidence_used=False,
                  parameter_mode=protocol.get('parameter_mode','fresh_fit'),
                  parameters_sha256=protocol.get('parameters_sha256'),
                  acceptance_conclusion=all(r['calibration_pass'] and r['signals_pass'] for r in results) if full else None)
    save(out/'report.json',report)
    with inside(out/'summary.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['cell','n_dev','n_test','ece_before','ece_after',
                                           'relative_improvement','minimum_actual_signals_per_test_sample'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in results)
    print(report['status'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['full','smoke'],default='full')
    p.add_argument('--phase',choices=['prepare','collect','evaluate','run'],default='run')
    p.add_argument('--parameters',type=Path,help='provided fitted factual parameters; never refit')
    args=p.parse_args()
    initialize()
    select_device()
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    out=inside(args.out)
    if args.phase in ('prepare','run'):
        out.mkdir(parents=True,exist_ok=False)
        prepare(out,args.mode,args.parameters)
    else:
        if args.parameters:
            p.error('--parameters is set only during prepare/run; later phases use the frozen copy')
        protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        if protocol['mode']!=args.mode or protocol['script_sha256']!=sha(__file__):
            raise ValueError('frozen run protocol mismatch')
        if protocol.get('parameter_helper_sha256') != sha(Path(__file__).with_name('factual_parameters.py')):
            raise ValueError('frozen factual evaluation/parameter helper changed')
        for name,digest in protocol['source_hashes'].items():
            if sha(SOURCE/name)!=digest:
                raise ValueError('collection source changed after preparation')
        for name,state in protocol['inputs'].items():
            if sha(out/'inputs'/(name+'.jsonl'))!=state['sha256']:
                raise ValueError('input changed after preparation')
    if args.phase in ('collect','run'):
        collect(out)
    if args.phase in ('evaluate','run'):
        evaluate(out)


if __name__=='__main__':
    main()
