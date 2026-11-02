"""Fresh eight-generation collector using the frozen primary-statistics head."""
from __future__ import annotations

import numpy as np

from msrc.batch_collection import collect_chunk
from msrc.primary_entropy_estimator import primary_features
from msrc.signals import Evidence, build_report
from msrc.types import Sample

SIGNALS = ('self_consistency','resample_entropy_primary_stats','typed_verification',
           'cross_model_agreement','grounding_check','object_support_views',
           'sequence_confidence','output_entropy')


def collect_fast_chunk(provider,rows,*,dataset,estimator,contract,batch_size,budget=None,
                       adaptive=True):
    """Skip sampled generations, never fabricate sampled-answer evidence."""
    estimator.assert_contract(contract)
    if contract.get('max_new_tokens')!=provider.max_new_tokens:
        raise ValueError('primary generation budget differs from the trained feature contract')
    if dataset not in {'pope','textvqa'}:
        raise ValueError('explicit supported dataset required')
    result = collect_chunk(provider,rows,batch_size=batch_size,k=0,budget=budget,adaptive=adaptive)
    x = np.asarray([primary_features(Evidence.from_dict(row['evidence']).primary,dataset)
                    for row in result])
    estimates = estimator.predict(x)
    for row,value in zip(result,estimates):
        ev = Evidence.from_dict(row['evidence'])
        if ev.resamples or len(ev.views)!=4 or len(ev.verifications)!=3:
            raise ValueError('the fast variant requires four views, three probes and no resamples')
        row.update(generation_count=8,resample_generation_count=0,
            entropy_prediction=dict(name='resample_entropy_primary_stats',risk=float(value),
                kind='internal_estimate',observed_resampling=False,
                weights_sha256=estimator.metadata['weights_sha256']),
            object_support_rule='four rephrasing views only; no resampling evidence')
    return result


def signal_matrix(records,donors=None):
    """Build the matching fusion inputs; unavailable donors stay unavailable."""
    donors = {} if donors is None else {row['id']:row for row in donors}
    values,available = [],[]
    for row in records:
        ev = Evidence.from_dict(row['evidence'])
        if ev.resamples:
            raise ValueError('fast evidence must not contain observed resamples')
        donor = donors.get(row['id'])
        if donor is not None:
            if donor['question']!=row['question']:
                raise ValueError('the independent donor must answer the identical question')
            ev.cross_model = [Sample(text=donor['answer'])]
        readings = {value.name:value for value in build_report(ev).signals}
        order = ('self_consistency','resample_consistency','typed_verification',
                 'cross_model_agreement','grounding_check','object_support',
                 'sequence_confidence','output_entropy')
        x = [readings[name].risk for name in order]
        a = [readings[name].available for name in order]
        prediction = row['entropy_prediction']
        if prediction.get('observed_resampling') is not False or prediction.get('kind')!='internal_estimate':
            raise ValueError('the predicted signal must be explicitly marked estimated')
        value = prediction['risk']
        if not np.isfinite(value) or not 0<=value<=1:
            raise ValueError('finite entropy estimate in [0,1] required')
        x[1],a[1] = value,True
        values.append(x);available.append(a)
    return np.asarray(values,dtype=float),np.asarray(available,dtype=bool)
