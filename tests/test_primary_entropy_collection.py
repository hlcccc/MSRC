import numpy as np
import pytest

from msrc.primary_entropy_collection import collect_fast_chunk,signal_matrix
from msrc.primary_entropy_estimator import PrimaryStatisticsEntropyEstimator
from msrc.providers.hf_vision import HFInternVLProvider
from msrc.signals import Evidence
from msrc.types import Sample


def fixture(tmp_path):
    rng=np.random.RandomState(27)
    records=[dict(id=str(i),image_group=str(i//2),acceptance_split='dev') for i in range(40)]
    model=PrimaryStatisticsEntropyEstimator().fit(rng.normal(size=(40,7)),rng.uniform(0,1,40),records)
    contract={'max_new_tokens':128}
    model.save(tmp_path/'head.npz',contract)
    provider=HFInternVLProvider(max_new_tokens=128,model_fn=lambda image,prompt,sampled: Sample(text='yes',
        sequence_confidence=.8,sequence_entropy=.4,first_token_margin=.7,
        meta={'generated':4,'prompt_len':20}))
    provider.ocr=lambda image:['cat']
    rows=[dict(id=str(i),image='/image',question='Is there a cat?',gold_answers=['yes'],
        answer_format='yes_no',answer_parser='pope_official') for i in range(3)]
    return provider,model,contract,rows


def test_fast_path_really_skips_three_generations_per_question(tmp_path):
    provider,model,contract,rows=fixture(tmp_path)
    result=collect_fast_chunk(provider,rows,dataset='pope',estimator=model,contract=contract,batch_size=2)
    assert provider.calls==24
    assert all(row['generation_count']==8 and row['resample_generation_count']==0 for row in result)
    for row in result:
        ev=Evidence.from_dict(row['evidence'])
        assert len(ev.views)==4 and len(ev.verifications)==3 and ev.resamples==[]
        assert row['entropy_prediction']['observed_resampling'] is False
    x,a=signal_matrix(result)
    assert x.shape==(3,8) and a[:,1].all() and not a[:,3].any()
    donors=[dict(row,answer='yes') for row in rows]
    _,with_donor=signal_matrix(result,donors)
    assert with_donor[:,3].all()


def test_fast_path_refuses_budget_mismatch_before_generation(tmp_path):
    provider,model,contract,rows=fixture(tmp_path)
    provider.max_new_tokens=64
    with pytest.raises(ValueError,match='generation budget'):
        collect_fast_chunk(provider,rows,dataset='pope',estimator=model,contract=contract,batch_size=2)
    assert provider.calls==0


def test_timing_summary_requires_paired_arms():
    from pipeline.primary_entropy_speed import summarize
    values=[dict(dataset=d,variant=v,elapsed_seconds=t,work_seconds=t/2,items=4)
            for d in ('pope','textvqa') for v,t in (('observed',11.),('primary_stats',8.))]
    report=summarize(values)
    assert report['pope']['wall_speedup']==pytest.approx(11/8)
    with pytest.raises(ValueError,match='paired complete'):summarize(values[:-1])
