import json

import numpy as np
import pytest

from msrc.entropy_estimator import ResampleEntropyEstimator
from msrc.primary_entropy_estimator import (FEATURES,PrimaryStatisticsEntropyEstimator,primary_features)
from msrc.types import Sample


def rows(n=40,split='dev'):
    return [dict(id=f'{split}:{i}',image_group=f'{split}:group:{i//2}',acceptance_split=split)
            for i in range(n)]


def test_only_primary_features_and_missing_values_rejected():
    sample=Sample(text='a fresh answer',sequence_confidence=.8,sequence_entropy=.2,
        first_token_margin=.7,meta=dict(generated=4,prompt_len=10))
    value=primary_features(sample,'pope')
    assert value.shape==(len(FEATURES),) and value[-1]==0.
    assert primary_features(sample,'textvqa')[-1]==1.
    sample.first_token_margin=None
    with pytest.raises(ValueError,match='primary scores'):primary_features(sample,'pope')


def test_primary_artifact_cannot_be_mistaken_for_hidden_student(tmp_path):
    rng=np.random.RandomState(9);x=rng.normal(size=(40,len(FEATURES)))
    y=np.clip(.5+.1*x[:,0],0,1)
    model=PrimaryStatisticsEntropyEstimator().fit(x,y,rows())
    path=tmp_path/'estimator.npz';contract={'source':'test fixture'}
    model.save(path,contract)
    restored=PrimaryStatisticsEntropyEstimator.load(path,contract=contract)
    assert np.allclose(model.predict(x),restored.predict(x))
    assert restored.metadata['hidden_states_used'] is False
    assert 'hidden_dimension' not in restored.metadata
    with pytest.raises(ValueError):ResampleEntropyEstimator.load(path)
    with pytest.raises(ValueError):PrimaryStatisticsEntropyEstimator.load(path,contract={'source':'other'})


def test_primary_fit_rejects_test_data_and_feature_dimension():
    with pytest.raises(ValueError,match='test data'):
        PrimaryStatisticsEntropyEstimator().fit(np.ones((40,len(FEATURES))),np.zeros(40),rows(split='test'))
    with pytest.raises(ValueError,match='feature order'):
        PrimaryStatisticsEntropyEstimator().fit(np.ones((40,2)),np.zeros(40),rows())


def test_cpu_control_uses_group_oof_and_new_fusion(tmp_path,monkeypatch):
    from pipeline import common
    from pipeline.primary_entropy_pilot import train,evaluate
    monkeypatch.setattr(common,'ROOT',tmp_path)
    rng=np.random.RandomState(20);dev,test=rows(80),rows(40,'test')
    for part in (dev,test):
        for i,row in enumerate(part):
            signals=rng.uniform(0,1,8)
            row.update(dataset='pope' if i%4<2 else 'textvqa',label=i%2,target=float(signals[1]),
                teacher_risks=signals.tolist(),available=[True]*8,object_support_views_risk=.1,
                object_support_views_available=True)
    model,states=train(tmp_path,dev,rng.normal(size=(80,len(FEATURES))),{'fixture':True})
    report=evaluate(dev,test,model.predict(rng.normal(size=(40,len(FEATURES)))),states)
    assert set(report)=={'pope','textvqa'}
    audit=json.loads((tmp_path/'crossfit_audit.json').read_text())
    assert audit['test_used_for_fitting'] is False
    for state in states.values():
        assert not set(state['train_ids']) & {r['id'] for r in test}
    for invalid in (np.zeros(39),np.full(40,np.nan),np.full(40,1.1)):
        with pytest.raises(ValueError,match='held-out test estimates'):
            evaluate(dev,test,invalid,states)
