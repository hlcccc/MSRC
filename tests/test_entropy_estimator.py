from types import SimpleNamespace

import numpy as np
import pytest

from msrc.entropy_estimator import (PromptStateCapture, ResampleEntropyEstimator,
    check_records, development_folds, feature_contract, generate_with_entropy)
from pipeline.entropy_estimator import cohort, matrices, train, fit_fusions, evaluate, extract


def records(n=64, split='dev'):
    return [dict(id=f'{split}:{i}',image_group=f'{split}:image:{i//2}',
                 acceptance_split=split) for i in range(n)]


def test_fit_is_development_only():
    with pytest.raises(ValueError, match='test data'):
        ResampleEntropyEstimator().fit(np.ones((4,3)),np.zeros(4),records(4,'test'))


def test_image_leak_and_duplicate_ids_rejected():
    dev,test=records(4),records(4,'test')
    test[0]['image_group']=dev[0]['image_group']
    with pytest.raises(ValueError,match='crosses'):
        check_records(dev+test)
    with pytest.raises(ValueError,match='unique'):
        check_records(dev+dev)


def test_grouped_crossfit():
    rows=records();folds=development_folds(rows)
    assert len(set(folds))==4
    for i in range(4):
        fit={r['image_group'] for r,f in zip(rows,folds) if f!=i}
        val={r['image_group'] for r,f in zip(rows,folds) if f==i}
        assert not fit & val
    assert np.array_equal(folds,development_folds(rows))


def test_regression_export_contract_and_no_test_normalization(tmp_path):
    rng=np.random.RandomState(8)
    x=rng.normal(size=(128,12));y=np.clip(.5+.15*x[:,0]-.1*x[:,1],0,1)
    model=ResampleEntropyEstimator(dimension=12,alpha=.01).fit(x,y,records(128))
    assert np.abs(model.predict(x)-y).mean()<.03
    path=tmp_path/'estimator.npz';contract={'model_sha':'a','dtype':'bf16'}
    model.save(path,contract)
    restored=ResampleEntropyEstimator.load(path,contract=contract)
    restored.assert_contract(contract)
    assert np.allclose(model.predict(x),restored.predict(x))
    assert restored.metadata['train_ids']==[r['id'] for r in records(128)]
    with pytest.raises(ValueError,match='mismatch'):
        ResampleEntropyEstimator.load(path,contract={'model_sha':'b','dtype':'bf16'})
    with pytest.raises(ValueError,match='mismatch'):
        restored.assert_contract({'model_sha':'b','dtype':'bf16'})
    with pytest.raises(ValueError,match='preserve'):
        model.save(path,contract)
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='mismatch'):
        ResampleEntropyEstimator.load(path)


@pytest.mark.parametrize('x,y',[(np.ones((4,3)),[0,1,float('nan'),0]),
    (np.ones((4,3)),[0,1,2,0]),(np.ones((3,3)),[0,0,0])])
def test_bad_training_values(x,y):
    with pytest.raises(ValueError):
        ResampleEntropyEstimator().fit(x,y,records(4))


def test_invalid_artifact_dimensions_are_rejected(tmp_path):
    import hashlib,json
    model=ResampleEntropyEstimator().fit(np.ones((4,3)),np.zeros(4),records(4))
    path=tmp_path/'bad.npz';model.save(path,{'identity':'fixture'})
    with np.load(path) as data: arrays={name:data[name] for name in data.files}
    arrays['coef']=np.ones(5)
    np.savez_compressed(path,**arrays)
    meta=json.loads(path.with_suffix('.json').read_text())
    meta['weights_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='dimensions'):
        ResampleEntropyEstimator.load(path)


def test_contract_is_portable_but_bound_to_checkpoint_bytes(tmp_path):
    import shutil
    pytest.importorskip('transformers')
    original=tmp_path/'a';original.mkdir()
    for name in ('config.json','tokenizer.json','model.safetensors'):
        (original/name).write_text('{}')
    provider=SimpleNamespace(model_path=str(original),max_new_tokens=128,
        _processor=SimpleNamespace(image_processor=SimpleNamespace(to_dict=lambda:{'size':448})))
    contract=feature_contract(provider)
    copied=tmp_path/'b';shutil.copytree(original,copied)
    provider.model_path=str(copied)
    assert feature_contract(provider)==contract
    (copied/'tokenizer.json').write_text('{"changed":true}')
    assert feature_contract(provider)!=contract


def test_plugin_generates_one_fresh_primary_and_estimates_without_resampling(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from msrc.types import Sample
    from msrc.providers import batched
    block=torch.nn.Identity()
    fake=SimpleNamespace(training=False,parameters=lambda:[],dtype=torch.bfloat16,
        model=SimpleNamespace(language_model=SimpleNamespace(layers=[block])))
    provider=SimpleNamespace(_model=fake,max_new_tokens=128)
    count=[]
    def fresh(provider,images,prompts):
        count.append(len(images));block(torch.ones((len(images),3,4)))
        return [Sample(text='fresh') for _ in images]
    monkeypatch.setattr(batched,'generate_batch',fresh)
    estimator=ResampleEntropyEstimator().fit(np.ones((4,4)),np.zeros(4),records(4))
    contract={'max_new_tokens':128,'weights_sha256':'fixture'}
    estimator.save(tmp_path/'head.npz',contract)
    samples,scores=generate_with_entropy(provider,['image'],['question'],estimator,contract=contract)
    assert count==[1] and samples[0].text=='fresh' and scores.shape==(1,)
    assert samples[0].meta['entropy_extra_generations']==0
    assert samples[0].meta['entropy_is_estimated'] is True


def test_direct_feature_forward_does_not_pass_ids_and_embeddings(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from pipeline import common
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(common,'ROOT',tmp_path)
    block=torch.nn.Identity()
    class Forward:
        training=False
        model=SimpleNamespace(language_model=SimpleNamespace(layers=[block]))
        def parameters(self):return []
        def __call__(self,**kwargs):
            assert 'input_ids' not in kwargs and 'inputs_embeds' in kwargs
            return block(kwargs['inputs_embeds'])
    def inputs(images,texts):
        return dict(input_ids=torch.ones((len(images),3)),inputs_embeds=torch.ones((len(images),3,4)))
    provider=SimpleNamespace(_model=Forward(),_torch=torch,
        _processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0),
            apply_chat_template=lambda *a,**kw:'prompt'),
        _batch_input_cache=SimpleNamespace(prepare=inputs))
    rows=[dict(id='dev1',image='image',question='question')]
    values=extract(provider,rows,tmp_path,'dev',SimpleNamespace(work=lambda stage:nullcontext()),1)
    assert values.shape==(1,4)


def test_cohort_is_label_blind_and_keeps_whole_groups():
    rows=[dict(id=str(i),acceptance_image_group=str(i//3),label=i%2) for i in range(30)]
    chosen=cohort(rows,7,42)
    assert len(chosen)==9
    for group in {r['acceptance_image_group'] for r in chosen}:
        assert sum(r['acceptance_image_group']==group for r in chosen)==3
    changed=[dict(r,label=1-r['label']) for r in rows]
    assert [r['id'] for r in chosen]==[r['id'] for r in cohort(changed,7,42)]
    with pytest.raises(ValueError):
        cohort([dict(id='x')],1,42)


def test_capture_takes_first_prefill_only_and_removes_hook():
    torch=pytest.importorskip('torch')
    block=torch.nn.Identity()
    fake=SimpleNamespace(training=False,parameters=lambda:[],
        model=SimpleNamespace(language_model=SimpleNamespace(layers=[block])))
    provider=SimpleNamespace(_model=fake)
    with PromptStateCapture(provider) as capture:
        block(torch.arange(24).reshape(2,3,4))
        block(torch.ones((2,1,4))*100)
    assert np.array_equal(capture.features(),np.asarray([[8,9,10,11],[20,21,22,23]]))
    assert not block._forward_hooks
    fake.training=True
    with pytest.raises(ValueError,match='frozen'):
        with PromptStateCapture(provider):pass


def test_surrogate_updates_object_support_without_mutating_teacher():
    row=dict(teacher_risks=[.1]*8,available=[True]*8,
        object_support_views_risk=.7,object_support_views_available=False)
    x,a=matrices([row],[.9])
    assert x[0,1]==.9 and x[0,5]==.7 and not a[0,5]
    assert row['teacher_risks']==[.1]*8
    control,control_available=matrices([row],views_only=True)
    assert control[0,1]==.1 and control[0,5]==.7 and not control_available[0,5]


def test_downstream_crossfit_and_inactive_signal_dimensions(tmp_path,monkeypatch):
    from pipeline import common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    rng=np.random.RandomState(12)
    dev,test=records(80),records(40,'test')
    for rows in (dev,test):
        for i,row in enumerate(rows):
            risk=rng.uniform(.05,.95,8);risk[5]=0.
            row.update(dataset='pope' if i%4<2 else 'textvqa',label=i%2,
                target=float(risk[1]),teacher_risks=risk.tolist(),
                available=[True]*5+[False]+[True]*2,
                object_support_views_risk=0.,object_support_views_available=False)
    x=rng.normal(size=(80,16))
    estimator,oof=train(tmp_path,dev,x,{'base_model':'test fixture'})
    states=fit_fusions(tmp_path,dev,oof)
    report=evaluate(tmp_path,dev,test,estimator.predict(rng.normal(size=(40,16))),states,estimator)
    assert report['verdict']=='PILOT_COMPLETED_NOT_ACCEPTANCE'
    assert report['full_acceptance'] is False
    assert all('observed_views' in item['variants'] for item in report['datasets'].values())
    assert (tmp_path/'report.md').is_file()
    for state in states.values():
        assert not state['live'][5]
        assert not set(state['train_ids']) & {r['id'] for r in test}
