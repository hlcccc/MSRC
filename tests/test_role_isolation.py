"""CPU synthetic invariants, not model performance or GPU acceptance evidence."""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from msrc.model import RiskCalibrator
from msrc.signals import Evidence, build_report
from msrc.types import Sample
from pipeline import factual

ROOT = Path(__file__).resolve().parents[1]


def function(path, name):
    parsed = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    return next(node for node in ast.walk(parsed)
                if isinstance(node, ast.FunctionDef) and node.name == name)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_evidence(answer, index):
    return Evidence(
        question='Synthetic question %d?' % index, answer=answer,
        primary=Sample(text=answer, sequence_confidence=.50 + index % 7 * .06,
                       sequence_entropy=.3 + index % 5 * .25),
        views=[Sample(text=answer if (index + k) % 3 else 'unclear') for k in range(4)],
        resamples=[Sample(text=answer if (index + k) % 4 else 'unclear') for k in range(3)],
        verifications=[Sample(text='yes' if (index + k) % 3 else 'no') for k in range(3)],
        ocr_texts=['yes no synthetic'],
    )


def test_acceptance_candidates_and_donors_are_fixed():
    assert factual.SIGNALS == [
        'self_consistency', 'resample_consistency', 'typed_verification',
        'cross_model_agreement', 'grounding_check', 'object_support',
        'sequence_confidence', 'output_entropy',
    ]
    assert factual.DONORS == {'llava': 'qwen', 'qwen': 'llava', 'internvl': 'qwen'}


def test_primary_prompt_does_not_take_answers_or_labels():
    calls = [node for node in ast.walk(function('scripts/collect_evidence.py', 'main'))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'provider'
             and node.func.attr == 'generate']
    assert len(calls) == 3
    for call in calls:
        assert ast.unparse(call.args[0]) == 'image'
        assert {node.id for node in ast.walk(call.args[1])
                if isinstance(node, ast.Name)} == {'question'}
        assert {key.arg: ast.literal_eval(key.value)
                for key in call.keywords} == {'do_sample': False}


def test_no_second_model_loaded_by_default():
    spec = importlib.util.spec_from_file_location('isolation_collector', ROOT / 'scripts/collect_evidence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build_second_provider(SimpleNamespace(second_model_path='')) is None
    assert '--second-model-path' not in ast.unparse(function('pipeline/factual.py', 'collect'))


def test_generation_has_fresh_single_turn_and_inference_context():
    generate = function('msrc/providers/hf_vision.py', 'generate')
    assignment = next(node for node in ast.walk(generate) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'conversation'
                              for target in node.targets))
    assert isinstance(assignment.value, ast.List) and len(assignment.value.elts) == 1
    first = assignment.value.elts[0]
    role = dict(zip((ast.literal_eval(key) for key in first.keys), first.values))['role']
    assert ast.literal_eval(role) == 'user'
    source = ast.unparse(generate)
    assert 'torch.inference_mode()' in source
    assert 'self._model.eval()' in ast.unparse(function('msrc/providers/hf_vision.py', 'load'))
    for forbidden in ('past_key_values', 'self.history', '.backward(', '.step(',
                      'save_pretrained(', 'load_adapter('):
        assert forbidden not in source


def test_posthoc_attachment_does_not_mutate_primary_or_donor():
    row = {'answer': 'yes', 'evidence': make_evidence('yes', 3).to_dict()}
    donor = {'id': 'synthetic:3', 'question': 'Synthetic question 3?', 'answer': 'no',
             'label': 1, 'risk': .99, 'calibrated_score': .98}
    before = copy.deepcopy((row, donor))
    evidence = Evidence.from_dict(row['evidence'])
    evidence.cross_model = [Sample(text=donor['answer'], meta={'fresh_donor': 'qwen',
                                                            'source_id': donor['id']})]
    build_report(evidence)
    assert (row, donor) == before
    assert row['evidence']['cross_model'] == []


def test_evaluator_reads_donor_text_not_donor_labels_or_scores():
    evaluator = function('pipeline/factual_parameters.py', 'features')
    fields = {ast.literal_eval(node.slice) for node in ast.walk(evaluator)
              if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
              and node.value.id == 'donor' and isinstance(node.slice, ast.Constant)}
    assert fields == {'id', 'question', 'answer'}
    assert 'from msrc.providers' not in ast.unparse(evaluator)


def test_calibrators_have_separate_parameters_and_scalers():
    x = np.asarray([[i / 10, (i % 4) / 4] for i in range(20)])
    y = np.asarray([i % 2 for i in range(20)])
    qwen = RiskCalibrator(l2=.05).fit(x, y)
    llava = RiskCalibrator(l2=.05).fit(x + .1, y)
    before = [value.copy() for value in (qwen.coef_, qwen.mean_, qwen.std_)]
    for a, b in zip((qwen.coef_, qwen.mean_, qwen.std_),
                    (llava.coef_, llava.mean_, llava.std_)):
        assert not np.shares_memory(a, b)
    llava.fit(x * 3 + 4, 1 - y)
    for old, now in zip(before, (qwen.coef_, qwen.mean_, qwen.std_)):
        np.testing.assert_array_equal(old, now)


def test_actual_evaluator_order_and_files_are_isolated(tmp_path, monkeypatch):
    def checked(path):
        path = Path(path).resolve()
        if not path.is_relative_to(tmp_path.resolve()):
            raise ValueError('synthetic test escaped its temporary directory')
        return path

    def save(path, payload):
        path = checked(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')

    def rows(path, values):
        path = checked(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in values),
                        encoding='utf-8')

    outputs = []
    for suffix, order in (('llava_first', ('llava', 'qwen', 'internvl')),
                          ('qwen_first', ('qwen', 'internvl', 'llava'))):
        out = checked(tmp_path / suffix)
        save(out / 'protocol.json', {'mode': 'smoke', 'synthetic_only': True})
        for dataset in ('pope', 'textvqa'):
            raw = [dict(id='synthetic:%s:%d' % (dataset, i), question='Synthetic question %d?' % i,
                        gold_answers=['yes'] * 10, acceptance_split='dev' if i < 12 else 'test')
                   for i in range(24)]
            rows(out / 'inputs' / (dataset + '.jsonl'), raw)
            for family, offset in (('llava', 0), ('qwen', 1), ('internvl', 2)):
                records = []
                for i, original in enumerate(raw):
                    answer = 'no' if (i + offset) % 3 == 0 else 'yes'
                    records.append(dict(original, answer=answer, label=int(answer != 'yes'),
                                        evidence=make_evidence(answer, i + offset).to_dict()))
                rows(out / 'evidence' / (family + '_' + dataset + '.jsonl'), records)
        files = sorted((out / 'evidence').glob('*.jsonl')) + sorted((out / 'inputs').glob('*.jsonl'))
        before = {str(path): digest(path) for path in files}
        monkeypatch.setattr(factual, 'MODELS', {key: 'synthetic-unused' for key in order})
        monkeypatch.setattr(factual, 'inside', checked)
        monkeypatch.setattr(factual, 'save', save)
        with contextlib.redirect_stdout(io.StringIO()):
            factual.evaluate(out)
        assert {str(path): digest(path) for path in files} == before
        report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
        assert report['status'] == 'SMOKE_ONLY_NOT_FULL_ACCEPTANCE'
        assert report['acceptance_conclusion'] is None
        calibrator_paths = list((out / 'calibrators').glob('*.json'))
        assert len(calibrator_paths) == 6
        results = {row['cell']: row for row in report['cells']}
        calibrators = {path.stem: json.loads(path.read_text(encoding='utf-8'))
                       for path in calibrator_paths}
        outputs.append((results, calibrators))
    assert outputs[0] == outputs[1]
