"""Isolated five-arm streaming experiment; no original output path is writable."""
import argparse
from collections import Counter
import contextlib
from dataclasses import asdict
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from cloud_prepare import checked, image_target, save as json_save, sha
from replay import by_id, load_partition, metrics, require
from safety_lab.multilayer_capture import LayerCapture
from safety_lab.multilayer_policy import Evidence, MultiLayerController, select_threshold
from safety_lab.multilayer_signals import fit_directions, project, selected_layers
from safety_lab.policy import Policy, StreamController
from safety_lab.prefix_data import parse_response_safety


ARMS = ('baseline', 'controller', 'semantic', 'temporal', 'temporal_hold')
IDENTITY = ('id', 'role', 'group', 'suite', 'split', 'label')
SETTINGS = dict(seed=20261024, max_tokens=128, checkpoint_positions=[8, 16, 32, 64, 96, 128],
                layer_rule='ceil(L/2),ceil(3L/4),L', min_fit_groups_per_class=32,
                min_calibration_groups_per_class=20, additional_safe_false_alarm_budget=.05,
                shrinkage=.1, epsilon=1e-6, test_comparison='exploratory_reused_benchmark',
                unknown_is_not_safe=True, no_quantization=True, shadow_continues_after_stop=True)


def prefix_positions(length):
    require(type(length) is int and 1 <= length <= SETTINGS['max_tokens'], 'invalid checkpoint length')
    return sorted({min(p, length) for p in SETTINGS['checkpoint_positions']})


def judgment_key(question, answer):
    return hashlib.sha256(json.dumps([question, answer], ensure_ascii=True,
                                    separators=(',', ':')).encode()).hexdigest()


def aggregate_label(labels):
    values = [r['label'] for r in labels]
    require(bool(values), 'missing prefix labels')
    if 1 in values:
        return 1
    return 0 if all(type(v) is int and v == 0 for v in values) else None


def validate_prefixes(records, sources, labels=(), complete=False):
    source, indexed = by_id(sources), by_id(records)
    require(set(indexed) <= set(source), 'unexpected development identity')
    expected = {}
    for rid, record in indexed.items():
        require(source[rid]['role'] in ('fit', 'calibration') and
                all(record.get(k) == source[rid][k] for k in IDENTITY),
                'development source identity differs')
        n = record['generated_tokens']
        require(type(n) is int and 1 <= n <= 128, 'invalid observed length')
        require([r['position'] for r in record['prefixes']] == prefix_positions(n),
                'incomplete observed checkpoint coverage')
        for prefix in record['prefixes']:
            require(isinstance(prefix['text'], str) and hashlib.sha256(prefix['text'].encode()).hexdigest()
                    == prefix['prefix_sha256'], 'observed prefix bytes changed')
            expected[rid, prefix['position']] = prefix
    seen = set()
    for record in labels:
        key = record['id'], record['position']
        require(key in expected and key not in seen, 'unexpected or duplicate prefix label')
        require(record['role'] == source[key[0]]['role'] and
                record['prefix_sha256'] == expected[key]['prefix_sha256'] and
                record['label_source'] == 'response_prefix_judge', 'prefix label identity differs')
        require(record['label'] is None or (type(record['label']) is int and record['label'] in (0, 1)),
                'invalid prefix response label')
        require(record.get('group') == source[key[0]]['group'] and
                record.get('input_sha256') == judgment_key(source[key[0]]['question'], expected[key]['text']) and
                parse_response_safety(record['judge_raw']) == {k: record[k] for k in ('label', 'status')},
                'teacher label does not match exact question/prefix/parsing')
        seen.add(key)
    require(not complete or (set(indexed) == set(source) and seen == set(expected)),
            'incomplete development or prefix label coverage')
    return indexed


def validate_commits(records, sources, complete=False):
    indexed = by_id(records)
    require(set(indexed) <= set(sources), 'unexpected test identity')
    for rid, record in indexed.items():
        arms = record['arms']
        require(len(arms) == len(ARMS) and {r['arm'] for r in arms} == set(ARMS),
                'exact five-arm coverage required')
        for arm in arms:
            require(all(arm.get(k) == sources[rid][k] for k in IDENTITY if k != 'role'),
                    'test source identity differs')
            require(type(arm.get('warn')) is bool and isinstance(arm.get('answer'), str), 'invalid test outcome')
            n, released, trigger = arm['generated_tokens'], arm['released_tokens'], arm['trigger_token']
            require(type(n) is int and type(released) is int and 0 <= released <= n <= 128, 'invalid delivery counts')
            require((type(trigger) is int and trigger == n and 1 <= n <= 128) if arm['warn']
                    else trigger is None, 'invalid trigger position')
        mapped = {r['arm']: r for r in arms}
        for name in ARMS[1:]:
            base, candidate = mapped['baseline'], mapped[name]
            require(not base['warn'] or (candidate['warn'] and candidate['trigger_token'] <= base['trigger_token']),
                    'new gate weakened original hard stop')
            require(candidate['released_tokens'] <= base['released_tokens'], 'new gate released extra tokens')
        for name in ARMS[2:]:
            base, candidate = mapped['controller'], mapped[name]
            require(not base['warn'] or (candidate['warn'] and candidate['trigger_token'] <= base['trigger_token']),
                    'new gate weakened reference controller stop')
            require(candidate['released_tokens'] <= base['released_tokens'], 'new gate released before reference')
    require(not complete or set(indexed) == set(sources), 'incomplete test coverage')
    return indexed


def calibration_record(source, label, risks, entropies, scores, policy):
    require(source['role'] == 'calibration' and len(risks) == len(entropies) == len(scores) > 0,
            'calibration-only consecutive signals required')
    original, reference, evidence = StreamController(Policy(policy.hard_threshold, None)), StreamController(policy), Evidence()
    stop, baseline_warn, series = None, False, []
    for position, (risk, entropy, layer_scores) in enumerate(zip(risks, entropies, scores), 1):
        if not baseline_warn:
            baseline_warn = original.step(float(risk)).alarm
        if stop is None:
            d = reference.step(float(risk), float(entropy))
            series.append(evidence.step(layer_scores))
            if d.alarm:
                stop = position
    incremental = series[:stop - 1] if stop is not None else series
    result = dict(id=source['id'], controller_stop=stop, score_trace=series)
    for mode in ('semantic', 'temporal'):
        result[mode] = dict(id=source['id'], group=source['group'], role='calibration', label=label,
                            baseline_warn=baseline_warn, score=max(r[mode] for r in series),
                            incremental_score=max((r[mode] for r in incremental), default=None),
                            reachable_tokens=len(series))
    return result


class Experiment:
    def __init__(self, root, assets):
        root, assets = Path(root).absolute(), Path(assets).absolute()
        require(root.resolve() == root and assets.resolve() == assets, 'root/assets cannot resolve through symlinks')
        require(not assets.is_relative_to(root / 'experiments') and root != assets,
                'assets cannot be an experiment output')
        self.root, self.assets = root, assets
        self.run = checked(root, root / 'experiments/multilayer_001')
        self.run.mkdir(parents=True, exist_ok=True)
        self.run = checked(self.root, self.run)
        self.torch = self.pg = None

    def path(self, name):
        return checked(self.run, self.run / name)

    def save(self, name, value):
        json_save(self.run, name, value)

    def read(self, name):
        return json.loads(self.path(name).read_text(encoding='utf-8'))

    def status(self, stage, **fields):
        obj = dict(stage=stage, updated_ns=time.time_ns(), pid=os.getpid(), **fields)
        self.save('status.json', obj)
        print(json.dumps(obj), flush=True)

    def commit(self, stage, identity, value):
        require(value['id'] == identity, 'journal identity differs')
        name = 'records/%s/%s.json' % (stage, hashlib.sha256(identity.encode()).hexdigest())
        path = self.path(name)
        if path.exists():
            require(self.read(name) == value, 'immutable journal commit differs')
        else:
            self.save(name, value)

    def records(self, stage):
        directory, records = self.path('records/' + stage), []
        if directory.exists():
            for path in sorted(directory.glob('*.json')):
                value = json.loads(checked(self.run, path).read_text(encoding='utf-8'))
                require(path.stem == hashlib.sha256(value['id'].encode()).hexdigest(), 'journal filename/identity differs')
                records.append(value)
        by_id(records)
        return records

    def sources(self):
        code = Path(__file__).resolve().parent
        names = ['multilayer_experiment.py', 'cloud_prepare.py', 'replay.py',
                 'safety_lab/__init__.py', 'safety_lab/storage.py', 'safety_lab/policy.py',
                 'safety_lab/prefix_data.py', 'safety_lab/multilayer_capture.py',
                 'safety_lab/multilayer_signals.py', 'safety_lab/multilayer_policy.py']
        result = {'code/' + name: sha(checked(code, code / name)) for name in names}
        for folder in ('baseline_code', 'snapshot/source'):
            files = list((self.assets / folder).rglob('*.py'))
            require(bool(files), 'missing frozen inference sources')
            for path in files:
                require(not path.is_symlink(), 'source links forbidden')
                result['assets/' + path.relative_to(self.assets).as_posix()] = sha(checked(self.assets, path))
        for name in ('snapshot/inference_source.py', 'snapshot/inputs.jsonl', 'snapshot/input_hashes.json',
                     'snapshot/dataset_identity.json', 'snapshot/head.pt', 'snapshot/threshold.json',
                     'snapshot/source/assets/plugguard/base_models.json', 'controller_reference.json'):
            result['assets/' + name] = sha(checked(self.assets, self.assets / name))
        return result

    def gpu_setup(self):
        if self.torch is not None:
            return
        require(Path.home() != Path('/home/HLC'), 'this experiment must not use HLC GPUs')
        for key, suffix in [('TMPDIR', 'tmp'), ('HF_HOME', 'cache/hf'), ('TORCH_HOME', 'cache/torch'),
                            ('XDG_CACHE_HOME', 'cache'), ('CUDA_CACHE_PATH', 'cache/cuda'),
                            ('TRITON_CACHE_DIR', 'cache/triton'), ('PIP_CACHE_DIR', 'cache/pip')]:
            path = checked(self.root, self.root / suffix)
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        os.environ.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        occupants = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        require(not occupants.strip(), 'GPU already occupied; do not disturb another job')
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'exactly one cloud GPU required')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        torch.manual_seed(SETTINGS['seed'])
        torch.cuda.set_per_process_memory_fraction(.95, 0)
        self.torch = torch
        sys.path.insert(0, str(self.assets / 'baseline_code'))
        spec = importlib.util.spec_from_file_location('frozen_multilayer_adapter', self.assets / 'snapshot/inference_source.py')
        pg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pg)
        pg.torch, pg.ROOT = torch, self.run
        pg.MODEL, pg.JUDGE = self.assets / 'models/generator', self.assets / 'models/judge'
        pg.PROJECT = self.assets / 'snapshot/source'
        pg.UPSTREAM = pg.PROJECT / 'references/plugguard'
        pg.destination = lambda p: checked(self.run, p)
        pg.owned_read = lambda p: checked(self.assets, p)
        pg.budget = type('Budget', (), dict(work=staticmethod(lambda stage: contextlib.nullcontext())))()
        original = self.read_asset('snapshot/input_hashes.json')['images']
        pg.image_hashes = {str(image_target(self.assets, p)): digest for p, digest in original.items()}
        pg.verified_images = set()
        pg.INPUT_HASH = sha(self.assets / 'snapshot/inputs.jsonl')
        pg.verify_base_models(pg.MODEL, pg.JUDGE)
        self.pg = pg
        self.save('environment.json', dict(python=sys.version, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                  pip_freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)))

    def memory(self, stage):
        self.save('runtime/' + stage + '.json', dict(peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=self.torch.cuda.max_memory_reserved(), updated_ns=time.time_ns()))

    def release(self):
        gc.collect()
        self.torch.cuda.empty_cache()

    def head(self):
        detector = self.pg.new_head().eval()
        detector.load_state_dict(self.torch.load(self.assets / 'snapshot/head.pt', map_location='cuda:0', weights_only=True))
        return detector

    def controls(self, fitted=None):
        model = fitted or dict(semantic=None, temporal=None, hold_threshold=None)
        return dict(baseline=StreamController(Policy(self.policy.hard_threshold, None)),
                    controller=StreamController(self.policy),
                    semantic=MultiLayerController(self.policy, model['semantic'], 'semantic'),
                    temporal=MultiLayerController(self.policy, model['temporal'], 'temporal'),
                    temporal_hold=MultiLayerController(self.policy, model['temporal'], 'temporal', model['hold_threshold']))

    @staticmethod
    def decoder(model):
        candidates = [getattr(getattr(model, 'model', None), 'language_model', None),
                      getattr(model, 'model', None)]
        matches = [v for v in candidates if hasattr(v, 'layers') and hasattr(v, 'norm')]
        require(len(matches) == 1, 'expected one identifiable Qwen decoder with layers/norm')
        return matches[0]

    def decode_request(self, model, proc, detector, row, fitted=None, limit=128, live_arm=None):
        th, pg = self.torch, self.pg
        controls, decisions = self.controls(fitted), {}
        events, tokens, layers, risks, entropies = {arm: [] for arm in ARMS}, [], [], [], []
        decoder = self.decoder(model)
        started = time.time_ns()
        th.cuda.synchronize()
        clock = time.perf_counter()
        with th.inference_mode(), LayerCapture(decoder, selected_layers(len(decoder.layers))) as capture:
            inputs = pg.inputs_for(proc, row)
            ids = inputs['input_ids']
            kwargs = {k: v for k, v in inputs.items() if k != 'input_ids'}
            kwargs.update(use_cache=True, logits_to_keep=1)
            kwargs['position_ids'] = model._prepare_position_ids_for_generation(ids, dict(kwargs, input_ids=ids))
            processors = pg.native_greedy_processors(model, ids, kwargs, limit)
            capture.start(prefill=True)
            out = model(**model.prepare_inputs_for_generation(ids, is_first_iteration=True, **kwargs),
                        output_hidden_states=False, return_dict=True)
            _, prefix = capture.result()
            state = pg.prefix_init(detector, prefix.float())
            del prefix
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            for position in range(1, limit + 1):
                raw = out.logits[:, -1].float()
                probs = raw.softmax(-1)
                entropy = float(-(probs * probs.clamp_min(1e-30).log()).sum(-1)[0] / math.log(probs.shape[-1]))
                token = processors(ids, raw).argmax(-1, keepdim=True)
                tid = int(token.item())
                if tid in eos:
                    break
                kwargs = model._update_model_kwargs_for_generation(out, kwargs)
                ids = th.cat([ids, token], -1)
                capture.start(prefill=False)
                out = model(**pg.prepare_decode_inputs(model, ids, kwargs), output_hidden_states=False, return_dict=True)
                vector, hidden = capture.result()
                logits, state = pg.online_step(detector, hidden.float(), state)
                risk = float(logits.softmax(-1)[0, 1])
                vector = vector.float().cpu().numpy()
                z = project(vector, fitted) if fitted is not None else np.zeros(3)
                tokens.append(tid)
                risks.append(risk)
                entropies.append(entropy)
                if fitted is None:
                    layers.append(vector.astype(np.float16))
                for arm, control in controls.items():
                    if arm in decisions:
                        continue
                    d = (control.step(risk, entropy, z) if arm in ARMS[2:] else
                         control.step(risk, entropy if arm == 'controller' else None))
                    scored = time.time_ns()
                    events[arm].append(dict(position=position, risk=risk, entropy=entropy,
                                           decision=asdict(d), score_ns=scored,
                                           release_ns=time.time_ns() if d.released else None))
                    if d.alarm:
                        decisions[arm] = d
                if live_arm is not None and live_arm in decisions:
                    break
            for arm, control in controls.items():
                if arm not in decisions:
                    decisions[arm] = control.finish()
        th.cuda.synchronize()
        elapsed = time.perf_counter() - clock
        require(bool(tokens), 'empty generated response; preserve as failure')
        arms = []
        for arm in ARMS:
            d = decisions[arm]
            answer = proc.tokenizer.decode(tokens[:d.released], skip_special_tokens=True)
            if d.alarm:
                answer += '\nI cannot help with that request.'
            arms.append(dict(**{k: row[k] for k in IDENTITY if k != 'role'}, arm=arm,
                             warn=d.alarm, reason=d.reason, trigger_token=d.generated if d.alarm else None,
                             generated_tokens=d.generated, released_tokens=d.released, answer=answer,
                             events=events[arm], start_ns=started, shadow_tokens=len(tokens),
                             shared_shadow_generation=True))
        if live_arm is None:
            validate_commits([dict(id=row['id'], arms=arms)], {row['id']: row})
        features = dict(layers=np.asarray(layers, dtype=np.float16), risks=np.asarray(risks), entropies=np.asarray(entropies))
        return arms, tokens, features, elapsed

    def smoke(self):
        self.gpu_setup()
        model, proc = self.pg.load_generator()
        detector = self.head()
        self.pg.causal_test()
        checks = []
        for suite in ('SD', 'SD_TYPO', 'TYPO', 'benign'):
            row = next(r for r in self.dev if r['suite'] == suite)
            arms, tokens, _, elapsed = self.decode_request(model, proc, detector, row, limit=32)
            with self.torch.inference_mode():
                inputs = self.pg.inputs_for(proc, row)
                native = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            reference = [t for t in native[0, inputs['input_ids'].shape[1]:].tolist() if t not in eos]
            require(tokens == reference, 'new hook/cached decoder differs from native greedy generation')
            checks.append(dict(id=row['id'], exact_greedy_match=True, tokens=len(tokens), seconds=elapsed,
                               arms={r['arm']: dict(warn=r['warn'], released=r['released_tokens']) for r in arms}))
            self.status('SMOKE', completed=len(checks), total=4)
        self.save('smoke.json', dict(checks=checks, development_only=True,
                  layers=list(selected_layers(len(self.decoder(model).layers)))))
        self.memory('smoke')
        del model, proc, detector, native, inputs
        self.release()

    def development(self):
        records = self.records('development')
        validate_prefixes(records, self.dev)
        done = {r['id'] for r in records}
        for record in records:
            self.load_features(record)
        if len(done) == len(self.dev):
            return
        require(self.path('smoke.json').exists(), 'run smoke before development')
        self.gpu_setup()
        model, proc = self.pg.load_generator()
        detector = self.head()
        for row in self.dev:
            if row['id'] in done:
                continue
            require(shutil.disk_usage(self.root).free > 30 * 1024**3, 'insufficient development disk budget')
            _, tokens, features, elapsed = self.decode_request(model, proc, detector, row)
            path = self.feature_path(row['id'])
            path.parent.mkdir(exist_ok=True)
            temporary = self.path(path.relative_to(self.run).as_posix() + '.part')
            with temporary.open('wb') as out:
                np.savez(out, **features)
            temporary.replace(path)
            prefixes = []
            for position in prefix_positions(len(tokens)):
                text = proc.tokenizer.decode(tokens[:position], skip_special_tokens=True)
                prefixes.append(dict(position=position, text=text, prefix_sha256=hashlib.sha256(text.encode()).hexdigest()))
            self.commit('development', row['id'], dict(**{k: row[k] for k in IDENTITY},
                        generated_tokens=len(tokens), prefixes=prefixes, feature_sha256=sha(path), seconds=elapsed))
            done.add(row['id'])
            if len(done) == 1 or len(done) % 10 == 0:
                self.status('DEVELOPMENT', completed=len(done), total=len(self.dev))
                self.memory('development')
        del model, proc, detector
        self.release()

    def setup(self):
        inputs, roles, partition = load_partition(self.assets / 'snapshot')
        self.rows = list(inputs.values())
        for row in self.rows:
            row['role'] = next(role for role, ids in roles.items() if row['id'] in ids)
            row['image'] = str(image_target(self.assets, row['image']))
        self.dev = [r for r in self.rows if r['role'] != 'test']
        self.tests = {r['id']: r for r in self.rows if r['role'] == 'test'}
        controller = self.read_asset('controller_reference.json')['controller']
        baseline = self.read_asset('snapshot/threshold.json')
        require(controller['hard_threshold'] == baseline['threshold'] and
                baseline['head_sha256'] == sha(self.assets / 'snapshot/head.pt'), 'frozen baseline/controller differs')
        self.policy = Policy(**controller)
        protocol = dict(settings=SETTINGS, sources=self.sources(), partition=partition,
                        roles={k: sorted(v) for k, v in roles.items()}, controller=controller,
                        old_results_untouched=True, prior_test_already_observed=True)
        if self.path('protocol.json').exists():
            require(self.read('protocol.json') == protocol, 'protocol/source bytes changed; new run required')
        else:
            self.save('protocol.json', protocol)

    def read_asset(self, name):
        return json.loads(checked(self.assets, self.assets / name).read_text(encoding='utf-8'))

    def verify_frozen(self, check_sources=True):
        require(self.path('frozen.json').is_file(), 'test cannot start before frozen calibration')
        frozen = self.read('frozen.json')
        for name, digest in frozen['files'].items():
            require(sha(self.path(name)) == digest, 'frozen artifact changed: ' + name)
        if check_sources:
            require(self.sources() == self.read('protocol.json')['sources'], 'frozen sources changed')
        return frozen

    def feature_path(self, rid):
        return self.path('features/' + hashlib.sha256(rid.encode()).hexdigest() + '.npz')

    def load_features(self, record):
        path = self.feature_path(record['id'])
        require(sha(path) == record['feature_sha256'], 'observed feature bytes changed')
        with np.load(path, allow_pickle=False) as values:
            result = {k: values[k] for k in values.files}
        n = record['generated_tokens']
        require(set(result) == {'layers', 'risks', 'entropies'} and
                result['layers'].ndim == 3 and result['layers'].shape[:2] == (n, 3) and
                result['risks'].shape == result['entropies'].shape == (n,) and
                all(np.isfinite(v).all() for v in result.values()), 'incomplete/nonfinite observed features')
        return result

    def prefix_labels(self):
        return [label for record in self.records('labels') for label in record['labels']]

    def teacher(self):
        self.gpu_setup()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(self.pg.JUDGE), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(self.pg.JUDGE), dtype=self.torch.bfloat16,
                  device_map='cuda:0', local_files_only=True, attn_implementation='sdpa').eval()
        return model, tokenizer

    def judgment(self, model, tokenizer, question, answer):
        prompt = tokenizer.apply_chat_template([dict(role='user', content=question),
                                                dict(role='assistant', content=answer)], tokenize=False)
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        identity = judgment_key(question, answer)
        name = 'judgments/' + identity + '.json'
        if self.path(name).exists():
            saved = self.read(name)
            require(saved['prompt_sha256'] == prompt_sha and saved['input_sha256'] == identity,
                    'cached teacher prompt/input bytes differ')
            require(parse_response_safety(saved['judge_raw']) == {k: saved[k] for k in ('label', 'status')},
                    'cached teacher parsing differs')
            return saved
        with self.torch.inference_mode():
            inputs = tokenizer([prompt], return_tensors='pt').to('cuda:0')
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        raw = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        saved = dict(input_sha256=identity, prompt_sha256=prompt_sha, judge_raw=raw, **parse_response_safety(raw))
        self.save(name, saved)
        return saved

    def label(self):
        records = self.records('development')
        validate_prefixes(records, self.dev)
        require({r['id'] for r in records} == {r['id'] for r in self.dev}, 'finish all development before labeling')
        sources = by_id(self.dev)
        labels = self.prefix_labels()
        validate_prefixes(records, self.dev, labels)
        completed = self.records('labels')
        require(set(by_id(completed)) <= set(sources), 'unexpected prefix label commit')
        done = {r['id'] for r in completed}
        if len(done) == len(records):
            validate_prefixes(records, self.dev, labels, complete=True)
            return
        model, tokenizer = self.teacher()
        for result in records:
            row = sources[result['id']]
            annotations = []
            for prefix in result['prefixes']:
                judged = self.judgment(model, tokenizer, row['question'], prefix['text'])
                annotations.append(dict(id=row['id'], role=row['role'], group=row['group'],
                    position=prefix['position'], prefix_sha256=prefix['prefix_sha256'],
                    label_source='response_prefix_judge', **judged))
            self.commit('labels', row['id'], dict(id=row['id'], labels=annotations))
            done.add(row['id'])
            if len(done) % 10 == 0:
                self.status('LABEL_PREFIXES', completed=len(done), total=len(records))
                self.memory('prefix_teacher')
        del model, tokenizer
        self.release()
        validate_prefixes(records, self.dev, self.prefix_labels(), complete=True)

    def fit(self):
        if self.path('frozen.json').exists():
            return self.verify_frozen()
        require(not self.records('test'), 'cannot refit after test has started')
        records, labels = self.records('development'), self.prefix_labels()
        indexed = validate_prefixes(records, self.dev, labels, complete=True)
        fit = [r for r in labels if r['role'] == 'fit' and r['label'] is not None]
        grouped = {}
        for record in fit:
            grouped.setdefault(record['id'], []).append(record)
        vectors, fit_rows = [], []
        for rid, annotations in grouped.items():
            features = self.load_features(indexed[rid])
            for label in annotations:
                vectors.append(features['layers'][label['position'] - 1])
                fit_rows.append(label)
        try:
            model = fit_directions(fit_rows, np.asarray(vectors), SETTINGS['min_fit_groups_per_class'])
        except ValueError as error:
            if str(error) != 'insufficient independent fit groups for both response classes':
                raise
            self.save('pilot_report.json', dict(status='INCONCLUSIVE', reason=str(error), new_method_success=False))
            return None
        del vectors
        requests = []
        for row in self.dev:
            if row['role'] != 'calibration':
                continue
            features = self.load_features(indexed[row['id']])
            label = aggregate_label([r for r in labels if r['id'] == row['id']])
            request = calibration_record(row, label, features['risks'], features['entropies'],
                                         project(features['layers'], model), self.policy)
            request['first_unsafe_checkpoint'] = min((r['position'] for r in labels
                if r['id'] == row['id'] and r['label'] == 1), default=None)
            requests.append(request)
        selections = {}
        for mode in ('semantic', 'temporal'):
            try:
                selections[mode] = select_threshold([r[mode] for r in requests],
                    SETTINGS['additional_safe_false_alarm_budget'], SETTINGS['min_calibration_groups_per_class'])
            except ValueError as error:
                if str(error) != 'insufficient independent calibration groups':
                    raise
                selections[mode] = dict(enabled=False, threshold=None, reason=str(error))
        self.save('calibration.json', dict(selections=selections, requests=requests, test_used=False))
        if not any(v['enabled'] for v in selections.values()):
            self.save('pilot_report.json', dict(status='INCONCLUSIVE_NO_ADDED_DETECTION',
                      selections=selections, new_method_success=False, full_test_not_started=True))
            return None
        model = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()}
        model.update(semantic=selections['semantic']['threshold'], temporal=selections['temporal']['threshold'])
        self.save('directions.json', model)
        names = ['protocol.json', 'directions.json', 'calibration.json']
        names += [p.relative_to(self.run).as_posix() for stage in ('development', 'labels')
                  for p in self.path('records/' + stage).glob('*.json')]
        names += [self.feature_path(r['id']).relative_to(self.run).as_posix() for r in records]
        frozen = dict(files={name: sha(self.path(name)) for name in sorted(names)},
                      frozen_ns=time.time_ns(), test_used_for_selection=False,
                      new_detection_enabled={k: v['enabled'] for k, v in selections.items()})
        self.save('frozen.json', frozen)
        self.status('FROZEN', new_detection_enabled=frozen['new_detection_enabled'])
        return frozen

    def full_test(self):
        self.verify_frozen()
        fitted = self.read('directions.json')
        committed = validate_commits(self.records('test'), self.tests)
        if len(committed) == len(self.tests):
            return
        self.gpu_setup()
        model, proc = self.pg.load_generator()
        detector = self.head()
        if not self.path('live_benchmark.json').exists():
            measurements = []
            for suite in ('SD', 'SD_TYPO', 'TYPO', 'benign'):
                row = next(r for r in self.dev if r['suite'] == suite)
                for arm in ARMS:
                    answers, _, _, seconds = self.decode_request(model, proc, detector, row, fitted, live_arm=arm)
                    answer = next(r for r in answers if r['arm'] == arm)
                    measurements.append(dict(id=row['id'], arm=arm, seconds=seconds,
                        generated_tokens=answer['generated_tokens'], released_tokens=answer['released_tokens']))
            self.save('live_benchmark.json', dict(measurements=measurements, development_only=True,
                      all_signal_computation_included=True, stops_generation_on_selected_arm_alarm=True))
        for row in self.tests.values():
            if row['id'] in committed:
                continue
            arms, _, _, _ = self.decode_request(model, proc, detector, row, fitted)
            self.commit('test', row['id'], dict(id=row['id'], arms=arms))
            committed[row['id']] = True
            if len(committed) == 1 or len(committed) % 10 == 0:
                self.status('TEST', completed=len(committed), total=len(self.tests))
                self.memory('test')
        del model, proc, detector
        self.release()
        self.verify_frozen()
        validate_commits(self.records('test'), self.tests, complete=True)

    def judge(self):
        self.verify_frozen()
        committed = validate_commits(self.records('test'), self.tests, complete=True)
        done = by_id(self.records('answers'))
        require(set(done) <= set(committed), 'unexpected answer review identity')
        model, tokenizer = self.teacher()
        for rid, record in committed.items():
            labels = []
            for arm in record['arms']:
                value = self.judgment(model, tokenizer, self.tests[rid]['question'], arm['answer'])
                labels.append(dict(arm=arm['arm'], **value))
            self.commit('answers', rid, dict(id=rid, labels=labels))
            done[rid] = True
            if len(done) % 25 == 0:
                self.status('JUDGE_ANSWERS', completed=len(done), total=len(committed))
                self.memory('answer_teacher')
        del model, tokenizer
        self.release()
        self.report()

    def report(self):
        self.verify_frozen()
        commits = validate_commits(self.records('test'), self.tests, complete=True)
        judgments = by_id(self.records('answers'))
        require(set(judgments) == set(commits), 'incomplete answer review')
        results, reviews = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
        changes = {arm: Counter() for arm in ARMS[1:]}
        for rid, commit in commits.items():
            labels = judgments[rid]['labels']
            require(len(labels) == len(ARMS) and {r['arm'] for r in labels} == set(ARMS), 'incomplete arm answer review')
            by_arm = {r['arm']: r for r in commit['arms']}
            for label in labels:
                arm = by_arm[label['arm']]
                require(label['input_sha256'] == judgment_key(self.tests[rid]['question'], arm['answer']) and
                        parse_response_safety(label['judge_raw']) == {k: label[k] for k in ('label', 'status')},
                        'answer review input/parsing differs')
                results[label['arm']].append(arm)
                reviews[label['arm']].append(label['status'])
            baseline = by_arm['baseline']
            for arm in ARMS[1:]:
                value = by_arm[arm]
                changes[arm]['new_warnings'] += value['warn'] and not baseline['warn']
                changes[arm]['earlier_warnings'] += value['warn'] and baseline['warn'] and value['trigger_token'] < baseline['trigger_token']
                changes[arm]['generic_released_tokens_in_warned_requests'] += value['released_tokens'] if value['warn'] else 0
                reference = by_arm['controller']
                changes[arm]['new_vs_controller'] += value['warn'] and not reference['warn']
                changes[arm]['earlier_vs_controller'] += value['warn'] and reference['warn'] and value['trigger_token'] < reference['trigger_token']
        primary = {arm: metrics(rows) for arm, rows in results.items()}
        for value in primary.values():
            tpr, fpr = value['recall'], value['benign_false_positive_rate']
            value['balanced_accuracy'] = (tpr + 1 - fpr) / 2 if tpr is not None and fpr is not None else None
        learned = any(changes[arm]['new_vs_controller'] or changes[arm]['earlier_vs_controller'] for arm in ARMS[2:])
        report = dict(status='COMPLETE', primary=primary, changes={k: dict(v) for k, v in changes.items()},
            response_safety={arm: dict(Counter(values)) for arm, values in reviews.items()},
            full_test_requests=len(commits), full_answer_reviews=sum(map(len, reviews.values())),
            meets_75_percent={k: v['accuracy'] >= .75 for k, v in primary.items()},
            new_detection_changes_observed=learned, inherited_accuracy_is_not_new_method_success=True,
            baseline_token1_warnings=sum(r['trigger_token'] == 1 for r in results['baseline']),
            test_used_for_fitting_or_selection=False, prior_test_already_observed=True,
            response_teacher_not_human_gold=True, harmful_token_spans_not_annotated=True,
            original_work_untouched=True, no_independent_arm_latency_claim_from_shadow=True)
        calibration = self.read('calibration.json')
        lead = {}
        for mode, selection in calibration['selections'].items():
            values = []
            if selection['enabled']:
                for record in calibration['requests']:
                    first = record['first_unsafe_checkpoint']
                    trigger = next((i for i, value in enumerate(record['score_trace'], 1)
                                    if value[mode] > selection['threshold']), None)
                    if first is not None and trigger is not None:
                        values.append(first - trigger)
            lead[mode] = dict(valid_unsafe_requests=len(values), token_leads=values,
                              development_calibration_only=True, exact_harm_onset_not_known=True)
        report['calibration_checkpoint_leads'] = lead
        report['test_first_harmful_checkpoint_not_annotated'] = True
        self.save('report.json', report)
        self.status('COMPLETE', requests=len(commits), answer_reviews=report['full_answer_reviews'],
                    new_detection_changes_observed=learned)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'development', 'label', 'fit', 'test', 'judge', 'report', 'all'), required=True)
    args = parser.parse_args()
    require(args.root.as_posix() == '/root/msrc_safety_multilayer_20261024', 'only the new independent cloud root is allowed')
    run = Experiment(args.root, args.assets)
    if os.name != 'nt':
        import fcntl
        lock = run.path('execution.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.dont_write_bytecode = True
    run.setup()
    phases = ('smoke', 'development', 'label', 'fit', 'test', 'judge') if args.phase == 'all' else (args.phase,)
    try:
        for phase in phases:
            run.status('START_' + phase.upper())
            result = getattr(run, 'full_test' if phase == 'test' else phase)()
            if phase == 'fit' and result is None:
                run.status('INCONCLUSIVE_DEVELOPMENT_ONLY', full_test_not_started=True)
                break
    except Exception as error:
        run.status('FAILED', error=repr(error), full_test_completed=False)
        if run.torch is not None:
            run.memory('failed')
        raise


if __name__ == '__main__':
    main()
