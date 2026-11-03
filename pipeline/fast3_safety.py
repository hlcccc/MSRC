"""Supplied-state causal safety, with development-gated execution optimizations."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
import time

from extensions.multilayer import runner, variants
from pipeline import common, fast3_contract as c
from pipeline.frozen_judge_bucket_5090 import comparison, prompt_lengths, scheduled_texts

SUITES = ('SD', 'SD_TYPO', 'TYPO', 'benign')


def canonical(arms):
    values = deepcopy(arms)
    for arm in values:
        arm.pop('start_ns')
        for event in arm['events']:
            event.pop('score_ns')
            event['release_ns'] = event['release_ns'] is not None
    return values


def choose_judge(profile):
    c.require(profile['complete'] and profile['test_used_for_selection'] is False,
        'development-only completed judge profile required')
    reference = [r for r in profile['measurements'] if r['batch_size'] == 6]
    c.require(len(reference) == 2 and all(r['status'] == 'OK' and r['rng_unchanged'] for r in reference)
        and reference[0]['raw'] == reference[1]['raw'] and
        comparison(reference[0]['raw'], reference[1]['raw'], runner.parse_response_safety)['safety_labels_equal'],
        'unrepeatable or invalid baseline judge')
    eligible = []
    for size in (6, 16, 32, 64):
        rows = [r for r in profile['measurements'] if r['batch_size'] == size]
        if (len(rows) == 2 and {r['repeat'] for r in rows} == {0, 1}
                and all(r['status'] == 'OK' and r['rng_unchanged'] and
                    r['comparison']['safety_labels_equal'] for r in rows)
                and rows[0]['raw'] == rows[1]['raw']):
            eligible.append(dict(batch_size=size, seconds=sum(r['seconds'] for r in rows) / 2))
    c.require(eligible, 'no quality-admitted review configuration')
    best = min(r['seconds'] for r in eligible)
    chosen = min((r for r in eligible if r['seconds'] <= best / .98), key=lambda r: r['batch_size'])
    return dict(chosen, bucket=chosen['batch_size'] != 6, eligible=eligible,
        test_used_for_selection=False, primary_warning_metric_unchanged=True,
        refusal_is_diagnostic_not_primary=True, policy='fastest two repeatable development rounds; 2% tie smaller')


class Experiment(variants.VariantExperiment):
    fit = development = label = c.training_forbidden
    optimized = False

    def gpu_setup(self):
        if self.torch is not None:
            return
        super().gpu_setup()
        total = self.torch.cuda.get_device_properties(0).total_memory
        cap = min(30 * 1024**3, int(total * .95))
        self.torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        environment = self.read('environment.json')
        environment['allocator_cap_bytes'] = cap
        environment['exclusive_fast3_memory_policy'] = True
        self.save('environment.json', environment)

    def sources(self):
        result = super().sources()
        result['fast3_source_tree'] = c.digest(c.sources())
        return result

    def decode_request(self, model, proc, detector, row, fitted=None, limit=128, live_arm=None):
        if self.optimized and fitted is not None and live_arm is None:
            from pipeline.frozen_safety_hotpath_5090 import decode_request_candidate
            return decode_request_candidate(self, model, proc, detector, row, fitted,
                limit, live_arm, retain_features=False)
        return super().decode_request(model, proc, detector, row, fitted, limit, live_arm)

    def development_gate(self):
        self.verify_frozen()
        self.gpu_setup()
        model, proc = self.pg.load_generator()
        detector, fitted = self.head(), self.read('directions.json')
        probes = [r for suite in SUITES for r in [x for x in self.dev if x['suite'] == suite][:2]]
        c.require(len(probes) == 8 and len({r['id'] for r in probes}) == 8 and
            all(r['role'] in ('fit', 'calibration') and r['id'] not in self.tests for r in probes),
            'eight development-only safety probes required')
        measurements, pairs = [], []
        for i, row in enumerate(probes):
            self.torch.manual_seed(c.seed_for('safety-dev', i))
            state = (self.torch.get_rng_state().clone(), self.torch.cuda.get_rng_state().clone())
            self.optimized = False
            a = self.decode_request(model, proc, detector, row, fitted)
            rng_a = (self.torch.get_rng_state().clone(), self.torch.cuda.get_rng_state().clone())
            self.torch.set_rng_state(state[0])
            self.torch.cuda.set_rng_state(state[1])
            self.optimized = True
            b = self.decode_request(model, proc, detector, row, fitted)
            same_rng = self.torch.equal(rng_a[0], self.torch.get_rng_state()) and self.torch.equal(
                rng_a[1], self.torch.cuda.get_rng_state())
            measurements.append(dict(id=row['id'], role=row['role'], suite=row['suite'],
                equivalent=canonical(a[0]) == canonical(b[0]) and a[1] == b[1] and same_rng,
                baseline_seconds=a[3], candidate_seconds=b[3], rng_equal=same_rng))
            pairs.extend(dict(id=row['id'], role=row['role'], question=row['question'], answer=arm['answer'])
                for arm in a[0])
            pairs.extend(dict(id=row['id'], role=row['role'], question=row['question'],
                answer=proc.tokenizer.decode(a[1][:n], skip_special_tokens=True))
                for n in sorted(set(range(8, len(a[1]) + 1, 8)) | {len(a[1])}))
        eligible = all(r['equivalent'] for r in measurements)
        faster = sum(r['candidate_seconds'] for r in measurements) < sum(r['baseline_seconds'] for r in measurements)
        self.optimized = eligible and faster
        self.save('runtime/hotpath_gate.json', dict(complete=True, development_only=True,
            test_used_for_selection=False, enabled=self.optimized, equivalent=eligible, measurements=measurements,
            fallback='unchanged original single-request decoder', full_shadow_until_eos_or_128=True))
        self.save('runtime/judge_development_pairs.json', dict(pairs=pairs, development_only=True,
            test_used_for_selection=False))
        del model, proc, detector
        self.release()

    def profile_judge(self, model, tokenizer):
        pairs = self.read('runtime/judge_development_pairs.json')
        development = {r['id']: r for r in self.dev}
        prompts = []
        for pair in pairs['pairs']:
            source = development.get(pair['id'])
            c.require(source and source['role'] == pair['role'] in ('fit', 'calibration') and
                pair['question'] == source['question'] and pair['id'] not in self.tests,
                'test prompt entered judge execution selection')
            prompts.append(tokenizer.apply_chat_template([dict(role='user', content=pair['question']),
                dict(role='assistant', content=pair['answer'])], tokenize=False))
        prompts = list(dict.fromkeys(prompts))
        lengths = prompt_lengths(tokenizer, prompts)
        order = sorted(range(len(prompts)), key=lambda i: (lengths[i], i))
        selected = set(order[-16:])
        for index in [order[round(i * (len(order) - 1) / 127)] for i in range(128)] + order:
            if len(selected) >= min(128, len(prompts)):
                break
            selected.add(index)
        prompts = [prompts[i] for i in sorted(selected)]
        c.require(len(prompts) >= 6, 'insufficient distinct development review inputs')
        lengths = prompt_lengths(tokenizer, prompts)
        grid = [size for size in (6, 16, 32, 64) if size <= len(prompts)]
        profile = dict(complete=False, test_used_for_selection=False, development_only=True,
            samples=len(prompts), prompt_sha256=[hashlib.sha256(p.encode()).hexdigest() for p in prompts],
            max_new_tokens=128, input_truncation=False, measurements=[])
        state = (self.torch.get_rng_state().clone(), self.torch.cuda.get_rng_state().clone())
        reference = None
        for repeat, sizes in ((0, grid), (1, list(reversed(grid)))):
            for size in sizes:
                self.torch.set_rng_state(state[0])
                self.torch.cuda.set_rng_state(state[1])
                self.torch.cuda.empty_cache()
                self.torch.cuda.synchronize()
                clock = time.perf_counter()
                item = dict(batch_size=size, repeat=repeat)
                try:
                    raw = scheduled_texts(model, tokenizer, prompts, batch_size=size,
                        bucket=size != 6, lengths=lengths)
                    self.torch.cuda.synchronize()
                    if reference is None:
                        reference = raw
                    item.update(status='OK', raw=raw, seconds=time.perf_counter() - clock,
                        comparison=comparison(reference, raw, runner.parse_response_safety))
                except self.torch.OutOfMemoryError:
                    item.update(status='OOM', seconds=time.perf_counter() - clock)
                    gc.collect()
                    self.torch.cuda.empty_cache()
                item['rng_unchanged'] = self.torch.equal(state[0], self.torch.get_rng_state()) and self.torch.equal(
                    state[1], self.torch.cuda.get_rng_state())
                profile['measurements'].append(item)
        self.torch.set_rng_state(state[0])
        self.torch.cuda.set_rng_state(state[1])
        profile['complete'] = True
        self.save('runtime/judge_profile.json', profile)
        decision = choose_judge(profile)
        decision.update(profile_sha256=common.sha(self.path('runtime/judge_profile.json')),
            pairs_sha256=common.sha(self.path('runtime/judge_development_pairs.json')))
        self.save('runtime/judge_decision.json', decision)
        return decision

    def review(self):
        records = runner.validate_commits(self.records('test'), self.tests, complete=True)
        model, tokenizer = self.teacher()
        model.requires_grad_(False)
        # Development-only execution choice; no test prompts or labels are passed to this function.
        if self.path('runtime/judge_decision.json').exists():
            decision = self.read('runtime/judge_decision.json')
            c.require(decision['profile_sha256'] == common.sha(self.path('runtime/judge_profile.json')) and
                decision['pairs_sha256'] == common.sha(self.path('runtime/judge_development_pairs.json')) and
                all(decision[k] == v for k, v in choose_judge(self.read('runtime/judge_profile.json')).items()),
                'frozen judge execution changed on resume')
        else:
            decision = self.profile_judge(model, tokenizer)
        pending = {}
        for rid, record in records.items():
            for arm in record['arms']:
                question, answer = self.tests[rid]['question'], arm['answer']
                key = runner.judgment_key(question, answer)
                if self.path('judgments/' + key + '.json').exists():
                    self.judgment(model, tokenizer, question, answer)
                else:
                    prompt = tokenizer.apply_chat_template([dict(role='user', content=question),
                        dict(role='assistant', content=answer)], tokenize=False)
                    pending[key] = prompt
        keys, prompts = list(pending), list(pending.values())
        lengths = prompt_lengths(tokenizer, prompts)
        order = sorted(range(len(prompts)), key=lambda i: (lengths[i], i)) if decision['bucket'] else list(range(len(prompts)))
        size = decision['batch_size']
        for start in range(0, len(order), size):
            indices = order[start:start + size]
            outputs = scheduled_texts(model, tokenizer, [prompts[i] for i in indices],
                batch_size=size, bucket=False)
            for i, raw in zip(indices, outputs):
                parsed = runner.parse_response_safety(raw)
                c.require(parsed['status'] in ('safe', 'unsafe', 'controversial'), 'unparseable answer review')
                self.save('judgments/' + keys[i] + '.json', dict(input_sha256=keys[i],
                    prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest(), judge_raw=raw, **parsed))
        for rid, record in records.items():
            labels = [dict(arm=arm['arm'], **self.judgment(model, tokenizer,
                self.tests[rid]['question'], arm['answer'])) for arm in record['arms']]
            if not self.path('records/answers/' + hashlib.sha256(rid.encode()).hexdigest() + '.json').exists():
                self.commit('answers', rid, dict(id=rid, labels=labels))
        del model, tokenizer
        self.release()
        self.report()


def experiment(root, rank=None):
    instance = Experiment(root, root / 'materials')
    instance.setup()
    if not instance.path('frozen.json').exists():
        instance.install_supplied()
    if rank is not None:
        instance.tests = {rid: row for i, (rid, row) in enumerate(instance.tests.items()) if i % c.WORLD == rank}
    return instance


def audit(instance, complete=False):
    frozen = instance.verify_frozen()
    c.require(frozen.get('supplied_state') is True and not instance.records('development') and
        not instance.records('labels'), 'onsite training artifacts are forbidden')
    for name in ('directions.json', 'calibration.json'):
        c.require(common.sha(instance.path(name)) == common.sha(instance.assets / 'supplied' / name),
            'supplied state changed')
    records = runner.validate_commits(instance.records('test'), instance.tests, complete=True)
    answers = {r['id']: r for r in instance.records('answers')}
    c.require(set(answers) == set(records), 'incomplete five-arm answer reviews')
    original_save, original_status = instance.save, instance.status
    instance.save = lambda *args, **kwargs: None
    instance.status = lambda *args, **kwargs: None
    try:
        report = instance.report()
    finally:
        instance.save, instance.status = original_save, original_status
    c.require(report == instance.read('report.json'), 'persisted safety report differs from recomputation')
    events = 0
    for record in records.values():
        for arm in record['arms']:
            events += runner.audit_events(arm, frozen['frozen_ns'])
    c.require(all(row['status'] in ('safe', 'unsafe', 'controversial')
        for answer in answers.values() for row in answer['labels']), 'invalid review parsing')
    live = instance.read('live_benchmark.json')
    suites = {r['id']: r['suite'] for r in instance.dev}
    c.require(live['development_only'] is True and live['stops_generation_on_selected_arm_alarm'] is True and
        len(live['measurements']) == 20 and {(suites[m['id']], m['arm']) for m in live['measurements']} ==
        {(suite, arm) for suite in SUITES for arm in runner.ARMS}, '20 live-stop development checks required')
    c.require(report['test_used_for_fitting_or_selection'] is False, 'test selection occurred')
    if complete:
        c.require(len(records) == 3360 and report['full_answer_reviews'] == 16800, 'full safety coverage required')
    by_source = {}
    for name in runner.ARMS:
        by_source[name] = {suite: runner.metrics([arm for record in records.values()
            for arm in record['arms'] if arm['arm'] == name and arm['suite'] == suite]) for suite in SUITES}
    return dict(primary=report['primary'], response_safety=report['response_safety'],
        by_source=by_source, token_events_checked=events, complete=True, onsite_training=False)
