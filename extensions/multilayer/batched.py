"""Batch only the independent prefix/answer teacher, not the causal online gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

from extensions.multilayer import runner, variants
from pipeline import common
from pipeline.batch_judge import judge_texts


class BatchedExperiment(variants.VariantExperiment):
    def __init__(self, root, assets):
        super().__init__(root, assets)
        self.judge_batch_size = int(os.environ.get('MSRC_JUDGE_BATCH_SIZE', '4'))
        if not 1 <= self.judge_batch_size <= 32:
            raise ValueError('judge batch size must be between 1 and 32')
        options = dict(judge_max_batch_size=self.judge_batch_size,
                       online_decoder='unchanged audited single-request causal gate')
        path = self.root / 'batch_execution.json'
        if path.exists() and runner.load(path) != options:
            raise ValueError('frozen judge execution options changed')
        if not path.exists():
            common.save(path, options)

    def sources(self):
        result = super().sources()
        for name in ('extensions/multilayer/batched.py', 'pipeline/batch_judge.py'):
            result[name] = common.sha(common.ROOT / name)
        result['batch_execution.json'] = common.sha(self.root / 'batch_execution.json')
        return result

    def precache(self, pairs, stage):
        model, tokenizer = self.teacher()
        missing = {}
        for question, answer in pairs:
            key = runner.judgment_key(question, answer)
            name = 'judgments/' + key + '.json'
            prompt = tokenizer.apply_chat_template([dict(role='user', content=question),
                                                   dict(role='assistant', content=answer)], tokenize=False)
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            if self.path(name).exists():
                # The original validator verifies exact prompt bytes and parsing.
                self.judgment(model, tokenizer, question, answer)
            else:
                missing[key] = dict(name=name, prompt=prompt, prompt_sha256=digest)
        values = list(missing.items())
        if stage == 'BATCH_PREFIX_JUDGE' and values and not self.path('runtime/judge_batch_smoke.json').exists():
            probes = [value['prompt'] for _, value in values[:8]]
            serial = judge_texts(model, tokenizer, probes, batch_size=1)
            batched = judge_texts(model, tokenizer, probes, batch_size=self.judge_batch_size)
            matches = [runner.parse_response_safety(a) == runner.parse_response_safety(b)
                       for a, b in zip(serial, batched)]
            self.save('runtime/judge_batch_smoke.json', dict(development_only=True,
                samples=len(probes), parser_matches=matches, serial_raw=serial, batched_raw=batched))
            if not all(matches):
                raise ValueError('development judge batch/serial parsing differs; inspect before training')
        for start in range(0, len(values), self.judge_batch_size):
            part = values[start:start + self.judge_batch_size]
            outputs = judge_texts(model, tokenizer, [value['prompt'] for _, value in part],
                                  batch_size=self.judge_batch_size)
            for (key, value), raw in zip(part, outputs):
                self.save(value['name'], dict(input_sha256=key, prompt_sha256=value['prompt_sha256'],
                    judge_raw=raw, **runner.parse_response_safety(raw)))
            if start % (self.judge_batch_size * 10) == 0:
                self.status(stage, completed=start + len(part), total=len(values))
        self.save('runtime/' + stage + '_batch.json', dict(max_batch_size=self.judge_batch_size,
            unique_uncached_pairs=len(values), actual_batches=getattr(model, '_msrc_judge_batches', [])))
        del model, tokenizer
        self.release()

    def label(self):
        sources = {row['id']: row for row in self.dev}
        records = self.records('development')
        if {row['id'] for row in records} != set(sources):
            raise ValueError('finish all development requests before labeling')
        self.precache([(sources[record['id']]['question'], prefix['text'])
                       for record in records for prefix in record['prefixes']], 'BATCH_PREFIX_JUDGE')
        return super().label()

    def judge(self):
        self.verify_frozen()
        records = runner.validate_commits(self.records('test'), self.tests, complete=True)
        self.precache([(self.tests[key]['question'], arm['answer'])
                       for key, record in records.items() for arm in record['arms']], 'BATCH_ANSWER_JUDGE')
        return super().judge()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--phase', choices=['prepare', 'check', 'smoke', 'development', 'label', 'fit', 'test', 'judge', 'all'],
                        required=True)
    args = parser.parse_args()
    root = runner.run_root(args.run_id)
    common.initialize(root)
    import fcntl
    # Keep the lock alive through every stage, as in the original entry point.
    lock = common.inside(root / 'variant_execution.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.phase == 'prepare':
        variants.prepare(root, 'no_fit')
        return
    variants.check_materials(root, 'no_fit', images=args.phase == 'check')
    if args.phase == 'check':
        print('BATCH_SAFETY_MATERIALS_OK', flush=True)
        return
    experiment = BatchedExperiment(root, root / 'materials')
    experiment.setup()
    phases = ('smoke', 'development', 'label', 'fit', 'test', 'judge') if args.phase == 'all' else (args.phase,)
    for phase in phases:
        experiment.status('START_' + phase.upper())
        value = getattr(experiment, 'full_test' if phase == 'test' else phase)()
        if phase == 'fit' and (value is None or not all(value['new_detection_enabled'].values())):
            raise ValueError('development evidence insufficient; testing not authorized')
    if args.phase in ('judge', 'all'):
        variants.audit_safety(experiment, 'no_fit')
        print('BATCH_MULTILAYER_SAFETY_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
