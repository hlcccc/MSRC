"""Stable length scheduling around the original untruncated answer judge."""
import re

from pipeline.batch_judge import judge_texts


def prompt_lengths(tokenizer, prompts):
    if not all(isinstance(value, str) for value in prompts):
        raise ValueError('judge prompts must be strings')
    if not prompts:
        return []
    encoded = tokenizer(prompts, padding=False, truncation=False)
    if len(encoded['input_ids']) != len(prompts):
        raise ValueError('one tokenized record per judge prompt required')
    lengths = [len(ids) for ids in encoded['input_ids']]
    if any(length < 1 for length in lengths):
        raise ValueError('empty judge input')
    return lengths


def batch_plan(lengths, size, *, bucket):
    if type(size) is not int or size < 1 or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError('positive integer lengths and batch size required')
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i)) if bucket else list(range(len(lengths)))
    return [order[start:start + size] for start in range(0, len(order), size)]


def scheduled_texts(model, tokenizer, prompts, *, batch_size, bucket, lengths=None, stats=None):
    lengths = prompt_lengths(tokenizer, prompts) if lengths is None else lengths
    if len(lengths) != len(prompts):
        raise ValueError('prompt length count differs')
    result = [None] * len(prompts)
    plan = batch_plan(lengths, batch_size, bucket=bucket)
    for indices in plan:
        values = judge_texts(model, tokenizer, [prompts[i] for i in indices],
                             batch_size=batch_size, adaptive=False)
        if len(values) != len(indices):
            raise ValueError('judge output count differs')
        for index, value in zip(indices, values):
            result[index] = value
    if any(value is None for value in result):
        raise ValueError('scheduled judge output missing')
    if stats is not None:
        stats.update(actual_batches=[len(part) for part in plan], input_tokens=sum(lengths),
            padded_input_slots=sum(len(part) * max(lengths[i] for i in part) for part in plan))
    return result


def comparison(expected, actual, parse):
    if len(expected) != len(actual):
        raise ValueError('judge comparison count differs')
    old, new = [parse(value) for value in expected], [parse(value) for value in actual]
    valid = all(value.get('status') in ('safe', 'unsafe', 'controversial') for value in old + new)
    refusal = lambda value: re.findall(r'^Refusal:\s*(Yes|No)\s*$', value, flags=re.MULTILINE)
    return dict(valid_parsed=valid, safety_labels_equal=valid and old == new,
        safety_differences=[i for i, (a, b) in enumerate(zip(old, new)) if a != b],
        raw_differences=[i for i, (a, b) in enumerate(zip(expected, actual)) if a != b],
        refusal_differences=[i for i, (a, b) in enumerate(zip(expected, actual)) if refusal(a) != refusal(b)])


def select_execution(profile, pending):
    if not profile.get('complete') or profile.get('test_used_for_selection') is not False:
        raise ValueError('complete development-only profile required')
    if type(pending) is not int or pending < 0:
        raise ValueError('invalid remaining review count')
    candidates = []
    for size in (16, 32):
        pairs = []
        for repeat in (0, 1):
            rows = {row['batch_size']: row for row in profile['measurements'] if row['repeat'] == repeat}
            if not {6, size} <= rows.keys():
                break
            a, b = rows[6], rows[size]
            if (a.get('status') != 'OK' or b.get('status') != 'OK' or not a.get('unchanged_rng')
                    or not b.get('unchanged_rng') or not b.get('comparison', {}).get('safety_labels_equal')):
                break
            pairs.append((a, b))
        if len(pairs) != 2 or pairs[0][0]['raw'] != pairs[1][0]['raw'] or pairs[0][1]['raw'] != pairs[1][1]['raw']:
            continue
        gains = [a['seconds'] / b['seconds'] - 1 for a, b in pairs]
        per_prompt = sum((a['seconds'] - b['seconds']) / profile['samples'] for a, b in pairs) / 2
        gross = per_prompt * pending
        net = .5 * gross - profile['elapsed_seconds']
        candidates.append(dict(batch_size=size, speedups=gains, projected_gross_saving_seconds=gross,
            conservative_net_saving_seconds=net, accepted=not profile.get('fatal_error')
                and min(gains) >= .10 and net >= 30))
    eligible = [row for row in candidates if row['accepted']]
    chosen = max(eligible, key=lambda row: row['conservative_net_saving_seconds']) if eligible else None
    return dict(accepted=chosen is not None, batch_size=chosen['batch_size'] if chosen else 6,
        bucket=chosen is not None, candidates=candidates, pending_unique_reviews=pending,
        profile_cost_seconds=profile['elapsed_seconds'], uncertainty_discount=.5,
        minimum_each_round_speedup=.10, minimum_conservative_net_saving_seconds=30,
        test_used_for_selection=False, original_warning_outputs_and_parameters_unchanged=True,
        policy='two repeatable development rounds; exact parsed safety labels; original 6 on failure')
