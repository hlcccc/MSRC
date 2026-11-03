"""Opt-in scheduling for an unchanged, untruncated text safety judge."""
from contextlib import nullcontext


def encoded_judge_texts(model, tokenizer, prompts, *, batch_size=4,
                        order_by_length=False, budget=None, statistics=None):
    import torch

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('positive integer judge batch size required')
    if not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError('judge prompts must be strings')
    if not prompts:
        return []
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('judge tokenizer needs a padding or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, padding=False, truncation=False)
    if ('input_ids' not in encoded or 'attention_mask' not in encoded
            or any(len(values) != len(prompts) for values in encoded.values())):
        raise ValueError('one unpadded encoded record per prompt required')
    lengths = [len(ids) for ids in encoded['input_ids']]
    if any(length < 1 for length in lengths):
        raise ValueError('empty tokenized judge input')
    order = sorted(range(len(prompts)), key=lambda i: (lengths[i], i)) if order_by_length else list(range(len(prompts)))
    result = [None] * len(prompts)
    stats = statistics if statistics is not None else {}
    stats.update(input_tokens=sum(lengths), padded_input_slots=0, batches=[])
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        records = [{key: values[i] for key, values in encoded.items()} for i in indices]
        with budget.work('scheduled_answer_judge') if budget else nullcontext():
            inputs = tokenizer.pad(records, padding=True, return_tensors='pt').to(model.device)
            width = inputs['input_ids'].shape[1]
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            values = tokenizer.batch_decode(outputs[:, width:], skip_special_tokens=True)
        if len(values) != len(indices):
            raise ValueError('judge output count differs from submitted prompts')
        for i, value in zip(indices, values):
            result[i] = value
        stats['padded_input_slots'] += len(indices) * width
        stats['batches'].append(dict(indices=indices, input_width=width, size=len(indices)))
    if any(value is None for value in result):
        raise ValueError('missing scheduled judge output')
    return result


def padded_slots(lengths, batch_size):
    if type(batch_size) is not int or batch_size < 1 or any(n < 1 for n in lengths):
        raise ValueError('positive lengths and integer batch size required')
    return sum(len(part) * max(part) for start in range(0, len(lengths), batch_size)
               if (part := lengths[start:start + batch_size]))
