"""Batch independent answer-verifier requests; never truncate judge inputs."""
from contextlib import nullcontext
import gc


def judge_texts(model, tokenizer, prompts, *, batch_size=4, budget=None):
    import torch
    if batch_size < 1:
        raise ValueError('positive judge batch size required')
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('judge tokenizer has no padding or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    results = []
    offset, size = 0, batch_size
    stats = getattr(model, '_msrc_judge_batches', [])
    while offset < len(prompts):
        count = min(size, len(prompts) - offset)
        try:
            with budget.work('batch_answer_judge') if budget else nullcontext():
                inputs = tokenizer(prompts[offset:offset + count], padding=True,
                                   truncation=False, return_tensors='pt').to(model.device)
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                values = tokenizer.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            if len(values) != count:
                raise ValueError('judge output count differs from inputs')
            results.extend(values)
            stats.append(count)
            offset += count
        except torch.OutOfMemoryError:
            if count == 1:
                raise
            if 'inputs' in locals():
                del inputs
            gc.collect()
            torch.cuda.empty_cache()
            size = max(1, count // 2)
    model._msrc_judge_batches = stats
    return results
