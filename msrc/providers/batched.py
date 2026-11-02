"""Batched HF generation with per-sequence EOS masking and score extraction.

The original single-request provider remains unchanged. Sampling consumes a
batch-dependent RNG stream; these outputs belong to a new experiment, not a
byte-identical replay of the historical serial run.
"""
from __future__ import annotations

from contextlib import nullcontext
import gc
from pathlib import Path

from msrc.types import Sample


def trim_at_eos(tokens, eos):
    result = []
    for token in tokens:
        result.append(int(token))
        if int(token) in eos:
            break
    return result


def score_readings(scores, token_rows, torch):
    """Exclude padding after the first EOS, but include EOS as the serial path does."""
    if not scores:
        return [(None, None, None) for _ in token_rows]
    device = scores[0].device
    lengths = torch.tensor([min(len(row), len(scores)) for row in token_rows], device=device)
    probability = torch.zeros(len(token_rows), dtype=torch.float64, device=device)
    entropy = probability.clone()
    margin = None
    for step, values in enumerate(scores):
        dist = values.float().softmax(-1)
        ids = torch.tensor([row[step] if step < len(row) else 0 for row in token_rows], device=device)
        valid = lengths > step
        probability += dist.gather(1, ids[:, None])[:, 0].double() * valid
        entropy += (-(dist * dist.clamp_min(1e-12).log()).sum(-1)).double() * valid
        if step == 0:
            top = dist.topk(min(2, dist.shape[-1]), dim=-1).values
            margin = top[:, 0] - top[:, 1] if top.shape[-1] == 2 else top[:, 0] * 0
    denominators = lengths.clamp_min(1)
    values = torch.stack([probability / denominators, entropy / denominators, margin.double()], -1).cpu().tolist()
    return [tuple(value) if length else (None, None, None)
            for value, length in zip(values, lengths.cpu().tolist())]


def generate_batch(provider, images, prompts, *, do_sample=False, collect_scores=True):
    if not images or len(images) != len(prompts):
        raise ValueError('a nonempty, aligned image/prompt batch is required')
    if provider.want_attention or provider.want_image_contrast:
        raise ValueError('batch adapter does not implement attention or image contrast')
    if provider._model_fn is not None:
        return [provider.generate(image, prompt, do_sample) for image, prompt in zip(images, prompts)]
    provider.load()
    torch = provider._torch
    from PIL import Image
    processor = provider._processor
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('batch padding needs a defined pad/eos token')
        tokenizer.pad_token = tokenizer.eos_token
    texts, pixels = [], []
    for image, prompt in zip(images, prompts):
        if not Path(image).is_file():
            raise FileNotFoundError(image)
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        texts.append(processor.apply_chat_template(messages, add_generation_prompt=True))
        with Image.open(image) as handle:
            pixels.append(handle.convert('RGB'))
    inputs = processor(images=pixels, text=texts, padding=True, return_tensors='pt').to(provider.device)
    kwargs = dict(max_new_tokens=provider.max_new_tokens, do_sample=bool(do_sample),
                  output_scores=collect_scores, return_dict_in_generate=True)
    if do_sample:
        kwargs.update(temperature=provider.temperature, top_p=provider.top_p)
    with torch.inference_mode():
        output = provider._model.generate(**inputs, **kwargs)
    eos = provider._model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    width = inputs['input_ids'].shape[-1]
    token_rows = [trim_at_eos(row, eos) for row in output.sequences[:, width:].cpu().tolist()]
    readings = score_readings(getattr(output, 'scores', None), token_rows, torch)
    answers = processor.batch_decode([row for row in token_rows], skip_special_tokens=True)
    lengths = inputs['attention_mask'].sum(-1).cpu().tolist()
    provider.calls += len(images)
    return [Sample(text=text.strip(), sequence_confidence=values[0], sequence_entropy=values[1],
                   first_token_margin=values[2], meta=dict(prompt_len=int(length), generated=len(tokens),
                                                        batch_size=len(images), scores_recorded=collect_scores))
            for text, values, length, tokens in zip(answers, readings, lengths, token_rows)]


def generate_many(provider, images, prompts, *, batch_size, do_sample=False,
                  collect_scores=True, budget=None, adaptive=True):
    if batch_size < 1 or len(images) != len(prompts):
        raise ValueError('positive batch size and aligned inputs required')
    def run(part_images, part_prompts):
        if provider._model_fn is not None:
            return generate_batch(provider, part_images, part_prompts, do_sample=do_sample,
                                  collect_scores=collect_scores)
        provider.load()
        torch = provider._torch
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        try:
            with budget.work('batch_generation') if budget else nullcontext():
                return generate_batch(provider, part_images, part_prompts, do_sample=do_sample,
                                      collect_scores=collect_scores)
        except torch.OutOfMemoryError:
            if not adaptive or len(part_images) == 1:
                raise
        # Restore before retrying; a failed sampled generation may already have consumed RNG.
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        gc.collect()
        torch.cuda.empty_cache()
        provider._batch_retries = getattr(provider, '_batch_retries', []) + [len(part_images)]
        split = len(part_images) // 2
        return run(part_images[:split], part_prompts[:split]) + run(part_images[split:], part_prompts[split:])
    results = []
    for start in range(0, len(prompts), batch_size):
        results.extend(run(images[start:start + batch_size], prompts[start:start + batch_size]))
    return results
