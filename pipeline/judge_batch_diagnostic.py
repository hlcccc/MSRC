"""Reproduce a development-only batching discrepancy and inspect its logits."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from pipeline import common


def divergence(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int, default=21)
    args = parser.parse_args()
    common.initialize(common.inside(common.ROOT / 'diagnostic_runtime'))
    os.environ['MSRC_GPU'] = '0'
    common.select_device()
    output = common.inside(args.output)
    if output.exists():
        raise ValueError('retain previous diagnostic output')
    for path in (args.cohort, args.reference, args.outputs, args.model):
        real = path.resolve(strict=True)
        if Path.home() == Path('/home/HLC') and (not real.is_relative_to(Path('/mnt/data/HLC'))
                or real.stat().st_uid != os.getuid()):
            raise ValueError('HLC-owned diagnostic inputs required')
    cohort = json.loads(args.cohort.read_text(encoding='utf-8'))
    references = json.loads(args.reference.read_text(encoding='utf-8'))['outputs']
    old_outputs = json.loads(args.outputs.read_text(encoding='utf-8'))
    index, prompts = args.index, cohort['prompts']
    if not 0 <= index < len(prompts) or cohort['origins'][index]['role'] not in ('fit', 'calibration'):
        raise ValueError('development-only diagnostic index required')
    checkpoint = json.loads((common.ROOT / 'assets/plugguard/base_models.json').read_text())['judge']
    for name, digest in checkpoint['files'].items():
        if common.sha(args.model / name) != digest:
            raise ValueError('official checkpoint differs')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pipeline.batch_training import BatchBudget
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(common.SEED)
    budget = BatchBudget(torch, 28, shared_duty=.5)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    with budget.work('load_diagnostic_judge'):
        model = AutoModelForCausalLM.from_pretrained(str(args.model), dtype=torch.bfloat16,
            device_map='cuda:0', attn_implementation='sdpa', local_files_only=True).eval()
        model.requires_grad_(False)
    versions = {name: value._version for name, value in model.named_parameters()}
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    lengths = [len(row) for row in tokenizer(prompts, padding=False, truncation=False)['input_ids']]
    sorted_indices = sorted(range(len(prompts)), key=lambda i: (lengths[i], i))
    slot = sorted_indices.index(index)
    groups = dict(serial=[index], original_4=list(range(index // 4 * 4, min(index // 4 * 4 + 4, len(prompts)))),
                  bucket_8=sorted_indices[slot // 8 * 8:slot // 8 * 8 + 8])
    records, token_ids, selected_scores = {}, {}, {}
    for name, indices in groups.items():
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        target = indices.index(index)
        captures = []

        def capture(module, unused_args, kwargs):
            if captures:
                return
            mask, positions = kwargs.get('attention_mask'), kwargs.get('position_ids')
            tokens = kwargs.get('input_ids')
            cache = kwargs.get('past_key_values')
            record = dict(attention_mask_dimensions=mask.ndim if mask is not None else None,
                          initial_cached_tokens=int(cache.get_seq_length()) if cache is not None else 0)
            if mask is not None and mask.ndim == 2 and tokens is not None:
                active = mask[target].bool()
                record['nonpad_input_ids'] = tokens[target][active].detach().cpu().tolist()
                if positions is not None and positions.ndim == 2:
                    record['nonpad_position_ids'] = positions[target][active].detach().cpu().tolist()
            captures.append(record)

        handle = model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with budget.work('diagnose_' + name), torch.inference_mode():
                encoded = tokenizer([prompts[i] for i in indices], padding=True,
                    truncation=False, return_tensors='pt').to(model.device)
                result = model.generate(**encoded, max_new_tokens=128, do_sample=False,
                    return_dict_in_generate=True, output_scores=True)
                generated = result.sequences[target, encoded['input_ids'].shape[1]:]
                token_ids[name] = generated.detach().cpu().tolist()
                selected_scores[name] = [score[target].detach().float().cpu() for score in result.scores]
                raw = tokenizer.decode(generated, skip_special_tokens=True)
        finally:
            handle.remove()
        expected = (references[index] if name == 'serial' else
            old_outputs['original_0' if name == 'original_4' else 'length_bucket_8_0'][index])
        records[name] = dict(indices=indices, raw=raw, reproduces_original_trial_raw=raw == expected,
            input_capture=captures[0], unchanged_rng=torch.equal(cpu_rng, torch.get_rng_state()) and
                torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        del result, encoded, generated
    position = divergence(token_ids['serial'], token_ids['original_4'])
    detail = None
    if position is not None:
        serial_token = token_ids['serial'][position]
        batch_token = token_ids['original_4'][position]
        detail = dict(first_divergent_generated_token=position + 1,
            common_generated_prefix=token_ids['serial'][:position],
            serial_token_id=serial_token, original_batch4_token_id=batch_token,
            serial_token_text=tokenizer.decode([serial_token]), batch4_token_text=tokenizer.decode([batch_token]),
            logits={})
        for name, scores in selected_scores.items():
            if len(scores) <= position:
                continue
            vector = scores[position]
            top = torch.topk(vector, 5)
            detail['logits'][name] = dict(serial_token=float(vector[serial_token]),
                batch4_token=float(vector[batch_token]),
                serial_minus_batch4_margin=float(vector[serial_token] - vector[batch_token]),
                top5=[dict(token_id=int(i), text=tokenizer.decode([int(i)]), score=float(v))
                      for i, v in zip(top.indices, top.values)])
        a, b = selected_scores['serial'][position], selected_scores['original_4'][position]
        finite = torch.isfinite(a) & torch.isfinite(b)
        detail['maximum_finite_logit_absolute_difference'] = float((a[finite] - b[finite]).abs().max())
    inputs = [row['input_capture'].get('nonpad_input_ids') for row in records.values()]
    positions = [row['input_capture'].get('nonpad_position_ids') for row in records.values()]
    result = dict(development_diagnostic_not_acceptance=True, origin=cohort['origins'][index],
        prompt_sha256=hashlib.sha256(prompts[index].encode()).hexdigest(),
        source_hashes={str(path): common.sha(path) for path in (args.cohort, args.reference, args.outputs)},
        torch_version=torch.__version__, transformers_version=__import__('transformers').__version__,
        gpu=torch.cuda.get_device_name(0), groups=records, divergence=detail,
        nonpad_input_tokens_identical=all(row == inputs[0] and row is not None for row in inputs),
        nonpad_positions_identical=all(row == positions[0] and row is not None for row in positions),
        nonpad_positions_contiguous=all(row is not None and row == list(range(len(row))) for row in positions),
        empty_cache_at_each_fresh_generation=all(row['input_capture']['initial_cached_tokens'] == 0 for row in records.values()),
        parameter_versions_unchanged=versions == {name: value._version for name, value in model.named_parameters()},
        source_code_sha256=common.sha(Path(__file__)), weights_or_thresholds_changed=False)
    common.save(output, result)
    print('BATCH_DIAGNOSTIC_COMPLETE ' + str(output), flush=True)


if __name__ == '__main__':
    main()
