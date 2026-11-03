"""Opt-in, reversible hot-path trials; never change frozen inference sources."""
from __future__ import annotations

from functools import wraps


def prepare_directions(model):
    """Convert immutable supplied weights once; keep the original projection code."""
    import numpy as np

    result = dict(model)
    for name in ('weight', 'bias'):
        value = np.array(model[name], dtype=np.float64, copy=True)
        value.setflags(write=False)
        result[name] = value
    return result


class BoundedNativeRope:
    """Skip only a no-op dynamic update inside the pinned native generation path.

    Fresh input_ids and a binary attention mask produce native positions bounded
    by prompt width plus max_new_tokens. The original decorated RoPE body still
    runs, with the same buffers and arithmetic; custom positions/cache fall back.
    """

    KEYS = frozenset(('input_ids', 'attention_mask', 'pixel_values', 'max_new_tokens',
        'do_sample', 'output_scores', 'return_dict_in_generate', 'temperature', 'top_p'))

    def __init__(self, provider):
        self.provider, self.torch, self.model = provider, provider._torch, provider._model
        self.rotary = self.model.model.language_model.rotary_emb
        if self.rotary.rope_type != 'dynamic':
            raise ValueError('only the frozen Qwen2 dynamic RoPE is supported')
        self.original_forward, self.original_generate = self.rotary.forward, self.model.generate
        self.had_forward = 'forward' in self.rotary.__dict__
        self.had_generate = 'generate' in self.model.__dict__
        function = getattr(self.original_forward, '__func__', None)
        dynamic = getattr(function, '__wrapped__', None)
        raw = getattr(dynamic, '__wrapped__', None)
        if raw is None or hasattr(raw, '__wrapped__'):
            raise ValueError('unexpected RoPE decorator layout')
        if raw.__module__ != 'transformers.models.qwen2.modeling_qwen2':
            raise ValueError('unknown RoPE implementation')
        self.raw = raw
        self.limit = int(self.rotary.original_max_seq_len)
        self.frequency = self.rotary.inv_freq
        self.original_frequency = self.rotary.original_inv_freq
        self.frequency_version = self.frequency._version
        self.original_frequency_version = self.original_frequency._version
        self.scaling = self.rotary.attention_scaling
        self.parameters = repr(self.rotary.config.rope_parameters)
        if (int(self.rotary.max_seq_len_cached) != self.limit or
                not self.torch.equal(self.frequency, self.original_frequency)):
            raise ValueError('RoPE is not at its unchanged original scale')
        self.bound = 0
        self.generations = self.fast_forwards = self.native_forwards = self.fallbacks = 0
        self.maximum_bound = 0

        @wraps(self.original_generate)
        def generate(*args, **kwargs):
            previous = self.bound
            self.bound = self.generation_bound(args, kwargs)
            self.generations += bool(self.bound)
            self.fallbacks += not bool(self.bound)
            self.maximum_bound = max(self.maximum_bound, self.bound)
            try:
                return self.original_generate(*args, **kwargs)
            finally:
                self.bound = previous

        @wraps(self.original_forward)
        def forward(x, position_ids, *args, **kwargs):
            if (self.bound and not args and not kwargs and self.stable()
                    and position_ids.ndim == 2 and x.shape[0] == position_ids.shape[0]
                    and position_ids.shape[-1] <= self.bound):
                self.fast_forwards += 1
                with self.torch.no_grad():
                    return self.raw(self.rotary, x, position_ids)
            self.native_forwards += 1
            return self.original_forward(x, position_ids, *args, **kwargs)

        self.rotary.forward, self.model.generate = forward, generate

    def stable(self):
        r = self.rotary
        return (r.rope_type == 'dynamic' and int(r.max_seq_len_cached) == self.limit
            and int(r.original_max_seq_len) == self.limit
            and r.inv_freq is self.frequency and self.frequency._version == self.frequency_version
            and r.original_inv_freq is self.original_frequency
            and self.original_frequency._version == self.original_frequency_version
            and r.attention_scaling == self.scaling)

    def generation_bound(self, args, kwargs):
        th = self.torch
        ids, mask = kwargs.get('input_ids'), kwargs.get('attention_mask')
        tokens = kwargs.get('max_new_tokens')
        config = self.model.generation_config
        defaults = (config._get_default_generation_params()
            if hasattr(config, '_get_default_generation_params') else {})

        def effective(name, fallback):
            value = getattr(config, name, None)
            return defaults.get(name, fallback) if value is None else value

        if (args or not set(kwargs) <= self.KEYS or not self.stable()
                or repr(self.rotary.config.rope_parameters) != self.parameters
                or self.model.training or th.is_grad_enabled()
                or not isinstance(ids, th.Tensor) or ids.ndim != 2 or ids.shape[-1] < 2
                or not isinstance(mask, th.Tensor) or mask.shape != ids.shape
                or type(tokens) is not int or tokens != self.provider.max_new_tokens
                or not 1 <= tokens <= 128
                or effective('use_cache', True) is not True
                or effective('cache_implementation', None) not in (None, 'dynamic')
                or effective('num_beams', 1) != 1
                or effective('num_return_sequences', 1) != 1
                or any(getattr(config, name, None) is not None for name in (
                    'assistant_early_exit', 'prompt_lookup_num_tokens', 'dola_layers',
                    'constraints', 'force_words_ids'))):
            return 0
        bound = ids.shape[-1] + tokens
        if bound > self.limit:
            return 0
        # One small prefill transfer replaces repeated per-token scalar syncs.
        cpu_mask = mask.detach().cpu()
        if not bool(th.all((cpu_mask == 0) | (cpu_mask == 1))):
            return 0
        return bound

    def close(self):
        if self.had_forward:
            self.rotary.forward = self.original_forward
        else:
            del self.rotary.forward
        if self.had_generate:
            self.model.generate = self.original_generate
        else:
            del self.model.generate

    def snapshot(self):
        return dict(guarded_generations=self.generations, unchanged_rope_body_calls=self.fast_forwards,
            native_rope_calls=self.native_forwards, generation_fallbacks=self.fallbacks,
            maximum_native_position_bound=self.maximum_bound, original_context_limit=self.limit,
            frequencies_changed=False, native_generation_and_sampling_unchanged=True)
