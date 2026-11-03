"""Execution-only auxiliary batching and isolated first-forward cache trials."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import gc
import hashlib
import time


def comparable_rows(rows):
    """Only timestamps and auxiliary execution batch metadata may differ."""
    values = deepcopy(rows)
    for row in values:
        row.pop('collected_ns', None)
        for name in ('views', 'verifications'):
            for sample in row['evidence'][name]:
                sample.get('meta', {}).pop('batch_size', None)
    return values


@contextmanager
def auxiliary_batches(size):
    from pipeline import batched_collection
    original = batched_collection.generate_many
    current_size = [size]

    @wraps(original)
    def generate(provider, images, prompts, **kwargs):
        options = dict(kwargs)
        if not options.get('do_sample', False) and options.get('collect_scores') is False:
            options['batch_size'] = current_size[0]
        th = getattr(provider, '_torch', None)
        if th is None or not kwargs.get('adaptive', True):
            return original(provider, images, prompts, **options)
        cpu = th.get_rng_state()
        cuda = th.cuda.get_rng_state() if th.cuda.is_available() else None
        options['adaptive'] = False
        try:
            return original(provider, images, prompts, **options)
        except th.OutOfMemoryError:
            cache = getattr(provider, '_request_prefix_cache', None)
            if cache is not None:
                cache.disabled = True
                cache.entries.clear()
                cache.bytes = 0
            current_size[0] = kwargs['batch_size']
            th.set_rng_state(cpu)
            if cuda is not None:
                th.cuda.set_rng_state(cuda)
            gc.collect()
            th.cuda.empty_cache()
            retries = getattr(provider, '_request_runtime_fallbacks', [])
            provider._request_runtime_fallbacks = retries + [dict(
                batch_size=kwargs['batch_size'], do_sample=bool(kwargs.get('do_sample')),
                same_shape_original_runtime_retried_before_native_adaptive_policy=True)]
            # Retry the original complete request group before its existing OOM policy.
            return original(provider, images, prompts, **kwargs)

    batched_collection.generate_many = generate
    try:
        yield
    finally:
        batched_collection.generate_many = original


class PrefixForwardCache:
    """Clone a complete empty-cache forward result; decoding and sampling stay native."""
    def __init__(self, provider, preprocess, max_bytes=384 * 1024**2, allow_bounded_dynamic=False):
        self.provider, self.preprocess = provider, preprocess
        self.torch, self.model = provider._torch, provider._model
        config = getattr(self.model, 'config', None)
        text = getattr(config, 'text_config', config)
        rope = getattr(text, 'rope_parameters', None) or getattr(text, 'rope_scaling', None) or {}
        self.rope_type = rope.get('rope_type', rope.get('type', 'default'))
        self.rotary = None
        self.dynamic_bypasses = 0
        self.disabled = False
        if self.rope_type == 'dynamic' and allow_bounded_dynamic:
            self.rotary = self.model.model.language_model.rotary_emb
            if self.rotary.rope_type != 'dynamic':
                raise ValueError('dynamic RoPE module/config mismatch')
        elif self.rope_type != 'default':
            raise ValueError('stateful or scaled RoPE is outside the exact-prefix trial')
        self.original = self.model.forward
        self.had_override = 'forward' in self.model.__dict__
        self.max_bytes = int(max_bytes)
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.bypasses = self.evictions = 0
        self.lookup_seconds = self.clone_seconds = 0.
        self.errors = []

        @wraps(self.original)
        def forward(*args, **kwargs):
            return self.forward(*args, **kwargs)

        self.model.forward = forward
        self.provider._request_prefix_cache = self

    def states(self):
        states = [self.torch.get_rng_state().clone()]
        if self.torch.cuda.is_available():
            states.append(self.torch.cuda.get_rng_state().clone())
        return states

    def freeze(self, value, name=None):
        th = self.torch
        if isinstance(value, th.Tensor):
            shape = (tuple(value.shape), str(value.dtype), str(value.device), tuple(value.stride()))
            if name == 'pixel_values':
                if (self.preprocess.current_signature is None or value.ndim != 4
                        or value.shape[0] != self.preprocess.current_patch_count):
                    raise ValueError('unidentified image tensor')
                return ('image', shape, self.preprocess.current_signature)
            if value.numel() * value.element_size() > 1024**2:
                raise ValueError('unsupported large non-image input')
            raw = value.detach().contiguous().cpu().view(th.uint8).numpy().tobytes()
            return ('tensor', shape, hashlib.sha256(raw).digest())
        if value is None or isinstance(value, (bool, int, float, str)):
            return (type(value).__name__, value)
        if isinstance(value, (tuple, list)):
            return (type(value).__name__, tuple(self.freeze(v) for v in value))
        if isinstance(value, Mapping):
            return tuple((k, self.freeze(v, k)) for k, v in sorted(value.items()))
        raise ValueError('unsupported forward input: ' + type(value).__name__)

    def key(self, args, kwargs):
        th = self.torch
        past = kwargs.get('past_key_values')
        ids = kwargs.get('input_ids')
        if (self.disabled or args or self.model.training or th.is_grad_enabled()
                or not isinstance(ids, th.Tensor) or ids.ndim != 2 or ids.shape[-1] < 2
                or type(past).__name__ != 'DynamicCache' or past.get_seq_length() != 0
                or getattr(past, 'offloading', False)
                or kwargs.get('use_cache', True) is not True
                or kwargs.get('output_hidden_states') or kwargs.get('output_attentions')
                or kwargs.get('return_dict', True) is not True):
            return None
        try:
            rotary_key = None
            if self.rotary is not None:
                original = int(self.rotary.original_max_seq_len)
                cached = int(self.rotary.max_seq_len_cached)
                positions = kwargs.get('position_ids')
                length = int(positions.max()) + 1 if isinstance(positions, th.Tensor) else ids.shape[-1]
                if cached > original or length + self.provider.max_new_tokens > original:
                    self.dynamic_bypasses += 1
                    self.entries.clear()
                    self.bytes = 0
                    return None
                rotary_key = (cached, original, self.freeze(self.rotary.inv_freq),
                    self.freeze(self.rotary.attention_scaling))
            return (rotary_key, self.freeze({k: v for k, v in kwargs.items() if k != 'past_key_values'}))
        except (ValueError, TypeError):
            return None

    def tensor_bytes(self, output):
        seen = set()

        def count(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            if isinstance(value, self.torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, Mapping):
                return sum(count(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(count(v) for v in value)
            return sum(count(getattr(value, n)) for n in ('layers', 'keys', 'values')
                       if hasattr(value, n))

        return count(output)

    def forward(self, *args, **kwargs):
        started = time.perf_counter()
        key = self.key(args, kwargs)
        self.lookup_seconds += time.perf_counter() - started
        if key is None:
            self.bypasses += 1
            return self.original(*args, **kwargs)
        if key in self.entries:
            started = time.perf_counter()
            try:
                result = deepcopy(self.entries[key][0])
            except (RuntimeError, TypeError) as error:
                self.errors.append(repr(error))
                _, size = self.entries.pop(key)
                self.bytes -= size
            else:
                self.clone_seconds += time.perf_counter() - started
                self.hits += 1
                self.entries.move_to_end(key)
                return result
        self.misses += 1
        before = self.states()
        result = self.original(*args, **kwargs)
        after = self.states()
        size = self.tensor_bytes(result)
        if not all(self.torch.equal(a, b) for a, b in zip(before, after)):
            self.errors.append('prefill consumed random state; result not cached')
            return result
        if size > self.max_bytes:
            return result
        while self.entries and self.bytes + size > self.max_bytes:
            _, (_, removed) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1
        started = time.perf_counter()
        try:
            cached = deepcopy(result)
        except (RuntimeError, TypeError) as error:
            self.errors.append(repr(error))
        else:
            self.entries[key] = (cached, size)
            self.bytes += size
        self.clone_seconds += time.perf_counter() - started
        return result

    def close(self):
        if self.had_override:
            self.model.forward = self.original
        else:
            del self.model.forward
        self.entries.clear()
        self.bytes = 0
        if getattr(self.provider, '_request_prefix_cache', None) is self:
            del self.provider._request_prefix_cache

    def snapshot(self):
        return dict(hits=self.hits, misses=self.misses, bypasses=self.bypasses,
            evictions=self.evictions, gpu_bytes=self.bytes, max_gpu_bytes=self.max_bytes,
            lookup_seconds=self.lookup_seconds, clone_seconds=self.clone_seconds,
            errors=self.errors[:8], native_generation_and_multinomial_unchanged=True,
            cached_results_deep_copied=True, dynamic_rope_bypasses=self.dynamic_bypasses,
            rope_type=self.rope_type, disabled=self.disabled,
            dynamic_reuse_only_below_original_context_limit=True)
