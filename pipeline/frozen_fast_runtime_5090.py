"""Execution-only NVML and exact image caches; no prediction or fitting changes."""
from __future__ import annotations

from collections import OrderedDict
import ctypes
import ctypes.util
import hashlib
import json
import os
import subprocess
import time
from types import SimpleNamespace

from pipeline.batch_training import BatchBudget


class NvmlProbe:
    class Memory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                    ('used', ctypes.c_ulonglong)]

    class Utilization(ctypes.Structure):
        _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]

    def __init__(self, index=0):
        self.lib = ctypes.CDLL(ctypes.util.find_library('nvidia-ml') or 'libnvidia-ml.so.1')
        self.handle = ctypes.c_void_p()
        self.check(self.lib.nvmlInit_v2())
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self.handle)))
        name = ctypes.create_string_buffer(128)
        self.check(self.lib.nvmlDeviceGetUUID(self.handle, name, ctypes.c_uint(len(name))))
        self.uuid = name.value.decode('ascii')
        self.calls = 0
        self.seconds = 0.

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError('NVML resource query failed: ' + str(code))

    def query(self):
        memory, util = self.Memory(), self.Utilization()
        started = time.perf_counter()
        self.check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)))
        self.check(self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util)))
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return self.uuid, int(memory.free // 1024**2), int(util.gpu)


class FastBudget(BatchBudget):
    """Keep the original memory checks, sync points, cap and duty policy."""
    def __init__(self, *args, probe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe = probe or NvmlProbe(int(os.environ.get('MSRC_GPU', '0')))

    def before(self):
        waiting_since = time.monotonic()
        while True:
            uuid, free, util = self.probe.query()
            own = self.torch.cuda.memory_reserved(0) / 1024**2
            if free + own >= (self.cap_gib + self.reserve_gib) * 1024:
                break
            # The expensive process inventory is retained for memory-pressure cases.
            apps = subprocess.check_output(['nvidia-smi',
                '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
            other = [int(pid) for gpu, pid in (line.split(',') for line in apps.splitlines() if line.strip())
                     if gpu.strip() == uuid and int(pid) != os.getpid()]
            if self.shared_duty == 1. and not other:
                cap = min(self.cap_gib, int((free + own - self.reserve_gib * 1024) // 512) / 2)
                if cap < 18:
                    raise RuntimeError('insufficient usable exclusive GPU memory after CUDA initialization')
                previous = self.cap_gib
                total = self.torch.cuda.get_device_properties(0).total_memory
                self.torch.cuda.set_per_process_memory_fraction(cap * 1024**3 / total, 0)
                self.cap_gib = cap
                self.memory_adjustments.append(dict(previous_cap_gib=previous, effective_cap_gib=cap,
                    usable_mib=free + own, reserve_gib=self.reserve_gib,
                    reason='exclusive CUDA context overhead outside allocator', time_ns=time.time_ns()))
                break
            if time.monotonic() - waiting_since >= 120:
                raise RuntimeError('GPU memory pressure persisted for 120 seconds; foreign PIDs: ' + str(other))
            print('BATCH_MEMORY_WAIT ' + str((uuid, free, util)), flush=True)
            time.sleep(10)
        self.snapshots.append(dict(gpu_uuid=uuid, free_mib=free, utilization=util, time_ns=time.time_ns()))
        self.snapshots = self.snapshots[-32:]

    def snapshot(self):
        return dict(super().snapshot(), resource_probe='direct NVML; original pressure checks retained',
                    probe_calls=self.probe.calls, probe_seconds=self.probe.seconds)


class ImagePreprocessCache:
    """Cache exact per-image CPU outputs while leaving the official processor in use."""
    def __init__(self, original, max_bytes=512 * 1024**2):
        self.original, self.max_bytes = original, int(max_bytes)
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.calls = self.evictions = 0
        self.current_signature = None
        self.current_patch_count = 0
        self.identity_seconds = self.preprocess_seconds = 0.

    def __getattr__(self, name):
        return getattr(self.original, name)

    @staticmethod
    def image_key(image, settings):
        return (settings, image.mode, image.size, hashlib.sha256(image.tobytes()).digest())

    def _store(self, key, value):
        size = value.numel() * value.element_size()
        if key in self.entries:
            old = self.entries.pop(key)
            self.bytes -= old.numel() * old.element_size()
        if size > self.max_bytes:
            return
        while self.entries and self.bytes + size > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
            self.evictions += 1
        self.entries[key] = value
        self.bytes += size

    def __call__(self, images=None, **kwargs):
        import torch
        self.calls += 1
        self.current_signature = None
        if not isinstance(images, (list, tuple)) or not images or any(not hasattr(v, 'tobytes') for v in images):
            return self.original(images=images, **kwargs)
        settings = json.dumps(kwargs, sort_keys=True, default=str)
        started = time.perf_counter()
        keys = [self.image_key(image, settings) for image in images]
        self.identity_seconds += time.perf_counter() - started
        self.hits += sum(key in self.entries for key in keys)
        if all(key in self.entries for key in keys):
            selected = [self.entries[key] for key in keys]
            for key in keys:
                self.entries.move_to_end(key)
            output = dict(pixel_values=torch.cat(selected), num_patches=[v.shape[0] for v in selected])
        else:
            started = time.perf_counter()
            output = dict(self.original(images=images, **kwargs))
            self.preprocess_seconds += time.perf_counter() - started
            if set(output) != {'pixel_values', 'num_patches'}:
                return output
            pixels, counts = output['pixel_values'], [int(n) for n in output['num_patches']]
            if (not isinstance(pixels, torch.Tensor) or pixels.device.type != 'cpu'
                    or len(counts) != len(keys) or sum(counts) != len(pixels)):
                return output
            self.misses += len({key for key in keys if key not in self.entries})
            seen = {}
            for key, value in zip(keys, pixels.split(counts)):
                if key in seen and not torch.equal(seen[key], value):
                    raise ValueError('image preprocessing is not independent of duplicate batch position')
                seen[key] = value
            for key, value in seen.items():
                self._store(key, value.detach().clone().contiguous())
        pixels, counts = output['pixel_values'], [int(n) for n in output['num_patches']]
        self.current_signature = (tuple(keys), tuple(counts), tuple(pixels.shape), str(pixels.dtype))
        self.current_patch_count = sum(counts)
        return output

    def snapshot(self):
        return dict(cpu_bytes=self.bytes, max_cpu_bytes=self.max_bytes, hits=self.hits, misses=self.misses,
                    calls=self.calls, evictions=self.evictions, identity_seconds=self.identity_seconds,
                    preprocess_seconds=self.preprocess_seconds, entries=len(self.entries))


class ExactBatchVisionCache:
    """Reuse only an identical complete visual batch, without GPU-to-CPU hashing."""
    def __init__(self, provider, preprocess, max_bytes=512 * 1024**2):
        self.provider, self.preprocess = provider, preprocess
        self.model = provider._model.model
        self.original = self.model.get_image_features
        self.had_instance_override = 'get_image_features' in self.model.__dict__
        self.max_bytes = int(max_bytes)
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0
        self.model.get_image_features = self.features

    def features(self, pixel_values, vision_feature_layer=None, vision_feature_select_strategy=None, **kwargs):
        import torch
        signature = self.preprocess.current_signature
        if (signature is None or self.model.training or torch.is_grad_enabled()
                or pixel_values.shape[0] != self.preprocess.current_patch_count
                or kwargs.get('return_dict') is not True
                or kwargs.get('output_attentions') or kwargs.get('output_hidden_states')):
            return self.original(pixel_values=pixel_values, vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy, **kwargs)
        key = (signature, tuple(pixel_values.shape), str(pixel_values.dtype), str(pixel_values.device),
               repr(vision_feature_layer), repr(vision_feature_select_strategy), repr(sorted(kwargs.items())))
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return SimpleNamespace(pooler_output=self.entries[key])
        result = self.original(pixel_values=pixel_values, vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy, **kwargs)
        value = result.pooler_output.detach().clone()
        size = value.numel() * value.element_size()
        if size <= self.max_bytes:
            while self.entries and self.bytes + size > self.max_bytes:
                _, old = self.entries.popitem(last=False)
                self.bytes -= old.numel() * old.element_size()
                self.evictions += 1
            self.entries[key] = value
            self.bytes += size
        self.misses += 1
        return result

    def close(self):
        if self.had_instance_override:
            self.model.get_image_features = self.original
        else:
            del self.model.get_image_features
        self.entries.clear()
        self.bytes = 0

    def snapshot(self):
        return dict(gpu_bytes=self.bytes, max_gpu_bytes=self.max_bytes, hits=self.hits,
                    misses=self.misses, evictions=self.evictions, entries=len(self.entries),
                    exact_whole_batch_keys=True, gpu_pixels_copied_to_cpu_for_lookup=False)


def sampling_schedule(rows, chunk_size, k=3):
    """Recover native multinomial call shapes from committed generated lengths."""
    result = []
    if chunk_size < 1:
        raise ValueError('positive original chunk size required')
    for start in range(0, len(rows), chunk_size):
        part = rows[start:start + chunk_size]
        samples = []
        for row in part:
            evidence = row['evidence']
            if (len(evidence['views']), len(evidence['resamples']), len(evidence['verifications'])) != (4, k, 3):
                raise ValueError('complete unchanged evidence required for RNG reconstruction')
            samples.extend(evidence['resamples'])
        index = 0
        while index < len(samples):
            size = samples[index]['meta']['batch_size']
            group = samples[index:index + size]
            if (size < 1 or size > chunk_size or len(group) != size
                    or any(s['meta']['batch_size'] != size for s in group)
                    or any(not 1 <= s['meta']['generated'] <= 128 for s in group)):
                raise ValueError('invalid native sampling sub-batch metadata')
            result.append((size, max(s['meta']['generated'] for s in group)))
            index += size
    return result


def replay_sampling_rng(torch, schedule, *, seed, vocab_size):
    """Replay only the native CUDA sampler draws, never model inference or answers."""
    torch.manual_seed(seed)
    by_size = {}
    for size, steps in schedule:
        if size not in by_size:
            by_size[size] = torch.full((size, vocab_size), 1. / vocab_size, device='cuda', dtype=torch.float32)
        for _ in range(steps):
            torch.multinomial(by_size[size], num_samples=1)
    torch.cuda.synchronize()
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
