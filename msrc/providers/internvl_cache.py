"""Bounded, process-local image work cache for frozen HF InternVL inference.

Only image preprocessing and projected visual embeddings are reused. Prompts,
language-model KV states, generated answers and uncertainty scores are never
cached. Official InternVL placeholder expansion and tokenization stay in use.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import time


class InternVLInputCache:
    def __init__(self, provider, max_bytes=1024**3):
        if max_bytes <= 0:
            raise ValueError('a positive CPU cache bound is required')
        self.provider = provider
        self.model = provider._model
        self.processor = provider._processor
        self.max_bytes = int(max_bytes)
        self.entries = OrderedDict()
        self.bytes = 0
        self.stats = dict(preprocess_misses=0, preprocess_hits=0, vision_images=0,
                          vision_hits=0, evictions=0, preprocess_seconds=0., vision_seconds=0.)
        self.signature = self._signature()
        self._guard()

    def _signature(self):
        return json.dumps(dict(image=self.processor.image_processor.to_dict(),
            model=self.model.config.to_dict(), image_seq_length=self.processor.image_seq_length),
            sort_keys=True, default=str)

    def _guard(self):
        if (self.provider._model is not self.model or self.provider._processor is not self.processor
                or self.model.training or any(p.requires_grad for p in self.model.parameters())
                or self.provider.want_attention or self.provider.want_image_contrast):
            raise ValueError('image cache requires the same frozen, eval-only InternVL instance')
        if self._signature() != self.signature:
            raise ValueError('model or image preprocessing configuration changed during caching')

    def _settings(self):
        from transformers.models.internvl.processing_internvl import InternVLProcessorKwargs
        return self.processor._merge_kwargs(InternVLProcessorKwargs,
            tokenizer_init_kwargs=self.processor.tokenizer.init_kwargs,
            padding=True, return_tensors='pt')

    @staticmethod
    def identity(image):
        path = Path(image).resolve(strict=True)
        stat = path.stat()
        if not path.is_file():
            raise ValueError('image cache accepts regular local image files only')
        return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    @staticmethod
    def _entry_bytes(entry):
        return sum(value.numel() * value.element_size() for value in (entry['pixels'], entry['features'])
                   if value is not None)

    def _store(self, key, entry):
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous['accounted_bytes']
        entry['accounted_bytes'] = self._entry_bytes(entry)
        if entry['accounted_bytes'] <= self.max_bytes:
            self.entries[key] = entry
            self.bytes += entry['accounted_bytes']
        while self.bytes > self.max_bytes:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= removed['accounted_bytes']
            self.stats['evictions'] += 1

    def _image(self, image, settings):
        from PIL import Image
        key = self.identity(image)
        if key in self.entries:
            self.stats['preprocess_hits'] += 1
            self.entries.move_to_end(key)
            return key, self.entries[key]
        started = time.perf_counter()
        with Image.open(key[0]) as handle:
            processed = self.processor.image_processor(images=[handle.convert('RGB')], **settings)
        patches = processed['num_patches']
        if len(patches) != 1 or int(patches[0]) != processed['pixel_values'].shape[0]:
            raise ValueError('unexpected InternVL single-image patch layout')
        pixels = processed['pixel_values'].detach().cpu().contiguous()
        entry = dict(pixels=pixels, patches=int(patches[0]), features=None)
        self.stats['preprocess_misses'] += 1
        self.stats['preprocess_seconds'] += time.perf_counter() - started
        self._store(key, entry)
        return key, entry

    def prepare(self, images, texts):
        self._guard()
        if not images or len(images) != len(texts):
            raise ValueError('aligned, nonempty images and texts required')
        torch = self.provider._torch
        settings = self._settings()
        selected = [self._image(image, settings['images_kwargs']) for image in images]
        # Duplicate paths within a batch must use one preprocessing/encoding result.
        unique = OrderedDict(selected)
        missing = [(key, entry) for key, entry in unique.items() if entry['features'] is None]
        self.stats['vision_hits'] += sum(entry['features'] is not None for _, entry in selected)
        if missing:
            started = time.perf_counter()
            pixels = torch.cat([entry['pixels'] for _, entry in missing]).to(self.provider.device)
            with torch.inference_mode():
                features = self.model.get_image_features(pixel_values=pixels,
                    vision_feature_layer=self.model.config.vision_feature_layer,
                    vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
                    return_dict=True).pooler_output
            counts = [entry['patches'] for _, entry in missing]
            if features.shape[0] != sum(counts):
                raise ValueError('cached visual embeddings do not match image patch counts')
            for (key, entry), value in zip(missing, features.split(counts)):
                entry['features'] = value.detach().to('cpu', copy=True).contiguous()
                self._store(key, entry)
            self.stats['vision_images'] += len(missing)
            self.stats['vision_seconds'] += time.perf_counter() - started
            del pixels, features

        import numpy as np
        counts = [entry['patches'] for _, entry in selected]
        # This official helper only slices pixels while expanding the placeholders.
        # Empty-width patches avoid copying RGB tensors that generation no longer needs.
        expanded, _, used, videos = self.processor._insert_media_placeholders(
            texts, np.empty((sum(counts), 0), dtype=np.uint8), None,
            counts, [], np.cumsum(counts), np.array([0]), np.array([0]))
        if used != len(images) or videos:
            raise ValueError('cache supports exactly one still image per request')
        text_settings = dict(settings['text_kwargs'])
        text_settings.pop('return_tensors', None)
        return_types = text_settings.pop('return_mm_token_type_ids', None)
        text_inputs = self.processor.tokenizer(expanded, return_tensors='pt', **text_settings)
        self.processor._check_special_mm_tokens(expanded, text_inputs, modalities=['image'])
        if return_types:
            text_inputs['mm_token_type_ids'] = self.processor.create_mm_token_type_ids(text_inputs['input_ids'])
        inputs = {key: value.to(self.provider.device) for key, value in text_inputs.items()}
        with torch.inference_mode():
            embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
            visual = torch.cat([entry['features'] for _, entry in selected]).to(
                device=embeddings.device, dtype=embeddings.dtype)
            mask = self.model.model.get_placeholder_mask(inputs['input_ids'], embeddings, visual)
            inputs['inputs_embeds'] = embeddings.masked_scatter(mask, visual)
        return inputs

    def snapshot(self):
        return dict(self.stats, cpu_bytes=self.bytes, max_cpu_bytes=self.max_bytes,
                    entries=len(self.entries), answers_or_language_kv_cached=False)

    def estimate_tokens(self, image, prompt):
        _, entry = self._image(image, self._settings()['images_kwargs'])
        return entry['patches'] * self.processor.image_seq_length + len(
            self.processor.tokenizer.encode(prompt, add_special_tokens=False))
