from types import SimpleNamespace

import pytest

from msrc.providers.internvl_cache import InternVLInputCache
from msrc.providers.batched import generate_many
from msrc.types import Sample


class Tokenizer:
    init_kwargs = {}

    def encode(self, text, **kwargs):
        return [ord(char) % 20 for char in text]

    def __call__(self, texts, **kwargs):
        import torch
        rows = [[1] + [99] * text.count('<I>') + [sum(map(ord, text)) % 20 + 2] for text in texts]
        width = max(map(len, rows))
        return dict(input_ids=torch.tensor([[0] * (width-len(row)) + row for row in rows]),
                    attention_mask=torch.tensor([[0] * (width-len(row)) + [1] * len(row) for row in rows]))


def cached_provider(tmp_path, max_bytes=1024**2):
    torch = pytest.importorskip('torch')
    from PIL import Image
    class ImageProcessor:
        calls = 0
        scale = 1
        def to_dict(self):
            return dict(scale=self.scale)
        def __call__(self, images, **kwargs):
            self.calls += 1
            image = images[0]
            return dict(num_patches=[2], pixel_values=torch.ones(2, 3, 2, 2) * image.getpixel((0, 0))[0])

    class Processor:
        image_seq_length = 1
        tokenizer = Tokenizer()
        image_processor = ImageProcessor()
        def _insert_media_placeholders(self, texts, pixels, videos, counts, *args):
            if any(text.count('<I>') != 1 for text in texts):
                raise ValueError('one image placeholder required')
            return [text.replace('<I>', '<I>' * count) for text, count in zip(texts, counts)], [], len(texts), 0
        def _check_special_mm_tokens(self, *args, **kwargs):
            pass

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(100, 4)
            self.config = SimpleNamespace(vision_feature_layer=-1, vision_feature_select_strategy='default',
                                          to_dict=lambda: dict(vision_feature_layer=-1))
            self.model = SimpleNamespace(get_placeholder_mask=lambda ids, embeds, features: (ids == 99).unsqueeze(-1))
            self.vision_calls = 0
        def get_input_embeddings(self):
            return self.embed
        def get_image_features(self, pixel_values, **kwargs):
            self.vision_calls += 1
            return SimpleNamespace(pooler_output=pixel_values.mean((1, 2, 3))[:, None, None].expand(-1, 1, 4))
    model = Model().eval().requires_grad_(False)
    provider = SimpleNamespace(_model=model, _processor=Processor(), _torch=torch, device='cpu',
                               want_attention=False, want_image_contrast=False)
    cache = InternVLInputCache(provider, max_bytes=max_bytes)
    cache._settings = lambda: dict(images_kwargs={}, text_kwargs=dict(padding=True, return_tensors='pt'))
    image = tmp_path / 'one.png'
    Image.new('RGB', (4, 4), (20, 30, 40)).save(image)
    return provider, cache, str(image)


def test_image_features_reused_but_new_prompt_is_processed(tmp_path):
    provider, cache, image = cached_provider(tmp_path)
    first = cache.prepare([image], ['<I> first'])
    again = cache.prepare([image], ['<I> first'])
    other = cache.prepare([image], ['<I> second'])
    assert provider._model.vision_calls == provider._processor.image_processor.calls == 1
    assert provider._torch.equal(first['inputs_embeds'], again['inputs_embeds'])
    assert not provider._torch.equal(first['input_ids'], other['input_ids'])
    assert 'pixel_values' not in first and cache.snapshot()['vision_hits'] == 2


def test_duplicate_images_encoded_once_and_remain_aligned(tmp_path):
    provider, cache, image = cached_provider(tmp_path)
    values = cache.prepare([image, image], ['<I> a', '<I> b'])
    assert values['input_ids'].shape[0] == 2
    assert provider._model.vision_calls == 1
    assert cache.snapshot()['vision_images'] == 1
    assert provider._torch.equal(values['inputs_embeds'][0, 1:3], values['inputs_embeds'][1, 1:3])


def test_lru_bound_and_file_change_invalidate_image(tmp_path):
    from PIL import Image
    provider, cache, image = cached_provider(tmp_path, max_bytes=140)
    cache.prepare([image], ['<I> a'])
    old = cache.identity(image)
    Image.new('RGB', (4, 4), (40, 30, 20)).save(image)
    assert cache.identity(image) != old
    changed = cache.prepare([image], ['<I> a'])
    assert provider._model.vision_calls == 2
    assert changed['inputs_embeds'][0, 1, 0] == 40
    assert cache.bytes <= cache.max_bytes and cache.stats['evictions'] > 0


@pytest.mark.parametrize('change', ['train', 'grad', 'preprocessor', 'attention'])
def test_unsafe_model_or_configuration_changes_fail_closed(tmp_path, change):
    provider, cache, image = cached_provider(tmp_path)
    if change == 'train':
        provider._model.train()
    elif change == 'grad':
        provider._model.requires_grad_(True)
    elif change == 'preprocessor':
        provider._processor.image_processor.scale = 2
    else:
        provider.want_attention = True
    with pytest.raises(ValueError):
        cache.prepare([image], ['<I> a'])


def test_greedy_length_sort_restores_request_order_but_sampling_never_sorts():
    calls = []
    provider = SimpleNamespace(_model_fn=True, _batch_sort_greedy=True,
        want_attention=False, want_image_contrast=False,
        _batch_input_cache=SimpleNamespace(estimate_tokens=lambda image, prompt: len(prompt)))
    def generate(image, prompt, sampled):
        calls.append((prompt, sampled))
        return Sample(text=prompt)
    provider.generate = generate
    prompts = ['longest', 'x', 'medium']
    result = generate_many(provider, ['a'] * 3, prompts, batch_size=2)
    assert [row.text for row in result] == prompts
    assert [text for text, _ in calls] == ['x', 'medium', 'longest']
    calls.clear()
    result = generate_many(provider, ['a'] * 3, prompts, batch_size=2, do_sample=True)
    assert [text for text, _ in calls] == prompts
    assert [row.text for row in result] == prompts
