import pytest


def test_batched_text_judge_preserves_order_and_does_not_truncate():
    torch = pytest.importorskip('torch')
    from pipeline.batch_judge import judge_texts

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2
        padding_side = 'right'

        def __call__(self, prompts, **kwargs):
            assert kwargs['padding'] is True and kwargs['truncation'] is False
            return Inputs(input_ids=torch.tensor([[0, int(prompt)] for prompt in prompts]),
                          attention_mask=torch.tensor([[0, 1] for _ in prompts]))

        def batch_decode(self, values, **kwargs):
            return [str(row[0].item()) for row in values]

    class Model:
        device = 'cpu'

        def generate(self, input_ids, attention_mask, **kwargs):
            assert kwargs == dict(max_new_tokens=128, do_sample=False)
            return torch.cat([input_ids, input_ids[:, -1:]], -1)

    tokenizer, model = Tokenizer(), Model()
    assert judge_texts(model, tokenizer, ['3', '4', '5', '6', '7'], batch_size=2) == ['3', '4', '5', '6', '7']
    assert model._msrc_judge_batches == [2, 2, 1]
    assert tokenizer.padding_side == 'left'


def test_invalid_judge_batch_refused():
    from pipeline.batch_judge import judge_texts
    with pytest.raises(ValueError):
        judge_texts(None, None, [], batch_size=0)
