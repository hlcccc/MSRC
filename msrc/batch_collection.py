"""Collect the original evidence requests in batches; only primary scores are used."""
from __future__ import annotations

from msrc.provider import PromptSet
from msrc.providers.batched import generate_many
from msrc.signals import Evidence
from scripts.collect_evidence import vqa_label, yes_no_label


def primary_prompt(row):
    question = row['question']
    return question if row.get('answer_format') == 'yes_no' else question + '\nAnswer with a short phrase only.'


def collect_chunk(provider, rows, *, batch_size, k=3, donor_only=False, budget=None, adaptive=True):
    if not rows or any(row.get('answer') or row.get('evidence') for row in rows):
        raise ValueError('fresh raw inputs without answers/evidence required')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate sample ID')
    images = [row['image'] for row in rows]
    primary = generate_many(provider, images, [primary_prompt(row) for row in rows], batch_size=batch_size,
                            collect_scores=not donor_only, budget=budget, adaptive=adaptive, stage='primary')
    if donor_only:
        return [dict(row, answer=sample.text, model_calls=1, donor_only=True)
                for row, sample in zip(rows, primary)]
    prompts = PromptSet()
    def extra(templates, stage, sampled=False):
        requests = [(row['image'], template.format(q=row['question'], a=sample.text))
                    for row, sample in zip(rows, primary) for template in templates]
        values = generate_many(provider, [r[0] for r in requests], [r[1] for r in requests],
                               batch_size=batch_size, do_sample=sampled, collect_scores=False,
                               budget=budget, adaptive=adaptive, stage=stage)
        return [values[i * len(templates):(i + 1) * len(templates)] for i in range(len(rows))]
    views = extra(prompts.views, 'views')
    resamples = extra([prompts.views[0]] * k, 'resamples', sampled=True)
    verifications = extra(prompts.probes, 'verifications')
    results = []
    for i, (row, sample) in enumerate(zip(rows, primary)):
        gold = row.get('gold_answers') or []
        if not gold:
            raise ValueError('reference answers required')
        if row.get('answer_format') == 'yes_no':
            label, parsed = yes_no_label(sample.text, gold[0], parser=row.get('answer_parser', 'legacy'))
        else:
            label, parsed = vqa_label(sample.text, gold), True
        warnings = []
        try:
            ocr_texts = list(provider.ocr(row['image']))
        except Exception as error:
            # Match gather_evidence: unavailable OCR stays missing, not a made-up reading.
            ocr_texts = []
            warnings.append(dict(stage='ocr', error=type(error).__name__))
        evidence = Evidence(question=row['question'], answer=sample.text, image=row['image'], primary=sample,
                            views=views[i], resamples=resamples[i], verifications=verifications[i],
                            ocr_texts=ocr_texts, cross_model=[])
        results.append(dict(row, answer=sample.text, label=label, answer_parsed=parsed,
                            evidence=evidence.to_dict(), collection_warnings=warnings,
                            model_calls=1 + len(prompts.views) + k + len(prompts.probes) + (not warnings)))
    return results
