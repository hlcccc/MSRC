"""Fresh fixed-test evidence with unchanged BF16 models and supplied calibrators."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import gc
import json
import time

import numpy as np

from pipeline import common, fast3_contract as c


def scoped():
    from pipeline import factual
    factual.MODELS = {'internvl': common.model_paths()['internvl']}
    return factual


def prepare(root):
    factual = scoped()
    root.mkdir(parents=True, exist_ok=False)
    factual.prepare(root, 'full', common.ROOT / 'assets/fitted/factual.json')
    protocol = json.loads((root / 'protocol.json').read_text())
    protocol.update(onsite_training=False, evaluated_models=['internvl'], auxiliary_models=['qwen'],
        scope='two InternVL cells only; NOT original six-cell acceptance',
        expected_cells=['internvl_pope', 'internvl_textvqa'],
        batch_sampling_protocol='fixed global-block seeds; rank-independent; not legacy serial draws')
    common.save(root / 'protocol.json', protocol)
    for dataset in c.COUNTS:
        rows = common.read_rows(root / 'inputs' / (dataset + '.jsonl'))
        c.write_rows(root / 'test_inputs' / (dataset + '.jsonl'), c.fixed_tests(rows, dataset))


def package(root):
    from pipeline.factual_parameters import load_package
    scoped()
    protocol = json.loads((root / 'protocol.json').read_text())
    c.require(protocol['parameter_mode'] == 'provided' and protocol['onsite_training'] is False and
        common.sha(root / 'provided_parameters.json') == protocol['parameters_sha256'],
        'supplied factual parameters changed')
    for name, state in protocol['inputs'].items():
        raw = common.read_rows(root / 'inputs' / (name + '.jsonl'))
        c.require(common.sha(root / 'inputs' / (name + '.jsonl')) == state['sha256'] and
            common.read_rows(root / 'test_inputs' / (name + '.jsonl')) == c.fixed_tests(raw, name),
            'fixed factual inputs changed')
    return load_package(root / 'provided_parameters.json', root, protocol)


@contextmanager
def caches(provider, dataset):
    from pipeline.frozen_fast_runtime_5090 import ImagePreprocessCache, ExactBatchVisionCache
    from pipeline.frozen_request_runtime_5090 import PrefixForwardCache, auxiliary_batches
    original = provider._processor.image_processor
    cpu = ImagePreprocessCache(original)
    provider._processor.image_processor = cpu
    vision = prefix = None
    try:
        vision = ExactBatchVisionCache(provider, cpu)
        if dataset == 'textvqa':
            prefix = PrefixForwardCache(provider, cpu, allow_bounded_dynamic=True)
        with auxiliary_batches(c.BATCH['internvl'][dataset]):
            yield
    finally:
        if prefix is not None:
            prefix.close()
        if vision is not None:
            vision.close()
        provider._processor.image_processor = original


class PrefetchOCR:
    """One independent CPU engine warms exact OCR text while GPU collection runs."""
    def __init__(self, provider, rows):
        from concurrent.futures import ThreadPoolExecutor
        from rapidocr_onnxruntime import RapidOCR
        from msrc.providers import HFOCRProvider
        self.original = provider.ocr_provider
        self.engine = HFOCRProvider(engine=RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='msrc-cpu-ocr')
        self.values = {image: self.pool.submit(self.engine.ocr, image)
            for image in dict.fromkeys(row['image'] for row in rows)}

    def ocr(self, image):
        return self.values[image].result() if image in self.values else self.original.ocr(image)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def probes(root, dataset, n):
    rows = [r for r in common.read_rows(root / 'inputs' / (dataset + '.jsonl'))
        if r['acceptance_split'] == 'dev']
    indices = {round(i * (len(rows) - 1) / max(n - 2, 1)) for i in range(n - 1)}
    indices.add(max(range(len(rows)), key=lambda i: len(rows[i]['question'])))
    for index in range(len(rows)):
        if len(indices) >= n:
            break
        indices.add(index)
    return [rows[i] for i in sorted(indices)][:n]


def check_values(values, reference, donor=False):
    # These are identical batch layouts and identical RNG, not a loose metric gate.
    c.require(values == reference, 'development execution changed raw evidence; test not authorized')
    c.aligned([{k: v for k, v in row.items() if k not in ('answer', 'evidence', 'label',
        'answer_parsed', 'collection_warnings', 'model_calls', 'donor_only')} for row in reference],
        values, donor=donor)


def gather(provider, raw, family, dataset, budget):
    from pipeline.batched_collection import collect_chunk
    size = c.BATCH[family][dataset]
    return [row for start in range(0, len(raw), size) for row in collect_chunk(provider,
        raw[start:start + size], batch_size=size, donor_only=family == 'qwen', budget=budget, adaptive=False)]


def collect(root, rank, family):
    from pipeline.batch_training import make_provider
    from pipeline.batched_collection import collect_chunk
    from pipeline.frozen_fast_runtime_5090 import FastBudget
    from pipeline.preflight import check_gpu
    import torch
    package(root)
    check_gpu(ignore_current_process=True)
    torch.set_num_threads(4)
    budget = FastBudget(torch, 30)
    provider = make_provider(family, budget)
    gate_path = root / (family + '_quality.json')
    gates = {'development_only': True, 'onsite_training': False, 'cells': {}}
    # Complete both development gates before collecting either test cell.
    for dataset in c.COUNTS:
        size = c.BATCH[family][dataset]
        raw = probes(root, dataset, size if family == 'qwen' else 8)
        seed = c.seed_for(dataset, -1, family)
        torch.manual_seed(seed)
        reference = gather(provider, raw, family, dataset, budget)
        reference_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
        torch.manual_seed(seed)
        with ExitStack() as stack:
            if family == 'internvl':
                stack.enter_context(caches(provider, dataset))
                ocr = PrefetchOCR(provider, raw)
                provider.ocr_provider = ocr
                stack.callback(lambda: setattr(provider, 'ocr_provider', ocr.original))
                stack.callback(ocr.close)
            candidate = gather(provider, raw, family, dataset, budget)
        check_values(candidate, reference, donor=family == 'qwen')
        c.require(torch.equal(reference_rng[0], torch.get_rng_state()) and
            torch.equal(reference_rng[1], torch.cuda.get_rng_state()), 'development RNG consumption changed')
        gates['cells'][dataset] = dict(passed=True, ids=[r['id'] for r in raw],
            batch_size=size, full_raw_evidence_exact=True, seed=seed,
            limitation='reference uses accepted batching, not historical serial generation')
    c.atomic(gate_path, dict(gates, complete=True))
    for dataset in c.COUNTS:
        raw = common.read_rows(root / 'test_inputs' / (dataset + '.jsonl'))
        output = root / 'blocks' / (family + '_' + dataset)
        output.mkdir(parents=True, exist_ok=True)
        selected = c.assigned(raw, dataset, rank)
        started = time.perf_counter()
        with ExitStack() as stack:
            if family == 'internvl':
                stack.enter_context(caches(provider, dataset))
                ocr = PrefetchOCR(provider, selected)
                provider.ocr_provider = ocr
                stack.callback(lambda: setattr(provider, 'ocr_provider', ocr.original))
                stack.callback(ocr.close)
            for block, part in c.blocks(raw, dataset, rank):
                path = output / ('%06d.json' % block)
                seed = c.seed_for(dataset, block, family)
                if path.exists():
                    saved = json.loads(path.read_text())
                    c.require(saved['seed'] == seed and saved['rank'] == rank and saved['block'] == block,
                        'wrong resumed block')
                    c.aligned(part, saved['rows'], donor=family == 'qwen')
                    continue
                # The seed makes atomic block resume independent of skipped work.
                torch.manual_seed(seed)
                values = gather(provider, part, family, dataset, budget)
                c.aligned(part, values, donor=family == 'qwen')
                c.atomic(path, dict(block=block, rank=rank, seed=seed, rows=values))
                print(json.dumps(dict(stage=family + '_' + dataset, rank=rank, block=block,
                    rows=len(part), elapsed_seconds=time.perf_counter() - started)), flush=True)
        c.atomic(root / (family + '_' + dataset + '_resources.json'), budget.snapshot())
    del provider
    gc.collect()
    torch.cuda.empty_cache()


class SelectedInputs:
    def __init__(self, root):
        self.root = root

    def __truediv__(self, name):
        c.require(name in ('inputs', 'evidence'), 'unexpected feature path')
        return self.root / ('test_inputs' if name == 'inputs' else name)


def evaluate(root):
    from pipeline import factual_parameters as fp
    factual, state = scoped(), package(root)
    original = fp.features
    fp.features = lambda out, family, dataset: original(SelectedInputs(root), family, dataset)
    try:
        factual.evaluate(root)
    finally:
        fp.features = original
    report = json.loads((root / 'report.json').read_text())
    c.require({v['cell'] for v in report['cells']} == {'internvl_' + d for d in c.COUNTS}, 'wrong scope')
    for cell in report['cells']:
        dataset = cell['cell'].split('_', 1)[1]
        raw, rows, readings, x, available, y, dev = original(SelectedInputs(root), 'internvl', dataset)
        c.aligned(raw, rows)
        c.aligned(raw, common.read_rows(root / 'evidence' / ('qwen_' + dataset + '.jsonl')), donor=True)
        c.require(not dev.any() and cell['n_test'] == c.COUNTS[dataset][1] and cell['n_dev'] == 0,
            'test coverage or split changed')
        supplied = state['states'][cell['cell']]
        c.require(json.loads((root / 'calibrators' / (cell['cell'] + '.json')).read_text()) == supplied,
            'calibrator was refitted or replaced')
        live = np.asarray(supplied['live'], dtype=bool)
        active = np.zeros(len(factual.SIGNALS), dtype=bool)
        active[live] = np.abs(supplied['coef'][1:]) > 1e-12
        baseline = factual.SIGNALS.index('sequence_confidence')
        c.require(available[:, baseline].all(), 'original confidence baseline missing')
        scores = fp.predict(supplied, x, available)
        before, after = factual.ece(y, x[:, baseline]), factual.ece(y, scores)
        gain = (before - after) / before if before else None
        for key, value in [('ece_before', before), ('ece_after', after), ('relative_improvement', gain)]:
            c.require(cell[key] is None if value is None else np.isclose(cell[key], value, atol=1e-10),
                'global ECE recomputation differs')
        c.require(cell['minimum_actual_signals_per_test_sample'] == int((available & active).sum(1).min()) and
            cell['calibration_pass'] == bool(gain is not None and gain >= .10), 'threshold/signal audit differs')
        details = common.read_rows(root / 'signal_usage' / (cell['cell'] + '.jsonl'))
        c.require([v['id'] for v in details] == [v['id'] for v in raw], 'signal report lost a test sample')
        weights = np.zeros(len(factual.SIGNALS))
        weights[live] = supplied['coef'][1:]
        for i, detail in enumerate(details):
            c.require(detail['answer'] == rows[i]['answer'] and detail['label'] == int(y[i]) and
                np.isclose(detail['risk'], scores[i], atol=1e-10) and
                detail['n_actual_signals'] == int((available[i] & active).sum()) and
                [v['name'] for v in detail['signals']] == factual.SIGNALS, 'per-sample signal audit differs')
            for j, signal in enumerate(detail['signals']):
                c.require(signal['available'] == bool(available[i, j]) and
                    signal['participates'] == bool(available[i, j] and active[j]) and
                    signal['weight'] == weights[j] and signal['risk'] == readings[i][signal['name']].risk,
                    'raw signal/weight/participation differs')
    c.atomic(root / 'no_training_audit.json', dict(complete=True, onsite_training=False,
        supplied_parameters_equal=True, global_metrics_recomputed=True, fixed_test_counts={d: n[1] for d, n in c.COUNTS.items()}))
    return report
