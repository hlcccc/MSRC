"""Maintainer CPU compatibility check on archived evidence, NOT a fresh GPU test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import inside, read_rows, save, sha
from pipeline.factual_parameters import load_package, predict
from pipeline.provided_assets import verify


def validate(package, factual_source, safety_source, out):
    package, factual_source, safety_source = map(Path, (package, factual_source, safety_source))
    out = inside(out)
    out.mkdir(parents=True, exist_ok=False)
    verify(package)
    protocol = json.loads((factual_source / 'protocol.json').read_text())
    params = load_package(package / 'factual.json', factual_source, protocol)
    results = []
    for cell, state in params['states'].items():
        details = read_rows(factual_source / 'signal_usage' / (cell + '.jsonl'))
        x = np.asarray([[s['risk'] for s in row['signals']] for row in details])
        available = np.asarray([[s['available'] for s in row['signals']] for row in details])
        if any([s['name'] for s in row['signals']] != state['signals'] for row in details):
            raise ValueError('archived signal order differs')
        risk = predict(state, x, available)
        maximum = float(np.max(np.abs(risk - np.asarray([row['risk'] for row in details]))))
        if maximum > 1e-12:
            raise ValueError('imported factual predictions differ from archived fitted state')
        labels = np.asarray([row['label'] for row in details])
        warn = risk >= state['warning_threshold']['value']
        tp, tn = int((warn & (labels == 1)).sum()), int((~warn & (labels == 0)).sum())
        fp, fn = int((warn & (labels == 0)).sum()), int((~warn & (labels == 1)).sum())
        results.append(dict(cell=cell, n=len(details), max_risk_difference=maximum,
                            threshold=state['warning_threshold']['value'], TP=tp, TN=tn, FP=fp, FN=fn,
                            accuracy=(tp + tn) / len(details)))
    import torch
    from pipeline import plugguard as pg
    torch.set_num_threads(2)
    pg.ROOT = out / 'safety_import'
    pg.ROOT.mkdir()
    pg.MODEL = Path(json.loads((safety_source / 'environment.json').read_text())['generator'])
    pg.image_hashes = json.loads((safety_source / 'input_hashes.json').read_text())['images']
    # Only the device placement differs; exercise the actual import/identity checks.
    pg.new_head = lambda: pg.head_class()(input_dim=3584, proj_dim=1024, mem_dim=1024, num_labels=2).float()
    pg.torch = SimpleNamespace(load=lambda path, **kwargs: torch.load(path, map_location='cpu', weights_only=True))
    head, threshold = pg.load_checkpoint(package / 'plugguard', read_rows(safety_source / 'inputs.jsonl'))
    if sha(pg.ROOT / 'head.pt') != sha(safety_source / 'head.pt') or head.training:
        raise ValueError('CPU safety checkpoint import differs')
    result = dict(status='FITTED_PARAMETERS_CPU_COMPATIBILITY_OK',
                  scope='archived evidence prediction equality and CPU checkpoint import only',
                  fresh_gpu_inference=False, new_acceptance_claim=False, factual_cells=results,
                  safety_head_sha256=sha(pg.ROOT / 'head.pt'), safety_threshold=threshold,
                  safety_checkpoint_loaded_on='cpu')
    save(out / 'validation.json', result)
    print(json.dumps(result, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--factual-source', type=Path, required=True)
    parser.add_argument('--safety-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    validate(args.package, args.factual_source, args.safety_source, args.out)


if __name__ == '__main__':
    main()
