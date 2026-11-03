"""Prepare only the four roles (three checkpoints) used by the fast3 scope."""
import argparse
import json
from pathlib import Path

from pipeline import common, fast3_contract as c


def prepare(package, model_file):
    from pipeline.prepared_bundle import restore
    common.initialize()
    assets = common.inside(common.ROOT / 'assets')
    restore(package, assets)
    mapping = json.loads(model_file.read_text(encoding='utf-8'))
    models = {}
    for role in ('internvl', 'qwen', 'generator', 'judge'):
        spec = mapping.get(role, mapping.get('qwen') if role == 'generator' else None)
        c.require(spec is not None, 'local model role missing: ' + role)
        value = spec['path'] if isinstance(spec, dict) else spec
        path = Path(value).expanduser()
        models[role] = str((model_file.resolve().parent / path).resolve())
    c.require(models['qwen'] == models['generator'], 'Qwen donor and safety generator must share pinned weights')
    common.save(common.ROOT / 'configs/paths.json', dict(gpu='0', runs_dir='runs', models=models,
        data=dict(factual=str(assets / 'data/factual'), safety=str(assets / 'data/safety'),
            mmsafety=str(assets / 'data/MM-SafetyBench/data'))))
    from pipeline.fast3 import verify_models
    verify_models()
    from pipeline.provided_assets import prepare as provided
    provided(check=True)
    print('FAST3_ASSETS_OK: three checkpoints; no LLaVA/SingGuard required', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-package', type=Path, required=True)
    parser.add_argument('--local-models', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.prepared_package, args.local_models)
