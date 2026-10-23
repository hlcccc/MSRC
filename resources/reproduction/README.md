# Public Reproduction Inputs

No model weights, trained risk heads, fitted calibrators, thresholds, model
answers, or cached scores are included here.

- `data_sources.json`: pinned public raw-data URLs and dataset commits.
- `inputs.sha256`: original raw/processed input and image hashes, not predictions.
- `benign.expected.json`: original benign instruction identity and image hashes.
- `DejaVuSans.ttf` and `FONT-LICENSE.txt`: the exact freely redistributable font
  used to render the original benign images.
- `models.json`: public model commits and expected file SHA256 values.
- `model_overrides.json`: small approved non-weight compatibility text files.
  Qwen's `configuration.json` is framework metadata. SingGuard's chat template
  is retained only for the existing asset-preparation compatibility role; it is
  never used by this branch's PlugGuard scoring path.

The benign builder is `pipeline/build_benign.py`: read only Alpaca instructions,
shuffle with seed `20261010`, select 1,638 instructions of 2--7 words without
digits, add 42 fixed benign lure phrases, shuffle again, assign five cyclic
MM-SafetyBench-style question templates, render 1024 x 180 RGB PNGs using the
frozen font, and verify all 1,680 original records/images. Labels are zero;
there are no old answers or split assignments. A mismatch stops preparation,
rather than silently replacing the benchmark or reusing its historical metrics.

Public dataset terms remain applicable: Alpaca and MM-SafetyBench are released
under CC BY-NC 4.0; COCO/TextVQA image rights remain with their respective owners.
Public checkpoints are downloaded from their original model repositories under
the applicable model licenses. The code does not upload or retrain them.

`scripts/export_public_reproduction.py` is a maintainer-only provenance exporter.
Third-party testers never run it or connect to the original experiment machine.
