# Prompt-state resampling-entropy estimator

This isolated experimental branch does not replace any original acceptance run.
It predicts the existing **K=3 normalized string-cluster entropy**, not semantic
entropy defined by bidirectional entailment. It is not output token entropy.

## Method

- Base model: the original frozen BF16 InternVL3-8B-hf checkpoint. No VLM training.
- Feature: final language-decoder block output, last prompt token, before final norm.
- One hook retains one vector from prefill. No extra model forward is needed when
  attached to fresh primary generation; padding must be on the left.
- Student: fixed seeded 64-dimensional random projection plus ridge regression,
  alpha=10. Centering/scaling and all fitting use development records only.
- A trained student replaces three sampled answers with one small matrix product.
- `object_support` also reads resamples. The new matched variant explicitly uses
  `object_support_views`, based only on the existing four rephrasings. Its values
  are not claimed identical to the original signal.
- The new signal is named `resample_entropy_estimate`, marked internal/estimated,
  not silently passed off as an observed external resampling entropy.
- Train a new matching fusion calibrator using image-group OOF student predictions
  on development data. Do not silently load the old original-method calibrator.
- A third, `observed_views` control retains observed entropy with views-only object
  support, separating the two changes in offline downstream analysis.
- Official BF16 weights, original prompts, generation limits, dataset and original
  development/test boundaries are preserved. NF4 is not validated by this branch.

## Pilot reproduction

From this repository, with its normal model configuration and dependencies:

```bash
python -m pipeline.entropy_estimator --source /path/to/previous/factual \
  --run-id entropy_pilot --phase prepare
python -m pipeline.entropy_estimator --source /path/to/previous/factual \
  --run-id entropy_pilot --phase run --cap-gib 19
```

The source must contain complete original inputs, InternVL evidence, Qwen donor
evidence and protocol. It is read-only. Historical evidence supplies regression
targets and offline downstream comparisons; it is **not fresh full acceptance**.
The selected cohort is fixed by seeded image-group hashes, never by labels or scores.
Default pilot sizes are at least 256 dev and 128 test per dataset, rounded up to
whole image groups. Image-file hashes guard cross-ID duplicate image leakage.

## Outputs

Under `runs/entropy_pilot/entropy_estimator/`:

- `estimator.npz`, `estimator.json`: trained student, target/feature definition,
  development training IDs/groups, model weight hashes and SHA256 verification.
- `matched_fusion_parameters.json`: new development-only matching fusion states.
- `crossfit_audit.json`, `frozen_before_test.json`: grouped OOF proof and pre-test freeze.
- `report.md`, `report.json`, `test_predictions.jsonl`: held-out entropy errors versus
  a dev-mean constant, downstream ECE gains and actual participating signal counts.
- `benchmark.json`: matched fresh evidence-generation benchmark, original 11
  versus surrogate 8 generations per question, including image preparation/OCR,
  excluding model load and the separate Qwen donor. Shared-A100 results are not
  exclusive-RTX5090 runtime claims. The cross-model donor signal is absent in both
  timing arms, not fabricated; offline downstream comparisons include the original
  donor evidence. Observed and estimated signal counts are reported separately.

For plugin use, compute `feature_contract(provider)` once for the frozen loaded
provider. Load with `ResampleEntropyEstimator.load(path, contract=contract)`, then
pass that same contract to `generate_with_entropy`. Moving identical checkpoint
bytes is supported; changing model weights, tokenizer, precision, preprocessing
or the feature definition requires a new verified estimator. The pilot also
records its source protocol and training cohort separately from that portable
inference contract.

The pilot must beat a constant entropy predictor and retain acceptable downstream
calibration on isolated evaluation before a larger rollout is justified. A
numerically passing pilot is not evidence that all 9,000/5,000 items pass.
Three generations saved out of eleven does not imply an elevenfold speedup.
