# Primary-statistics entropy control

This research-only variant runs independently of the prompt-hidden-state student.
It does not change the original acceptance scripts, weights, parameters or reports.

## Inputs And Fitting

- Target: the existing K=3 normalized string-cluster resampling entropy, not
  entailment-based semantic entropy and not primary token-distribution entropy.
- Seven cheap inputs: primary-answer confidence, primary token entropy, first-token
  margin, log token count, log prompt length, log answer length and dataset identity.
  No ground-truth answers, correctness labels, resamples or verification answers
  enter the estimator feature function.
- Fixed ridge alpha=10 and fixed seeded projection. Normalization and all fitting
  use only original development records. Four image-group folds supply OOF
  estimator predictions to new matching fusion calibrators.
- The original frozen BF16 InternVL is not trained. The estimator artifact format
  differs from the prompt-hidden-state artifact and cannot be loaded as that model.
- Removing the three resamples also removes their contribution to object support.
  The new variant uses four rephrasing views only. A separate observed-entropy,
  views-only control makes that dependency explicit.
- Model and matching fusion files are hashed before held-out evaluation. Original
  test data do not choose the features, hyperparameters or fitted parameters.

## Reproduction

This is a historical-evidence control, not a standalone fresh-inference workflow.
The read-only source must already contain the complete POPE/TextVQA inputs,
InternVL primary and auxiliary evidence, independent Qwen donor evidence and the
original factual protocol. Original image files must be accessible for SHA256
group checks. No new model download or GPU inference occurs here.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -B -m pipeline.primary_entropy_pilot \
  --source /path/to/completed/factual \
  --run-id primary_stats_entropy_001
```

Outputs are under `runs/primary_stats_entropy_001/primary_entropy_control/`:

- `estimator.npz` and `estimator.json`: fitted lightweight predictor and contract.
- `matched_fusion_parameters.json`: new development-only fusion parameters.
- `development_oof.npz`, `crossfit_audit.json`, `frozen_before_test.json`: fit/freeze evidence.
- `protocol.json`: source, code hashes, fixed settings and scope.
- `test_predictions.jsonl`, `report.json`, `report.md`: held-out errors and calibration.

## Recorded Control

Run `primary_stats_entropy_20261103_001` used all 14,000 historical records:
7,004 development and 6,996 held-out test records. Image-content groups crossing
the original development/test boundary: zero.

| Dataset | Dev | Test | Original ECE gain | Views-only control gain | Estimated ECE gain |
|---|---:|---:|---:|---:|---:|
| POPE | 4,500 | 4,500 | 64.65% | 64.65% | 64.02% |
| TextVQA | 2,504 | 2,496 | 68.28% | 67.86% | 68.29% |

The estimated variant has at least six active signal values per test record,
including one estimated value; at least five are directly observed. Numerical
signal-count and ECE thresholds remain satisfied for these two historical cells.
This does not establish any content-safety result or all-model acceptance.

Entropy prediction has mixed quality:

| Dataset | Predictor MAE | Dev-mean constant MAE | Always-zero MAE | MSE reduction vs dev-mean |
|---|---:|---:|---:|---:|
| POPE | 0.05754 | 0.05655 | 0.03026 | 14.52% |
| TextVQA | 0.10184 | 0.14531 | 0.07899 | 42.45% |

Most teacher targets are zero. The predictor improves squared error relative to
the development-mean constant, but loses to the always-zero baseline on MAE in
both datasets and slightly loses to the mean constant on POPE MAE. Retaining
downstream calibration therefore does not prove faithful entropy estimation.
No feature or hyperparameter tuning was performed using these test results.

The CPU study itself took about 5.82 seconds after the historical evidence was
available. This is not full collection time or a measured inference speedup.
The proposed generation count is 11 to 8 per item; fresh matched GPU timing is
still required. Keep this variant experimental rather than replacing the
original measured-entropy acceptance method.
