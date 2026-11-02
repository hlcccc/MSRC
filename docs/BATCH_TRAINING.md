# Isolated InternVL training-throughput experiment

This branch does not replace the original three-model delivery. It evaluates
only InternVL3-8B-hf on all 9,000 POPE and 5,000 TextVQA inputs. Qwen supplies
fresh primary answers for the cross-model feature, without collecting its
unused uncertainty evidence. The third indicator retains the full original
PlugGuard plus multi-layer onsite-training, testing and answer-review procedure.

Primary answers, four views, three resamples and three verification prompts are
batched. Only primary token distributions are retained: the existing eight
feature functions do not read auxiliary-generation distribution scores. OCR is
still obtained from the input images. Per-sequence EOS masking excludes other
batch members' padding from confidence and entropy. Sampled outputs use a
batch-dependent RNG stream and cannot be described as byte-identical historical
outputs. All calibrators are freshly fitted on development inputs only.

The baseline detector's fit, gradient accumulation, calibration split, causal
online decoder and five-arm multi-layer test are not changed. Independent
teacher requests are batched with exact question/answer identity and prompt
hashes. Out-of-memory retries may reduce teacher batch size, never input length.

The original six-cell acceptance checker remains strict by default. This
experiment requests an explicit two-cell audit and reports SCOPED_PASS or
SCOPED_FAIL, not original three-model acceptance. No test result selects a
batch size; throughput profiling uses development inputs only.
The batch grid doubles from 1 through 64 and stops at the first out-of-memory
candidate. Each candidate has enough development inputs to exercise its actual
batch size; the fastest successful candidate is chosen. Primary predictions
are saved for auditing batch-versus-single-request differences.

With paths configured and the validated environment installed:

```bash
python -B -m pipeline.batch_training --run-id batch_train_001 --phase prepare
python -B -m pipeline.batch_training --run-id batch_train_001 --phase profile
python -B -m pipeline.batch_training --run-id batch_train_001 --phase collect
python -B -m pipeline.batch_training --run-id batch_train_001 --phase evaluate
MSRC_JUDGE_BATCH_SIZE=4 python -B -m pipeline.plugguard --out runs/batch_train_001/safety --refit
MSRC_JUDGE_BATCH_SIZE=4 python -B -m extensions.multilayer.batched --run-id batch_train_001 --phase prepare
MSRC_JUDGE_BATCH_SIZE=4 python -B -m extensions.multilayer.batched --run-id batch_train_001 --phase check
MSRC_JUDGE_BATCH_SIZE=4 python -B -m extensions.multilayer.batched --run-id batch_train_001 --phase all
MSRC_JUDGE_BATCH_SIZE=4 python -B -m pipeline.batch_training --run-id batch_train_001 --phase report
```

The shared A100 experiment caps its allocator at 24 GiB, reserves 8 GiB for
other jobs, and yields half its measured work time. It does not terminate
another user's process. This is not a measurement on an exclusive RTX 5090.

Outputs: `runs/batch_train_001/factual/batch_profile.json` contains measured
batch throughput, memory and factual-stage projections. These are projections,
not full-run completion. Final per-indicator HTML/Markdown files, `report.html`,
`report.md` and `summary.json` are under
`runs/batch_train_001/multilayer/joint/`. Fresh raw evidence and calibrators are
under `factual/`; safety training and token-event evidence stay under `safety/`
and `multilayer/`. Feature extraction preserves a 64 GiB disk reserve and the
original 128 GiB feature cap; model/data files are referenced read-only.
