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

InternVL additionally caches official image preprocessing and projected visual
embeddings in a bounded 1 GiB CPU LRU cache. Cache entries are local to one
frozen model/processor instance and include image file identity and timestamps.
No answers, language-model KV states, labels or risk scores are cached. Image
resolution, patch count, prompts, token limits and the eight feature functions
are unchanged. Deterministic requests may be grouped by estimated input length;
their outputs are restored to the original request order. Sampled requests keep
their original order. Per-stage preparation, generation, scoring and padding
counters are written alongside the throughput measurements.

Before profiling or collection, a fail-closed development-only GPU audit compares
uncached, cold-cache and warm-cache full evidence for both datasets, including
all eleven generated texts, available numeric readings and signal values.
Numeric tolerance is absolute 1e-6 / relative 1e-5; texts must match exactly in
this batch-1 audit. Failure stops the controller. This is a targeted cache audit,
not a proof of identical stochastic outputs under different batch sizes.

With paths configured and the validated environment installed:

```bash
python -B -m pipeline.batch_training --run-id batch_train_001 --phase prepare
python -B -m pipeline.batch_training --run-id batch_train_001 --phase optimization_check
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
The HLC controller accepts `--cap-gib 22` to lower the allocator cap when less
memory is available. Its start threshold then becomes 30 GiB, preserving the
same 8 GiB reserve for both factual and safety stages. The default remains
24 GiB; scientific settings and the frozen-source checks are unchanged.
Use `--gpu 1 --cap-gib 19` on the HLC controller to select GPU1 with a 19 GiB
allocator and the same 8 GiB reserve. Selecting GPU1 does not pause the named
older GPU0 job. The pretrial still chooses the fastest batch that fits this cap.
Candidate batches use the same fixed development cohort within each model and
dataset (at least 16 inputs for InternVL and 64 for Qwen, plus the longest
question). InternVL profiling stops at 16 under the shared 24 GiB cap. These
are tested candidates, not a claim of the hardware's absolute maximum batch.

Outputs: `runs/batch_train_001/factual/batch_profile.json` contains measured
batch throughput, memory and factual-stage projections. These are projections,
not full-run completion. Final per-indicator HTML/Markdown files, `report.html`,
`report.md` and `summary.json` are under
`runs/batch_train_001/multilayer/joint/`. Fresh raw evidence and calibrators are
under `factual/`; safety training and token-event evidence stay under `safety/`
and `multilayer/`. Feature extraction preserves a 64 GiB disk reserve and the
original 128 GiB feature cap; model/data files are referenced read-only.
