# Single-GPU safety evaluation scheduling trial

This is an isolated development experiment, not a replacement acceptance flow.
It leaves the warning model, frozen thresholds, model precision, source split,
full test scope and original formal runs unchanged. No cloud job is modified.

The original multi-layer implementation already shares one shadow generation
across five control arms and caches exact duplicate question/answer judgments.
Those existing optimizations are not counted as new gains here.

The candidates are one-time unpadded tokenization, length-ordered judge batches
and a larger length-ordered batch. All use the same official Qwen3Guard-Gen-8B
checkpoint, SDPA, BF16, greedy generation, 128-token maximum and untruncated
inputs. Output records are restored to the original prompt order.

Fixed fit/calibration requests are selected without inspecting test labels.
Previously collected development answers and their text prefixes are only
fixed inputs to fresh judge inference. They are not represented as new VLM
answers, original token-trace checkpoints or new warnings.

Fresh serial inference provides the raw-output and parsed-label reference.
Each scheduled mode is measured twice in opposite order. A candidate requires
exact raw strings, parsed labels and RNG equality, plus at least 5% speedup in
both paired rounds, before it is marked eligible for further validation. No
candidate is activated automatically. OOM is recorded rather than silently
changing batch composition or discarding inputs.

```bash
python -B -u -m pipeline.safety_judge_speed \
  --source /path/to/original/run/safety --model /path/to/official/judge \
  --run-id safety_judge_speed_001
```

Reports, full fresh judge outputs, paired timings and cohort provenance are in
`runs/safety_judge_speed_001/judge_speed/report.md` and `report.json`.

Timings include tokenization, padding, transfer, generation and shared-GPU
yielding. Model loading is reported separately. This does not remeasure the
primary warning accuracy or establish a full-pipeline speedup or acceptance
verdict. A100 shared-card timings must not be relabeled as RTX 5090 timings.
