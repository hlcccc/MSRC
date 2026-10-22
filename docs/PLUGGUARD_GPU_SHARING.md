# Explicit PlugGuard GPU Sharing

The default remains exclusive execution: a busy GPU is rejected or waited for.
Sharing requires separate explicit user permission and a run-local JSON policy;
it is not a fallback after a failed preflight. Only step 04 supports this bounded
mode. Do not use this policy to run uncapped factual collectors on a busy card.

For an explicitly authorized GPU1 run, create `configs/gpu1_shared.local.json`
using the actual device UUID and authorized process owners:

```json
{
  "version": 1,
  "gpu_index": 1,
  "gpu_uuid": "GPU-replace-with-the-actual-authorized-uuid",
  "allowed_other_users": ["HLC"],
  "allocator_cap_gib": 24,
  "min_start_free_mib": 32768,
  "min_run_free_mib": 8192,
  "target_duty_when_shared": 0.5,
  "check_interval_seconds": 10,
  "authorization": "Record the explicit user authorization for this run"
}
```

```bash
bash scripts/04_score_safety.sh --run-id shared_splitv2_full_001 --gpus 1 \
  --mode full --shared-gpu-policy configs/gpu1_shared.local.json --execute
```

Preflight checks the exact GPU index/UUID and all existing process owners.
There must be 32 GiB free before loading, allowing a 24 GiB allocator limit and
an 8 GiB reserve. During work, remaining allocation headroom is also reserved;
insufficient memory or an unexpected owner stops only this worker, never another
job. Work blocks synchronize and voluntarily yield equal measured wall time while
another authorized job is present. The target duty is 50%, not a hardware compute
partition or a zero-interference guarantee. If no other job is present, yielding
is unnecessary; checks continue at most ten seconds apart between work blocks.

The policy is copied into the run and its SHA256 is frozen. Changing permission
during execution is rejected. Protocol and runtime statistics disclose sharing,
memory limits, measured work, and voluntary idle time; the final safety checker
also verifies this disclosure. Shared execution can take longer and does not
establish isolated latency. It does not change input grouping, labels, head
architecture, seeds, training epochs, generation budget, calibration candidates,
threshold selection, accuracy formula, or acceptance thresholds.

Use a new code snapshot and run ID before GPU inference. Never replace the code
under a previously started experiment. Retain any preparatory/failed attempts and
the original SingGuard rollback unchanged.
