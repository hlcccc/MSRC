# Explicit Cooperative GPU Sharing

Exclusive use remains the default. Sharing is an operational option requiring
recorded user permission, one exact GPU index/UUID, permitted process owners,
memory reserves, and cooperative idle time. It is not hardware isolation and
does not guarantee zero interference or comparable latency.

Factual collection supports a version-2 policy with `workload: "factual"`,
`allocator_cap_gib: 36`, `min_start_free_mib` at least 49152, and
`min_run_free_mib` at least 12288. The other required fields are `gpu_index`,
`gpu_uuid`, `allowed_other_users`, `target_duty_when_shared` (at most 0.5),
`check_interval_seconds` (at most 10), and nonempty `authorization`.
The policy must be owned by the executing user and stored inside the project.

```bash
bash scripts/03_collect_factual.sh --run-id thirdparty_001 --gpus 0 \
  --mode full --phase collect --shared-gpu-policy configs/factual_shared_gpu0.json --execute
bash scripts/03_collect_factual.sh --run-id thirdparty_001 --gpus 0 \
  --mode full --phase fit --execute
```

The collector applies its allocator cap before loading weights. Between complete
samples it checks GPU ownership and memory headroom and yields according to the
recorded duty target. Each evidence file gets a `.resources.json` sidecar; the
protocol freezes the policy and resource-controller hashes. The acceptance
checker verifies full sample coverage and the actual recorded memory ceiling.
Insufficient resources stop or delay our work; they never trigger a smaller
dataset, precision change, shorter answer, fewer resamples, or test-set retuning.

PlugGuard retains its existing version-1 policy and 24 GiB allocator ceiling,
32768 MiB minimum initial free memory and 8192 MiB minimum running reserve:

```bash
bash scripts/04_score_safety.sh --run-id thirdparty_001 --gpus 0 \
  --mode full --refit --shared-gpu-policy configs/plugguard_shared_gpu0.json --execute
```

Do not interchange these two policy files. Public foundation weights, inputs,
splits, prompts, generation budgets, fitting rules, and metric formulas remain
unchanged. Sharing changes scheduling and memory limits, not the tested method.
Record the source revision and any operational transition in the execution log;
do not overwrite an earlier run's protocol or restart completed model experiments.

## Waiting Instead of Exiting When Other Users Arrive

An explicit version-3 policy opts into sharing with any process owner. Set
`version: 3`, `allowed_other_users: ["*"]`, and `workload` to `factual` or
`plugguard`. All remaining fields and that workload's memory bounds remain as
above. Record the user's authorization; versions 1 and 2 still permit only their
named owners and do not accept the wildcard.

Version 3 observes other process owners but never signals or changes their jobs.
New occupants do not stop collection. If actual free memory drops below the
reserve, our process waits between complete work blocks and rechecks the frozen
permission. It resumes when memory permits without changing the RNG state,
sample sequence, or inference settings. A changed permission or device identity
still fails loudly; no policy can guarantee that an unrelated allocation cannot
race a GPU operation.

This in-process wait is not a claim of restart equivalence after a process dies.
If an old attempt lacks an RNG checkpoint, preserve its evidence and use a
separate, explicitly recorded full attempt rather than silently appending with
a reset random stream. Keep the original failure and never select attempts by
their test performance.
