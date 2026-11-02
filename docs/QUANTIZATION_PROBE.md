# Language-only 4-bit probe

This is an optional development-only experiment on a separate branch. The
original delivery and active BF16 run are not replaced. NF4 stores selected
language-layer weights in four bits and computes in BF16. It is not a claim
of native INT4 matrix multiplication or of parity with AWQ/GPTQ serving speeds.

Unchanged: official base checkpoints, official image processing, prompt set,
128-token generation limit, four views, three resamples, three probes and all
eight candidate signal definitions. The vision encoder, projector and output
head are excluded from quantization. An actual module inventory checks this.

The probe selects development inputs without consulting answers or labels.
It produces independent fresh Qwen donor answers, tests complete InternVL
eleven-generation evidence collection at several batch sizes, and separately
times 100 primary answers per dataset. Full evidence includes per-item readings
and unavailable flags. Candidate count is not fitted/active signal count.

```bash
python -B -m pipeline.quantization_probe \
  --run-id nf4_probe --precision bnb_nf4 \
  --inputs /absolute/path/to/prepared/factual/inputs --cap-gib 8
```

Outputs are in `runs/nf4_probe/bnb_nf4/`: `probe.json` and
`pope_full_batchN.json` / `textvqa_full_batchN.json`. Timings include voluntary
sharing delays, reported separately. Warmup and load costs are not primary
answer throughput. This is a shared A100 measurement, not an exclusive RTX5090
measurement. OOM is bounded by the stated experiment cap, not whole-card capacity.

Install the pinned `bitsandbytes==0.48.2` wheel into an isolated dependency
directory; leave the validated environment unchanged. The HLC launcher waits
for an eight-GiB free reserve and never signals another process.

Triton initialization also needs Python 3.10 development headers. If the server
lacks them, extract the matching Python-minor-version Ubuntu development package
under `deps/python_headers` and use its include directories through `CPATH`.
No system package installation or shared-environment edit is needed.

No calibrator or safety threshold is loaded or reused by this probe. A production
switch requires a fresh development-side calibrator fit and complete held-out
evaluation. The content-safety pipeline is not changed by this experiment.
