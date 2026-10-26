# Multi-layer Safety Delivery

## Isolation

This branch starts from `62cc46298e58dc5ee1fd52a92e7eec836a4a148d`
(`acceptance-shared-wait-20261024`). Existing pipeline modules, dependencies,
four variant scripts, upstream PlugGuard code and original acceptance documents
are unchanged. New entry: `scripts/run_safety_multilayer.sh`.

The original ten experiment/core files and original inference adapter are stored
byte-identically under `extensions/multilayer/frozen/`, with a SHA256 inventory
in `resources/multilayer/source_manifest.json`. Their historical command-line
entry points are archival, not the public entry. The new portable runner overrides
only setup, input transport, paths, resource hooks and source receipts. Decode,
layer capture, projection fitting, threshold selection and controller equations
remain the audited implementations. No cloud credentials, SSH access, copied
environment or author-specific input bundle is required by the public procedure.

## Parameters

- Original PlugGuard head/threshold: same fixed, hash-pinned public Release used
  by the existing supplied-parameter variant; see `resources/reproduction/fitted.json`.
- Soft controller: development-selected `cloud_prefix_002` state, explicitly
  supplied in `resources/multilayer/controller.json`, not freshly selected here.
- Three projection directions: fresh onsite fit, using judged fit-role answer
  prefixes at positions 8, 16, 32, 64, 96, 128 (capped to observed length).
- Added semantic/temporal thresholds: fresh calibration-only selection with
  a 5% additional safe-response false-alarm budget. Unknown reviews are not safe.
- Disagreement hold threshold: 95th percentile of fit-side safe projection
  standard deviations. Neither the test set nor the best test arm selects it.
- Factual calibrators: fresh onsite fitting in this procedure. Downloading the
  existing Release does not cause its factual calibrators to be used.

This is a mixed inherited-head/fresh-added-signals procedure, not the original
all-components-fresh `safety_no_fit` procedure. Request-risk accuracy does not
measure answer safety or exact harmful-token lead time.

## Verification Scope

The unchanged experimental core completed an independent RTX 3090 run and an
independent HLC A100 run. HLC testing used 3,360 held-out requests and 16,800
answer reviews. Original request-label accuracy: 99.970238%; predeclared
`temporal_hold` accuracy: 97.619048%. Both exceed 75%; the enhancement did not
improve overall accuracy. Historical evidence is descriptive, never loaded by
the new scoring commands to manufacture current results.

This delivery adds a portable preparation/runner/report transport. Publication,
CPU tests, a complete original GPU experiment, and a new blank-machine end-to-end
GPU run are distinct checks. The new portable entry is not advertised as a newly
completed blank-machine GPU acceptance. See `resources/multilayer/verification.json`
for the checks actually performed for this release. Every future run must produce
its own material receipts, frozen source/parameter hashes and per-request outcomes.

Publication checks passed 121 CPU tests, four shell syntax checks, a no-execution
workflow plan, exact public-Release parameter verification, and all 6,720 raw
input/image/role checks plus portable setup. These reused an already validated
environment and raw assets read-only. A new Release download timed out; the
verified existing archive was used instead. No GPU inference was started for
publication, and no new blank-machine installation/GPU completion is claimed.

The fixed tag is `acceptance-multilayer-20261026-v2`. New metadata and documents
use LF, with a checkout-byte regression test. The first publication tag is kept;
its Windows-archive metadata digest differed from Linux checkout bytes. Frozen
algorithm bytes and original GPU experiments were not changed by this correction.

## Contents

Git contains source, focused tests, fixed dependency recipe, model/data identities,
font/license, inherited-controller state, six-step Markdown/Word documents and
small verification receipts. Large base weights, datasets/images, environments,
development feature arrays, raw answers and full test predictions are not uploaded.
Project-fitted base-head weights remain in the existing public parameter Release.
There is no overwrite of that Release, original branches, original GPU results,
or other users' jobs. Existing upstream and dataset/model licenses still apply.

## Checks and Recovery

```bash
bash scripts/test_multilayer.sh
bash scripts/run_safety_multilayer.sh --run-id plan_001 --gpus 0
```

The second command prints a plan only. Partial material preparation is preserved
for diagnosis; do not delete or silently overwrite it. Use a new run ID after
repair. Once prepared, `--phase check` verifies every material/image hash.
Completed phase journals may resume with the same frozen source bytes via
`scripts/04_score_multilayer.sh --phase development|label|fit|test|judge`.
Do not use the full one-step entry to overwrite an existing run.
Default GPU use is exclusive; shared operation needs the original explicit
run-local sharing policy and authorization described in `GPU_SHARING.md`.
