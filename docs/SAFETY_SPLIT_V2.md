# Safety Split V2

## What Changed

The earlier release grouped the three image variants by official original-question
ID. That fixed variant-wise leakage but did not link different official IDs with
identical image bytes and exactly the same question text. The archived full input
manifest contains six such duplicate-request groups, four crossing development
and test in the old partition.

`pipeline/safety_split.py` now unions those original-question groups transitively
using image SHA256 and exact question text before any model fitting or scoring.
It preserves all rows and their original source IDs; `split_group` records the
connected component, and `image_sha256` records the input fingerprint. Same-image
requests with different text are not automatically classified as identical.

The prespecified seed remains 20260920. Whole components are selected using the
existing seeded original-group order and an exact balanced subset, separately by
label. The calibration selection likewise preserves the former hash order when
there are no aliases. No answers, predictions, risk scores or observed accuracy
are used to choose partitions. If exact balance cannot be achieved without
splitting a component, preparation fails rather than relaxing isolation.

Both development/test and fit/calibration must respect the connected components.
The report and acceptance check independently reconstruct them from image
fingerprints; a stored `group_overlap=0` or fabricated `split_group` is not enough.
The split implementation fingerprint is included in each new run's protocol.

## Full Manifest Validation

The verified archive `d06a73ced0064ba808e1855d87b1cb1506dc14d26ebb120c9f0981e9a2de489d`
was used only to validate all 6,720 input identities and their frozen image hashes.
This local CPU validation did not rehash the remote image bytes or run a model.

| Check | Result |
|---|---:|
| Rows retained | 6,720 |
| Original question groups | 3,360 |
| Connected split groups | 3,354 |
| Multi-origin connected components | 6 |
| Exact request groups crossing development/test, before / after | 4 / 0 |
| Original question groups crossing development/test | 0 |
| Components crossing fit/calibration/test | 0 |
| Fit / calibration / test rows | 2,688 / 672 / 3,360 |
| Each SD / SD_TYPO / TYPO / benign stratum per half | 840 |
| Rows changing development/test side | 24 |

This table is a split-integrity result, not an accuracy result. No performance
number or threshold was updated based on it. The main accuracy formula, official
base weights, PlugGuard head architecture, training budget and six factual cells
are unchanged.

## Reproduce The Split Check

Use matching `inputs.jsonl` and `input_hashes.json` from a verified source package.
Choose a new output directory inside this project:

```bash
python scripts/validate_safety_split.py \
  --inputs /path/to/verified/inputs.jsonl \
  --image-hashes /path/to/verified/input_hashes.json \
  --out runs/split_v2_validation_001
```

This creates a new input partition, roles, fingerprints and `split_validation.json`.
It never copies old scores or heads and prints `SPLIT_CHECK_OK_NOT_MODEL_EVALUATION`.
The production preparation step hashes the live input image files again and uses
the same component and partition implementation.

## Performance Must Be Rerun

Run step 04 in full mode with a new run ID and without `--head-package`. It will
prepare the corrected partition, generate development features, refit the head,
select and freeze a development-only threshold, then generate and judge all
held-out responses. Do not reuse the previous 0.764 threshold or trained head.
Old head packages are rejected by split-version, input-identity and role checks.
A new v2 head package is required before the loaded-head or smoke modes can be
used on this partition.

```bash
bash scripts/04_score_safety.sh --run-id plugguard_splitv2_full_001 \
  --gpus 0 --mode full --execute
```

Do not switch code underneath an active job or overwrite the old run. Historical
99.940476% (PlugGuard) and 87.053571% (SingGuard) remain historical measurements,
not corrected-split results. The original SingGuard rollback tag and source tree
remain untouched; the new branch's shared safety preparation uses split v2.
