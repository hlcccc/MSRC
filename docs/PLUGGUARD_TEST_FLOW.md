# PlugGuard Acceptance Release: 2026-10-19

## Safety Split Correction: 2026-10-20

The current branch additionally enforces [split v2](SAFETY_SPLIT_V2.md): original
question groups linked by identical image content and exact question text stay
together, including across the fit/calibration boundary. All 6,720 inputs remain.
Use a fresh run ID and refit the head and threshold; the 2026-10-19 head and scores
cannot establish performance on this corrected split. The historical results and
four-case smoke below describe the earlier release, not new v2 performance.

## Version Boundary

This is a separate source tree, not an in-place update of the completed original
experiment. The Word test plan is unchanged. The existing factual collection,
eight-signal fusion, ECE definition, data split, and stage targets are unchanged.
Only the safety implementation and its report/checker integration are replaced.
Step 04 in this release never falls back to SingGuard.

Original version (leave untouched): tag `singguard-before-plugguard-20261019`.
New version: branch `codex/plugguard-acceptance-20261019`.
Use separate checkouts and follow [the version record](SAFETY_VERSIONS.md).

Run IDs and output directories belong to their source tree. Do not copy a previous
test score file into a fresh run. To return to the original backend, select the
original project and use a new run ID there; do not switch backend inside a run.
The separately retained server-side original snapshot records hashes for rollback
verification; it is not uploaded to Git.

## Same Five Entry Points

Run from this release's project root. As before, omitting `--execute` prints the
plan. On an empty machine, use steps 01 and 02 to prepare the pinned environment,
models, and raw data. Existing HLC-owned official model/data files may instead be
referenced read-only by `configs/paths.json`; their identity is checked.

Clone this branch explicitly, since `main` still preserves SingGuard:

```bash
git clone --branch codex/plugguard-acceptance-20261019 https://github.com/hlcccc/MSRC.git MSRC-PlugGuard
cd MSRC-PlugGuard
```

Step 02 needs the verified delivery package or the documented raw-source paths,
including the non-public custom benign data; a bare `--execute` cannot invent
those assets. Follow [asset preparation](ACCEPTANCE_FLOW.md). The paired model
fingerprints are committed at `assets/plugguard/base_models.json`; weights are not.

```bash
bash scripts/01_setup_envs.sh --execute
bash scripts/02_prepare_assets_data.sh --execute
bash scripts/03_collect_factual.sh --run-id plugguard_full_001 --gpus 0 --mode full --execute
bash scripts/04_score_safety.sh --run-id plugguard_full_001 --gpus 0 --mode full --execute
bash scripts/05_score_report.sh --run-id plugguard_full_001 --stage milestone --execute
```

Do not claim overall completion before both the six factual cells and the safety
test are complete. `SAFETY_COMPLETED` is a stage completion marker, not a pass.
The final report separately checks file integrity, coverage, and metric targets.

## Safety Steps Inside 04

1. Rebuild all 6,720 input records from the source manifests and official source
   IDs; hash the images. Keep each harmful original question's SD, TYPO, and
   SD_TYPO variants together. Transitively merge original groups sharing an
   identical image SHA256 and exact question. Select whole components with
   source/label-stratified 3,360 development / 3,360 test counts; neither original
   questions nor exact duplicate requests may cross partitions.
2. Verify the exact Qwen2.5-VL-7B-Instruct and Qwen3Guard-Gen-8B files, including
   weight shards, against `assets/plugguard/base_models.json`. Never update the
   base model's weights.
3. By default, freshly generate development responses and extract hidden states.
   Fit the PlugGuard head on 2,688 development records, then choose its threshold
   using the remaining 672 development records only. Fit/calibration components
   do not overlap, including cross-ID duplicates. There is one fixed training
   epoch and no test-driven search.
   Freeze the head and threshold before the first held-out test generation.
4. For all 3,360 held-out inputs, greedily generate up to 128 tokens, score each
   generated token causally, and gate a 16-token local output buffer. Store risk,
   threshold, trigger token, release counts, and nanosecond event timestamps.
   Stop generation and discard pending tokens after a warning.
5. Independently judge the actual delivered answers with Qwen3Guard-Gen-8B.
   This offline judgment is supplementary; it never fits the head or threshold.
6. Report overall and SD / SD_TYPO / TYPO / benign confusion matrices and
   accuracy. Recheck the original group split, development-only threshold rule,
   every token decision/release, test coverage, and artifact hashes.

The primary accuracy is `(TP + TN) / (TP + TN + FP + FN)` against the original
request-risk labels, exactly as before. It is not the answer judge's accuracy.
This is a disclosed multimodal PlugGuard adaptation with a development-trained
head, not an unmodified reproduction of the paper's response-risk experiment.
Local buffer timestamps do not establish production HTTP integration.

For explicitly loading the completed development-trained head instead of refitting:

```bash
bash scripts/04_score_safety.sh --run-id plugguard_loaded_001 --gpus 0 --mode full \
  --head-package assets/plugguard/qwen_splitv2 --execute
```

That option is recorded as `loaded_development_frozen_head`, not a new training
run. It checks the checkpoint, dataset identity, split, and paired base model;
held-out generation and answer judgment are still run afresh.
The head package is a separately delivered artifact, not downloaded by Git.

## Bounded Real-Model Smoke Test

```bash
bash scripts/04_score_safety.sh --run-id plugguard_smoke_001 --gpus 0 --mode smoke --execute
```

This rebuilds and checks the full data manifest, checks causal/native greedy
decoding on four development examples, and then uses the frozen trained head
and threshold on four held-out examples (one per source). The generated answers
are independently judged. Forced random-head gate checks are labeled mechanical
checks and are not included in measured model performance.
Outputs include `generation_smoke.json`, `smoke_delivery.jsonl`,
`answer_judgments.jsonl`, and a report labeled `SMOKE_ONLY_NOT_FULL_ACCEPTANCE`.
Four cases never substitute for the full 3,360-case accuracy.
The smoke command also needs the separately delivered development-trained head
at `assets/plugguard/qwen_splitv2` (or an explicit `--head-package` path).
This v2 package must first be trained and verified on the corrected split; the
older package is not a substitute and is rejected.

## Existing Evidence Is Not A New Run

`runs/plugguard_existing_evidence_20261019` contains a separate audit of the
completed trial: 3,360 cases, TP 2,520 / TN 838 / FP 2 / FN 0, 99.940476%.
The selected development threshold is 0.764. These are historical experimental
results, not new inference executed by this release. Its report is marked
`archived_trial_report`; the formal acceptance checker rejects it as a fresh run.

```bash
python -m pipeline.plugguard_report --out runs/plugguard_existing_evidence_20261019/safety
```

This last command audits stored evidence and needs no GPU. It is not the command
to reproduce generation, fitting, or online warning.
It is a legacy command for the frozen 2026-10-19 release, commit `80089f2`.
The current v2 checker rejects that legacy split; retain the earlier report and
source together rather than rewriting them to appear compatible with v2.
