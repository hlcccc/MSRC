# GitHub-Only Reproduction

The operational six-step procedure is [ACCEPTANCE_FLOW.md](ACCEPTANCE_FLOW.md).
It does not need the authors' server, local delivery package, old predictions,
trained PlugGuard head, fitted calibrators, or thresholds. All learned components
used for acceptance are fitted afresh on the current development partition.
Downloading a publicly pretrained base model is not pretraining that base model
from random initialization; base models stay frozen throughout this protocol.

## Public Inputs

`resources/reproduction/data_sources.json` freezes public locations and commits:

| Input | Source | Scope |
| --- | --- | --- |
| POPE | RUCAIBox/POPE, commit `08d957b917e5a378a2f99d35b6293c536a66298b` | Three COCO settings, 9,000 questions / 500 original images |
| COCO images | Official COCO public S3 bucket, HTTPS path-style URL | Only the 500 images referenced by POPE |
| TextVQA | Official `dl.fbaipublicfiles.com/textvqa` release | 5,000 validation questions / 3,166 original JPEGs |
| MM-SafetyBench | PKU-Alignment/MM-SafetyBench, commit `7da6155621db5733e7f72fafd85da14ba45053d7` | 39 parquets, 13 categories, three image variants, 5,040 requests |
| Benign instructions | tatsu-lab/alpaca, commit `dce01c9b08f87459cf36a430d809084718273017` | Original deterministic selection, no Alpaca output text |

The TextVQA source ZIP is approximately 7.1 GB. Its official MD5 is verified;
only requested validation JPEG bytes are extracted, without recompression. The
frozen SHA256 list subsequently checks all original/prepared input files and
images against the validated experiment, not just their counts. MM-SafetyBench
parquets are checked against the same original files. No hidden test predictions
are consulted during download, input preparation, fitting, or threshold selection.

## Rebuild Benign Samples

The actual original generator from the project has been made portable as
`pipeline/build_benign.py`, not replaced with newly invented negative examples:

1. Download the fixed 52,002-row Alpaca parquet and read only `instruction`.
2. Shuffle with seed `20261010`. Retain instructions starting with an ASCII
   letter, containing 2--7 whitespace-separated words and no digits. Preserve
   the original de-duplication/normalization order and select 1,638 phrases.
3. Add the original 42 fixed benign lure phrases and shuffle with the same seed.
4. Assign the five fixed MM-SafetyBench-style question templates cyclically.
5. Render 1,680 RGB 1024 x 180 PNGs, white background and black centered text,
   with the exact licensed DejaVuSans font and the original adaptive font sizes.
6. Verify the entire ordered instruction manifest and every image SHA256 against
   raw-input identities. Mismatch stops the build; it does not silently keep
   old metrics or accept a different negative benchmark.

The seed above builds raw data. The separate fixed seed `20260920` controls the
acceptance split. Neither seed is selected on the held-out test accuracy. Label
zero denotes the original benign request label, not an answer judge's verdict.

## Models and Training

`resources/reproduction/models.json` contains revisions and file hashes, not
weights. Weights are downloaded from the corresponding public Hugging Face
repositories using those exact revisions and approved file names. Qwen is shared
between its factual role and the safety generator, without updating its weights.
The existing SingGuard asset-preparation compatibility role is retained but never
used as a PlugGuard fallback. Only small approved non-weight configuration/chat
template text is reconstructed from `model_overrides.json` before verification.

Step 03 recollects all six factual evidence files. Step 04 refits each factual
calibrator independently from that run's development evidence. Full step 05
recollects all development features, fits a new risk head, selects the threshold
only on development calibration data, freezes both, and runs fresh held-out
generation with independent answer judgment. No `--head-package` is used.
Pretrained-head loading and smoke tests are maintenance options, not this public
from-scratch acceptance procedure.

Step 06 checks the recorded training provenance with
`scripts/check_fresh_training.sh`, rejecting imported head/threshold packages,
reused development generation and calibrator training IDs from a different split.
Its `FRESH_TRAINING_SCOPE_OK` receipt is not an accuracy or completeness verdict;
the normal scoring report must still pass its independent artifact checks.

## Resource and Interpretation Boundaries

Use the validated Ubuntu 22.04 / A100-80GB-class environment and pinned packages.
Step 05 checks that free disk remains above 180 GiB while collecting features;
reserve at least 350 GiB initially for public checkpoints, originals, duplicate
prepared inputs, and generated artifacts. Additional safety margin may be needed.
GPU occupancy checks wait instead of killing or displacing other users' jobs.

Warning timing is causal per-token scoring with a controlled local output buffer,
not a claim of production HTTP integration or of every harmful answer being
blocked before its first token. Request-risk accuracy is distinct from answer
safety. Each new test reports its own confusion matrix and answer judgments.
Different hardware/numerical kernels need not produce bit-identical retrained
heads; neither historical accuracy nor a completed program substitutes for the
new run's `ACCEPTANCE_PASS` decision.
