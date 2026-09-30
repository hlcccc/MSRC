# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — unreleased

First working version. The framework and its accounting are complete; the
evaluation on public benchmarks is the next step, and no results are claimed here
until they have been produced.

### Added

- **Signal layer** (`msrc/signals.py`) — nine independent readings, six external
  and three internal, each with its provenance recorded:

  | signal | kind | borrowed from |
  |---|---|---|
  | `self_consistency` | external | SelfCheckGPT |
  | `resample_consistency` | external | Semantic Entropy |
  | `typed_verification` | external | typed evidence probing |
  | `cross_model_agreement` | external | cross-model / judge agreement |
  | `grounding_check` | external | retrieval- and rule-grounded checking |
  | `policy_probe` | external | policy self-report |
  | `sequence_confidence` | internal | LLM-Check, SAPLMA |
  | `output_entropy` | internal | distributional uncertainty |
  | `visual_attention` | internal | attention-based hallucination work |

  Eight apply to either risk family: three internal and five external.

- **Two risk families** — `factual` (does the answer contradict the image) and
  `safety` (does the answer violate a content policy). They share the signals, the
  calibrator and the metrics; what differs is the label, the evidence, and which
  family-specific signal applies. The safety family expects a policy taxonomy from
  the deployment rather than shipping one, because what counts as unsafe is a
  policy decision and a hard-coded list would be wrong outside the place it was
  written.

- **Decision layer** (`msrc/model.py`, `msrc/pipeline.py`) — an L2-regularised
  logistic risk map solved by damped Newton, NumPy only; a guarded fusion head;
  JSON scorer export with provenance. Two invariants are enforced rather than
  documented:

  - every signal's `risk` is oriented so higher always means riskier, so the
    calibrator never has to discover a sign;
  - a signal a deployment cannot supply reports `available=False` and is named in
    the payload. It is never filled with a neutral value, because a constant
    column trains a calibrator that looks healthy and encodes nothing.

- **Conformal selective prediction** (`msrc/conformal.py`) — split-conformal
  p-values with Benjamini-Hochberg and Benjamini-Yekutieli multiplicity control,
  for the second decision a deployment has to make: which items to act on and
  which to send to a human. BY is the default because scores from one model on
  items sharing an image are not independent. `VALIDATED` is `False` and travels in
  every result — the procedure is implemented from the standard definition and
  unit tested against textbook cases, but it has not been checked on this project's
  data, and an unmeasured guarantee should not be reported as one. The report also
  keeps `realized_fdp` separate from the guarantee, since the guarantee is on the
  expectation over draws and the realized value is one draw.

- **CLI** (`msrc/cli.py`) — `signals`, `demo`, `explain`, `fit`, `score`,
  `evaluate`, `version`. Declared in `pyproject.toml`; a test asserts the entry
  point resolves to a callable rather than only that the string is present.

- **Safety label rule** (`msrc/judge.py`) — refusal detection in English and
  Chinese, with the awkward middle reported rather than hidden: a response that
  refuses and then complies is marked uncertain, and `uncertain_rate()` returns
  the fraction so a safety result can quote the size of the judgement call.

- **Benchmark adapters** (`evaluation/datasets/`) — MM-SafetyBench into the record
  shape the collector reads, plus `stratified_sample` for a category-balanced set.
  `pyarrow` is a declared extra because pandas reads parquet only through an engine
  it does not install.

- **Two model backends** (`msrc/providers/hf_vision.py`) — `HFLLaVAProvider` and
  `HFQwenVLProvider` over one shared implementation, selected with `--provider`.
  Both produce the same signals; the second exists because the safety family's label
  is "did the model refuse", which needs a model that does both, and LLaVA-1.5-13B
  refuses none of MM-SafetyBench's TYPO set. The shared implementation handles both
  image-token layouts (one placeholder expanded inside the model, or one per patch
  spliced in by the processor).

- **Evaluation layer** (`evaluation/metrics.py`) — AUROC with correct tie
  handling, Brier, ECE with a documented binning dependence, threshold metrics,
  and `summarise()` which returns all of them together so a favourable subset
  cannot be quoted by accident.

- **`scripts/threshold_analysis.py`** — whether an accuracy shortfall is a badly
  chosen operating point or a score that cannot separate the classes. Sweeps the
  threshold on dev and prints the best accuracy any threshold reaches on test as a
  ceiling, labelled as a ceiling rather than a result. It shares
  `evaluation/split.py` with `evaluate.py` so the two cannot draw different splits.

- **`examples/run_stub.py`** — the whole chain on a CPU with no model and no
  weights, run twice: once as a text-only platform and once with generation-time
  internals supplied, to show which channels a deployment can actually feed.

- CI on Python 3.9–3.12, plus a job that asserts the decision layer still works
  with NumPy alone and a job that checks the declared signal inventory against the
  code rather than against the prose.

### Fixed

- **The safety family could not be run end to end.** The MM-SafetyBench adapter
  emits records with no accepted answers, by design — there is nothing to compare a
  response against, only a harmful request — and the collector had no branch that
  would label such a record. It demanded an answer or accepted answers and exited,
  so the two other files that documented the safety workflow described something
  that did not work. The collector now labels safety items by refusal after
  generating them, stores the verdict beside the label so the judgement can be
  audited, and reports the rate at which the rule was unsure of itself.

- **`scripts/check_safety_labels.py` did not exist**, though `msrc.judge` named it
  as the way to measure the disagreement rate that module's own docstring says must
  accompany a safety result. It now does: a labelling sheet can be generated, and
  checking one reports agreement, Cohen's kappa, the confusion counts and every
  disagreement with its response text.

- **`evaluate.py` described every result as a factual one.** Pointed at safety
  evidence it printed a closing paragraph stating that the number was not a safety
  judgement. The family now comes from the evidence, the indicator heading and
  caveats follow it, and a file that mixes two families is refused rather than
  averaged: 1 does not mean the same kind of thing in the two cases.

- **`ModelProvider` did not describe what the collector calls.** It declares
  `generate` and `ocr`, but the collector also called `load()` and read
  `want_attention`; a provider written against the documented protocol would have
  failed. `load` is now documented as optional and skipped when absent, and
  `calls` is declared as the counter it already relied on.

- **`evaluation/datasets/` was never in the repository.** The `.gitignore` entry
  `datasets/` was unanchored, so it matched `evaluation/datasets/` as well as the
  downloaded-data directory it was written for. The MM-SafetyBench adapter was
  consequently absent from every commit -- including the one whose message says it
  adds it -- while `git status` reported a clean tree, and the test that imports the
  adapter pointed at a module the repository did not contain. The bulk directories
  are now anchored to the repository root, which closes the same trap for a future
  `msrc/models/` or `msrc/cache/`.

- **The NumPy-only CI job had never passed**, on any commit, since the first one.
  It fitted the pipeline on invented signal names and then scored it; the pipeline
  refuses a signal set that differs from the one it was fitted on -- correctly, but
  at scoring time rather than at the mistake. The check is now
  `examples/numpy_only.py`, a file that can be run by hand and is covered by the
  suite, and `msrc.signals.signal_names_for()` supplies the column list for the
  documented `fit_from_signals` path instead of leaving each caller to write it out.

- **`realized_fdp` was computed with the polarity inverted.** The module documents
  the null hypothesis as "this item is not risky", so a rejection means "this looks
  riskier than the safe calibration items did" and a *false* discovery is a selected
  item that is in fact **correct**. The code counted `label == 1` — the genuinely
  risky items — so a selection that flagged every risky item and nothing else
  reported an FDP of 1.0, and one that flagged only safe items reported 0.0. The
  metric was anti-correlated with the thing it exists to measure, and a test pinned
  the wrong polarity. `correct_retention` is renamed `correct_items_flagged`, which
  is what it computes.

- **`msrc.conformal` is now measured rather than disclaimed.** `VALIDATED` was
  `False` with a note that the procedure had never been checked against this
  project's data. `scripts/validate_conformal.py` does that check over repeated
  three-way splits; with `BH`, `E[FDP]` stayed at or under `alpha` at every level on
  both risk families, and the flag is `True` on that basis. The same measurement
  found that `BY` rejects nothing at all at this sample size, and that the realized
  FDP conditional on the procedure firing is several times `alpha` even though the
  expectation is controlled. Both are documented, because the second is what a
  deployment actually experiences.

### Notes

- The `safety` family's detection quality depends entirely on the policy taxonomy
  and labels supplied to it. This release provides the mechanism and the
  measurement, not a validated safety classifier.
- ECE is gameable: a constant predictor at the base rate scores exactly 0. The
  reporting helper prints Brier and AUROC beside it, and a test pins the property.
