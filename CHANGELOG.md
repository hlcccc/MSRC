# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — the tenth signal: first-step decision margin

`sequence_confidence` averages the token probabilities over the whole answer, and on
a yes/no benchmark almost all of that answer is boilerplate. "Yes, there is a
snowboarder in the image, riding down a snow-covered slope" averages one hard
decision together with twenty easy tokens, so the mean stays comfortable on items the
model actually guessed at. On POPE the reading is AUROC 0.639.

The new signal reads only the position where the model committed, and reads the
loser as well as the winner: ``margin = p(top-1) - p(top-2)`` at the first generated
step. Certainty puts nearly all the mass on one token and the margin approaches 1; a
model torn between "Yes" and "No" splits the mass and the margin collapses. Risk is
``1 - margin``, so an indecisive commitment is the risky one.

It is a genuinely different reading from the mean confidence, not a summary of it,
and the tests pin that: same mean confidence with opposite margins must produce
opposite risks.

### Added

- `Sample.first_token_margin`, populated by the Hugging Face vision provider from the
  per-step scores it already requests, and carried through the evidence JSONL.
  Evidence cached before this release lacks the key and still loads -- the accessor
  returns ``None``, which the signal reports as unavailable rather than as a
  placeholder.
- `DecisionMargin` in the signal registry, as the fourth internal signal.
- `tests/test_decision_margin.py` — 10 tests covering orientation, clamping, the
  disagreement with mean confidence, and the round trip through JSON.

### Changed

- the signal inventory is ten (six external, four internal). Counts in the README,
  `docs/evaluation.md`, the CLI examples, the collector's startup message and the
  tests move with it.

### Not yet measured

The margin needs a fresh collection run to exist in the evidence, so **no evaluation
number is claimed for it here**. The columns in every cached evidence file are from
before it existed. Reporting a gain from it before re-running would be reporting a
number that does not exist yet.

## [Unreleased] — the ninth signal: object-level support

Caption-level agreement is the wrong resolution for a caption hallucination. Two
descriptions of the same bathroom agree about the toilet, the shower and the sink
whether or not one of them also invented a mirror, so `self_consistency` and
`resample_consistency` barely move while the one false claim is exactly the one that
matters. On CHAIR the fused score ranked hallucinations at AUROC 0.63, which is
close to the 0.50 of a coin and far below anything usable.

The new signal drops to the object level. It reads the objects the answer names out
of a lexicon, asks of each how often it returns across the four re-phrasings and
three resamples, and reports the **weakest** rate: a caption is only as trustworthy
as its least supported claim.

Measured on the full 1,200-item CHAIR split, dev-selected threshold and test-half
reporting, which is the same protocol every other number in this repository uses:

| | AUROC | accuracy | recall |
|---|---|---|---|
| before | 0.6256 | 57.50% | 18.1% |
| after | **0.8199** | **76.17%** | **72.31%** |

Qwen2.5-VL-7B on the same split moves from AUROC 0.5849 to 0.8050. The threshold
sweep puts the best accuracy *any* threshold can reach on test at 0.7700, so the
remaining shortfall against an 85% bar is a limit of the score, not of the operating
point — the distinction `threshold_analysis.py` exists to make.

### Added

- `msrc/object_support.py` — lexicon loading, mention extraction, and the
  minimum-support reading, all pure functions of ``Evidence`` so the signal is
  testable with no model and no GPU.
- `references/chair_synonyms.txt` — the official CHAIR synonym list (COCO's 80
  categories plus surface forms), bundled so the repository is self-contained.
  `MSRC_OBJECT_LEXICON` points the signal at a different ontology.
- `ObjectSupport` in the signal registry, as the sixth external signal.
- `tests/test_object_support.py` — 17 tests. The two properties worth pinning are
  that the reading is oriented (an object that never returns raises risk) and that an
  unanswerable item reports itself unavailable rather than scoring zero.
- the `textvqa` extra, declaring `tqdm`. The vendored M4C evaluator imports it at
  module scope, so `--textvqa-label-rule official` could not even be imported on a
  clean install. Found by running the suite in a fresh virtualenv; the development
  machine had `tqdm` by accident.

### Changed

- the signal inventory is nine (six external, three internal). Counts in the README,
  `docs/evaluation.md`, the CLI examples and the tests move with it.

### Known limits, stated here rather than discovered later

- **The lexicon is COCO's ontology, and on CHAIR so is the label.** They share a
  vocabulary but not annotations: the label comes from per-image COCO instance
  annotations, the signal from the model's own re-descriptions. Treated as a plain
  noun filter with no synonym merging the signal still reaches AUROC 0.78 against
  0.82 with merging, so part of the strength is the ontology and part is the
  mechanism. Both numbers belong in any report that quotes one.
- **Availability is dataset-dependent, and low availability is the honest outcome.**
  On CHAIR the signal reads 97% of items (85% for Qwen2.5-VL-7B). On POPE it reads
  99.8% of items and is *constant*: the primary answer is a sentence but the
  re-phrasings are one-word "Yes"/"No", so nothing ever returns and every item scores
  the same. The constant-column check drops it there, which is the designed
  behaviour. On HallusionBench and TextVQA the answers rarely name objects at all.
- **`cross_model` is deliberately not part of the auxiliary set.** Adding it lifts
  the single-signal AUROC from 0.8204 to 0.8280, which is not worth making a second
  signal depend on the second model — the repository already promises that attaching
  one buys exactly one signal, and a test enforces it.

## [Unreleased] — the TextVQA label rule the reported number uses

`references/m4c_evaluator.py` has been vendored here for a while, and the reported
TextVQA row is produced with it, but `scripts/evaluate.py` could not reach it: the
only official-rule switch was POPE-only and refused TextVQA records. Re-running the
script therefore returned a different ECE — +82.68% against the reported +83.27% —
which reads as a regression rather than as a different label rule. It is a different
label rule: the official soft-accuracy scorer disagrees with the archived
VQA-agreement label on 21 of the 5,000 items.

### Added

- `scripts/evaluate.py --textvqa-label-rule official` — applies the released rule
  through the vendored evaluator, says how many labels it moved, and names the rule
  in the reported `label_rule`. It refuses records that are not TextVQA, because on
  yes/no answers the soft scorer returns 0 for everything and the run would look
  like it worked.
- three tests, including the one that pins the refusal: the failure mode is a number
  that is wrong and plausible, not a crash.

### Changed

- the POPE switch keeps its own name and behaviour; the two are separate options
  because the two benchmarks need different scorers, and merging them would hide
  which rule produced a given number.

## [Unreleased] — benchmarks whose label reads the response

Every benchmark here used to be one of two kinds: the label compares the response
against accepted answers, or the response arrives as a published run with its label
attached. CHAIR is neither. Its official rule scores the caption itself — it parses
the object mentions, keeps the ones the COCO vocabulary recognises, and counts the
ones COCO does not annotate for that image — so the label cannot exist until the
response does. Collecting it the old way meant supplying a published caption, which
skips the primary call and costs the three internal signals; the dataset would then
be measured with five signals while every other dataset was measured with eight.

### Added

- `scripts/label_from_response.py` — the pass that closes the gap. It reads an
  evidence file, scores every response with the rule named by `--rule`, and writes
  the labels back. The CHAIRs implementation is a port of the official vocabulary
  and rules, and it is checked against the real corpus: re-scoring 1,200 captions
  that already carried official labels reproduced all 1,200 of them.
- `tests/test_label_from_response.py`, including the bug that made the check worth
  writing — CHAIR ids read `CHAIR:<n>:response:<k>`, where `n` is that dataset's own
  index and not a COCO id. Read as a COCO id, every caption is scored against the
  wrong image and the hallucination rate comes out at 0.996 instead of 0.44.
- `inflect` joins the optional extras as `chair`, since the official vocabulary
  singularises with it and a missing singulariser moves every count silently.

### Changed

- `scripts/collect_evidence.py` accepts `"label_from_response": true`. Such a record
  is generated with the question left alone — a caption benchmark is not a
  short-answer one — and written with `label` unset, which the script says out loud
  at the end of the run. The old refusal still fires for any record that declares
  neither an answer, nor accepted answers, nor this flag.
- `scripts/label_from_response.py` refuses a file whose records already carry labels,
  so one rule can never quietly overwrite another's work.

## [Unreleased] — content-safety residue removed

The content-safety family left this repository in the entry below, but the machinery
that served it did not: the MM-SafetyBench adapter, the refusal-reading keyword
judge, and the ASR scoring scripts all stayed, because a pipeline outside this
repository was reading them. The tree therefore still described two risks while the
model describes one — the 0.1.0 entry below records those files as *kept
deliberately*. That decision is reversed here. Content-safety code now lives only in
the separate module, so nothing under this tree serves a risk this repository does
not model, and the factual path is unchanged.

### Removed

- `evaluation/datasets/mm_safety.py` — the MM-SafetyBench adapter: the
  harmful-instruction parquet release into MSRC input records, plus
  `stratified_sample` for a category-balanced set. The benchmark is a content-safety
  one and the adapter labelled nothing; its records existed only as the input to the
  out-of-repo labelling run.
- `msrc/judge.py` — the keyword refusal classifier (`classify_refusal`,
  `label_from_refusal`, `uncertain_rate`), whose own docstring already stated that it
  was not part of the factual risk path.
- `scripts/mmsafety_asr_judge.py` — the official MM-SafetyBench ASR judging prompts
  and verdict parsing, for a local judge model standing in for the paper's GPT-4.
- `scripts/apply_asr_labels.py` — turned those judge verdicts into the `label` field
  of an evidence file, so the official ASR criterion could reach the indicator
  instead of the keyword proxy.
- the tests pinning the four modules above: `tests/test_dataset_mm_safety.py`,
  `tests/test_mm_safety_sampling.py`, `tests/test_mmsafety_asr_judge.py`,
  `tests/test_apply_asr_labels.py` and `tests/test_judge.py`.

### Changed

- `evaluation/datasets/__init__.py` no longer imports or exports `build_mm_safety`,
  and its adapter table lists POPE only. `evaluation/datasets/images.py` stays — the
  POPE adapter uses it — and only its rationale for being a shared module is
  reworded, since there is no longer a second adapter to share with.
- `scripts/group_report.py` and `docs/evaluation.md` no longer cite MM-SafetyBench's
  harm categories as a second example of subsets that are not interchangeable.
- `tests/test_groups.py` used an MM-SafetyBench row as a generic non-POPE fixture; it
  now uses a TextVQA row. The assertion is unchanged, because that test covers
  `image_group_id` — factual-path code that was not touched.
- `pyproject.toml` drops the `content-safety` keyword and states in the description
  that content-safety risk is modelled outside this repository.

The eight signals, the calibration, the metrics and the rest of the factual path are
untouched by this cleanup.

## [0.1.0] — unreleased

First working version. The framework and its accounting are complete; the
evaluation on public benchmarks is the next step, and no results are claimed here
until they have been produced.

### Added

- **Signal layer** (`msrc/signals.py`) — eight independent readings, five external
  and three internal, each with its provenance recorded:

  | signal | kind | borrowed from |
  |---|---|---|
  | `self_consistency` | external | SelfCheckGPT |
  | `resample_consistency` | external | Semantic Entropy |
  | `typed_verification` | external | typed evidence probing |
  | `cross_model_agreement` | external | cross-model / judge agreement |
  | `grounding_check` | external | retrieval- and rule-grounded checking |
  | `sequence_confidence` | internal | LLM-Check, SAPLMA |
  | `output_entropy` | internal | distributional uncertainty |
  | `visual_attention` | internal | attention-based hallucination work |

  Seven run with a single model; `cross_model_agreement` is the one a second model
  buys.

- **One risk: factual hallucination.** *Does the answer contradict the image?* The
  framework and its accounting are complete; content-safety risk is a different
  question with a different label and different evidence, and it is modelled
  outside this repository (see **Removed** below).

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

- **Refusal reading** (`msrc/judge.py`) — refusal detection in English and
  Chinese, with the awkward middle reported rather than hidden: a response that
  refuses and then complies is marked uncertain, and `uncertain_rate()` returns
  the fraction so a content-safety result can quote the size of the judgement call.
  This module is not part of the factual risk path; it is kept because the
  content-safety work that moved out of this repository labels with it.

- **Benchmark adapters** (`evaluation/datasets/`) — POPE for the factual path, and
  MM-SafetyBench into the record shape the out-of-repo content-safety labelling
  reads, plus `stratified_sample` for a category-balanced set.
  `pyarrow` is a declared extra because pandas reads parquet only through an engine
  it does not install.

- **Two model backends** (`msrc/providers/hf_vision.py`) — `HFLLaVAProvider` and
  `HFQwenVLProvider` over one shared implementation, selected with `--provider`.
  Both produce the same signals. The shared implementation handles both
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

- **`scripts/group_report.py`** — metrics per value of a grouping field, with the
  **base rate** beside each: what a system that never raises a warning scores. A
  pooled number over subsets that are not interchangeable describes no benchmark,
  and an accuracy quoted without its base rate is unreadable — on a group where the
  model is right 89% of the time, 89% accuracy is the floor, not a result.

- **POPE adapter** (`evaluation/datasets/pope.py`) — all three splits, plus
  `parse_yes_no`. POPE's reference is one word, so the label is what the framework is
  meant to predict with no answer-matching noise on top, which TextVQA cannot offer.
  Responses that say neither yes nor no return `None` and are counted as unparsed
  rather than guessed at.

- **`evaluation/datasets/images.py`** — the image-storage conventions both adapters
  need, in one place rather than a partial copy in the second one. Both adapters now
  also do their path and argument checks *before* importing pandas, so a wrong root
  reports the root instead of a missing optional dependency.

- **`examples/run_stub.py`** — the whole chain on a CPU with no model and no
  weights, run twice: once as a text-only platform and once with generation-time
  internals supplied, to show which channels a deployment can actually feed.

- CI on Python 3.9–3.12, plus a job that asserts the decision layer still works
  with NumPy alone and a job that checks the declared signal inventory against the
  code rather than against the prose.

### Removed

- **The content-safety half of the framework.** Content-safety risk is being
  modelled as a separate module outside this repository, so what went with it is:

  - `RISK_SAFETY` and the `risk_family` plumbing that threaded a family string
    through the types, provider, pipeline, service, CLI, scorer and provider
    adapters — along with the `families` / `applies_to` mechanism that existed only
    to keep the two families' signal sets disjoint;
  - the four safety-only signals (`policy_probe`, `refusal_strength`,
    `actionability`, `guard_model`) and their helpers, taking the declared
    inventory from 12 to **8: three internal and five external**;
  - the dual-head scorer (`msrc/dual.py`), now a single-head `msrc/scorer.py`;
  - `scripts/check_safety_labels.py` and `scripts/run_guard.py`, and the
    `policy_probe` / `--dual` / `--risk-family` channels in the collector;
  - the safety branches in `scripts/evaluate.py`, whose report now describes one
    risk, and the safety section of `scripts/audit_dsh_report.py`.

  Kept deliberately, because a pipeline outside this repository consumes them:
  `scripts/mmsafety_asr_judge.py`, `scripts/apply_asr_labels.py`, the
  MM-SafetyBench adapter, and the refusal reading in `msrc/judge.py`.

### Fixed

- **`evaluate.py` described every result as a factual one.** Pointed at
  content-safety evidence it printed a closing paragraph stating that the number
  was not a safety judgement. It now describes the one risk it reports, and
  content-safety evidence is not evaluated here at all.

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
  suite, and `msrc.signals.SIGNAL_NAMES` supplies the column list for the
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
  this project's evidence, and the flag is `True` on that basis. The same measurement
  found that `BY` rejects nothing at all at this sample size, and that the realized
  FDP conditional on the procedure firing is several times `alpha` even though the
  expectation is controlled. Both are documented, because the second is what a
  deployment actually experiences.

### Notes

- Content-safety work built on this repository carries its own error floor: a
  keyword refusal rule is a judgement, not an observation, and its quality depends
  entirely on the labels supplied to it. It is no longer measured or reported here.
- ECE is gameable: a constant predictor at the base rate scores exactly 0. The
  reporting helper prints Brier and AUROC beside it, and a test pins the property.
