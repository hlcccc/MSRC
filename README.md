# MSRC — multi-signal risk calibration for multimodal generation

> **PlugGuard acceptance branch, 2026-10-19.** The five entry points and the
> factual eight-signal protocol are unchanged; step 04 now uses PlugGuard only.
> [Test flow](docs/PLUGGUARD_TEST_FLOW.md) |
> [Version boundaries and SingGuard rollback](docs/SAFETY_VERSIONS.md).
> The original SingGuard code is retained at tag
> `singguard-before-plugguard-20261019`; never mix its operating point or results
> with this branch. Model weights, trained heads, and experimental predictions
> are not stored in Git. The [public six-step acceptance flow](docs/ACCEPTANCE_FLOW.md)
> downloads public originals, rebuilds the benign inputs from a fixed recipe,
> and trains all acceptance heads/calibrators afresh, without a local delivery package.

**Twelve candidate signals, seven external and five internal, mapped to a
calibrated hallucination-risk score from a single input.**

MSRC scores a *frozen* model's answer after the fact. It does not retrain or
modify the model under evaluation, and it does not need its weights — but it does
use more than the model's text, which is the point:

| | signal | kind | borrowed from |
|---|---|---|---|
| 1 | `self_consistency` | external | SelfCheckGPT (Manakul et al., EMNLP 2023) |
| 2 | `resample_consistency` | external | Semantic Entropy (Kuhn et al., ICLR 2023) — string-cluster form, see note below |
| 3 | `typed_verification` | external | typed evidence probing |
| 4 | `cross_model_agreement` | external | cross-model / judge agreement |
| 5 | `grounding_check` | external | retrieval- and rule-grounded checking |
| 6 | `object_support` | external | CHAIR's object-level rule, read as a signal |
| 7 | `cross_model_object_support` | external | the same reading, taken from a second model |
| 8 | `sequence_confidence` | **internal** | LLM-Check, SAPLMA |
| 9 | `output_entropy` | **internal** | distributional uncertainty |
| 10 | `decision_margin` | **internal** | first-token margin instead of a whole-answer mean |
| 11 | `image_reliance` | **internal** | Visual Contrastive Decoding (Leng et al., CVPR 2024) |
| 12 | `visual_attention` | **internal** | attention-based VLM hallucination work |

The risk is factual: **does the answer contradict the image?** Content-safety risk
— does the answer violate a stated policy — is a different question with a
different label and different evidence. It does not share this factual decision
layer: this branch uses a separate PlugGuard adaptation, development-only head
fitting and threshold selection, and a causal per-token output gate during
generation. The original SingGuard pre-generation gate remains in the rollback
tag, not as a fallback inside this branch's step 04.

### Where the borrowed names overstate what is implemented

Two of the names in the table are looser than their citation suggests. The
difference is stated here rather than left to be discovered in the code:

* **`resample_consistency` is a string-cluster entropy, not semantic entropy.**
  Re-sampled answers are clustered by normalised string equality, with no entailment
  or paraphrase judgement. `a red car` and `the car is red` therefore fall into two
  clusters and score maximum entropy, and the containment rule in `agreement()`
  scores `a car` against `no car` at 0.9. Kuhn et al. (ICLR 2023) cluster by
  bidirectional entailment; this does not, so it must not be described as a faithful
  implementation of that method.
* **The legacy decision-layer default fills missing readings with `risk = 0.0`.** The value is
  recorded as unavailable and named in the fit warnings, but the number that reaches
  the calibrator is 0.0 — a *low-risk* value. On a column that is unavailable for
  every item the column is constant and the L2 penalty drives its weight to zero,
  which is harmless. On a column that is available for a minority of items it is
  not: "could not be read" is then encoded as "no risk", and the model can learn
  that absence of evidence is evidence of safety. See
  `docs/signal-applicability.md` for the per-cell counts. The fixed acceptance
  pipeline instead fills from available development-set readings, removes
  development-constant columns, and records availability separately; it does not
  count imputed readings as signals actually used on a test sample.

### How many of them actually run

Twelve signal types are defined: **seven external and five internal**. What a
deployment can feed decides how many of them produce a reading:

| configuration | signals that run |
|---|---|
| text-only serving stack, one model | **5** (the external ones that need no internals and no second model) |
| one model at full access | **10** |
| a second model attached | **12** |

Those three rows are what `msrc signals` prints; they are counts of the registry,
not of any particular dataset. On real data the number is usually lower again,
because a signal whose evidence is absent on every item -- a caption benchmark that
names no object in the lexicon, a yes/no benchmark with nothing to ground -- reports
itself unavailable rather than constant. `msrc explain` is what tells the two apart.

A second model buys **two** signals, and both read the same channel at different
resolutions: `cross_model_agreement` compares the whole answer, and
`cross_model_object_support` asks, object by object, whether the other model also
failed to name what this answer names. The reason the cross-model channel exists is
narrow but real -- resampling *one* model is correlated with itself, so a
systematic blind spot (a chart the model always misreads) makes every sample agree and
every consistency signal report confidence. A different model is the only way to break
that correlation, and `cross_model_object_support` is what reads it at the resolution
one wrong object in an otherwise correct caption needs.

The five internal readings (`sequence_confidence`, `output_entropy`, `decision_margin`,
`image_reliance`, `visual_attention`) are kept deliberately: "is the model unsure" is
evidence about whether an answer is grounded. Token confidence, entropy, margin
and visual attention require internals recorded during the original generation;
`image_reliance` additionally requires a masked-image forward pass. The fixed
eight-signal acceptance protocol does not select `image_reliance` or
`visual_attention`.

For any of these configurations, the signals that cannot run say so by name in the
result.

## The two ideas worth reading the code for

**Signals are oriented, and missing is not zero.** Every signal returns a `risk`
in `[0, 1]` where higher always means riskier, so the calibrator only has to
weight signals and never to discover their signs. A signal the deployment cannot
supply returns `available=False` and is *named* in the result — it is never
silently filled with a neutral value. A constant column trains a calibrator that
looks healthy and encodes nothing, which is the failure mode this framework was
built to make impossible.

**Internal signals need to be captured, not recovered.** A serving stack that
returns text only is supported: the five internal signals report themselves
unavailable and the other five carry on. But if the platform *can* expose token
logprobs, it must record them **when it generates the answer being evaluated** —
re-running the model afterwards produces a different generation, and its
internals are not the ones that produced the answer. Pass them in as
`primary_sample`.

## Install

```bash
git clone https://github.com/hlcccc/MSRC.git && cd MSRC
pip install -e ".[test]"
python -m pytest -q
```

The decision layer is NumPy-only. The model adapters live behind extras:

```bash
pip install -e ".[hf,ocr]"       # torch + transformers + rapidocr
pip install -e ".[datasets]"     # pandas + pyarrow, for the benchmark adapters
pip install -e ".[service]"      # the HTTP service
```

> `pyarrow` is listed rather than assumed. pandas reads parquet only through an
> engine it does not install, and without one the failure surfaces deep inside
> pandas as a message about pyarrow that never mentions what was being done.

## Acceptance Reproduction

For project acceptance, use the [six-step acceptance flow](docs/ACCEPTANCE_FLOW.md),
not the small examples below. It rebuilds inputs from raw POPE/TextVQA and
MM-SafetyBench assets, collects new model evidence, fits only on development
groups, then validates complete artifacts before judging the metric targets.

The acceptance protocol fixes **eight candidates out of the twelve registered
signals**. Counts of signals actually participating in fusion are checked per
test sample. Missing artifacts or smoke runs produce `INCOMPLETE`, not a pass;
`RUN_COMPLETED` alone does not mean `ACCEPTANCE_PASS`.

Public models and benchmarks are not the whole delivery: the custom 1,680 benign
samples and frozen non-weight model metadata must also be supplied. Pin the code,
model inventory, environment and raw assets before collection; do not update a
running frozen experiment. CPU regression tests verify behavior, not the
performance of an unfinished GPU benchmark. See the flow document for the
separate factual/safety labels and the read-only auxiliary-answer boundary.

## Commands

```bash
msrc signals                     # what the signal layer defines, and what runs
msrc signals --second-model      # the same, with a second model attached
msrc demo                        # the whole chain on a CPU, no model
msrc explain  --data evidence.jsonl
msrc fit      --data evidence.jsonl --out scorer.json
msrc score    --scorer scorer.json --evidence one.json
msrc evaluate --data evidence.jsonl
```

`msrc signals` reports the honest count — twelve defined, **ten that run with a
single model** — and names the two signals that need a second model.

`msrc explain` is the first thing to run on real data: it says which signals
carry information and which came out constant, because "twelve signals are defined"
and "six ran on this data" are different claims and only the second is a result.

`msrc score` refuses a scorer with no fitted coefficients. Returning an
uncalibrated prior as though it were a calibrated probability is how a caller ends
up trusting a number that means nothing.

## Run it without a model

The stub provider produces plausible readings on every channel, including the
internal ones, so the whole chain — signals, calibration, fusion guard,
evaluation — runs on a CPU in seconds:

```bash
python examples/run_stub.py
```

It also demonstrates the difference the internal signals make, by running the
same data twice: once with a text-only platform (four signals live) and once with
generation-time internals supplied (seven live).

`examples/run_stub.py` is explicit that its numbers are meaningless — the stub is
not a detector. It proves the wiring, not the method.

## Scoring

```python
from msrc import MSRCConfig, MSRCPipeline, Sample

pipeline = MSRCPipeline(provider=your_provider, config=MSRCConfig(k=5))
pipeline.fit(records)          # records: {"question", "answer", "image", "label"}

result = pipeline.score(
    "what is the website?",
    "Flickr",
    "/data/img.jpg",
    primary_sample=Sample(text="Flickr", sequence_confidence=0.88,
                          sequence_entropy=0.5, visual_attention_mass=0.62),
)
print(result.risk_score, result.calibrated_confidence, result.is_high_risk)
for signal in result.signals.signals:
    print(f"  {signal.name:24} risk={signal.risk:.3f}  {signal.detail}")
```

`result.signals` carries every reading with a one-line justification, so a score
can be explained rather than merely reported.

## Labels, and why you must bring your own

The calibrator is fitted on **your** labelled items, and the signals transfer
between models while the coefficients do not. That is not a limitation to work
around; it is what calibration means.

The label is the dataset's own ground truth: does the answer contradict the image
(the VQA agreement rule against the accepted answers, or POPE's yes/no reference).
The other kind of content risk — does the answer violate a stated policy — needs a
different label and a policy taxonomy the deployment owns, and it is modelled
outside this repository.

## Selecting, not just scoring

A risk score answers "how likely is this wrong". A deployment usually needs a
second decision: **which items do I act on, and which go to a human?** That is a
selection problem with a guarantee, and split conformal prediction is how to get
one:

```python
from msrc import select, selective_report

accepted = select(calibration_null_scores, test_scores, alpha=0.10, procedure="BY")
report = selective_report(calibration_null_scores, test_scores,
                          test_labels=labels, alpha=0.10)
```

The calibration set must hold items known to satisfy the null — correct answers —
and must be exchangeable with the test items. Nothing in the code can check that;
the caller owns it.

`scripts/validate_conformal.py` measures the guarantee instead of asserting it, over
repeated three-way splits. On this project's data, with `BH`, `E[FDP]` stayed at or
under `alpha` at every level tested, and `VALIDATED` is `True` on that basis. Read it
as the narrow claim it is — it says nothing about a deployment whose calibration set
is not exchangeable with its traffic. Two things the measurement also turned up, and
they matter more in practice than the headline:

> **The expectation is not what a deployment sees.** `E[FDP]` averages over all
> draws including the ones that reject nothing, where the false discovery proportion
> is zero by definition. Conditional on the procedure actually firing, the realized
> FDP was several times `alpha`. The report keeps `realized_fdp` separate from the
> guarantee for this reason.

> **`BY` had no power at this scale.** It rejected nothing in 100 draws out of 100 up
> to `alpha = 0.15`. Split-conformal p-values are multiples of `1/(n+1)`, so a
> rejection needs roughly `n_calibration >= n_test / alpha` *null* items before it is
> even possible, and `BY`'s harmonic correction puts every threshold below that
> floor at a few hundred items a side. It remains the safer choice under dependence;
> it is not a usable one at this sample size. Check the requirement before choosing
> it.

## Measuring

`evaluation/metrics.py` produces the numbers a result needs, and `summarise()`
returns all of them together so a favourable subset cannot be quoted by accident:

```python
from evaluation.metrics import format_summary, summarise
print(format_summary(summarise(labels, scores)))
```

The output has this shape — **these are placeholder values showing the format, not
a result from this project**:

```
  items            : <n>   positive rate <p>
  AUROC            : 0.????        <- can it rank a risky item above a safe one
  Brier            : 0.????        <- proper score; the one ECE cannot replace
  ECE              : 0.????   (n_bins=15)
  threshold 0.50   : acc 0.????  prec 0.????  rec 0.????  F1 0.????
  ECE across binnings: 0.???? .. 0.????   <- quote the setting with the value
```

> **ECE is gameable, and the summaries say so.** A predictor that outputs the base
> rate for every item has an ECE of exactly `0` and an AUROC of `0.5`. ECE measures
> whether the stated probability matches the observed frequency; it says nothing
> about whether the classes are separated. Never quote an ECE gain without Brier or
> AUROC beside it.

## Results

**No results are claimed in this repository yet.** The framework, the signal layer,
the calibration and the evaluation are complete and tested; the evaluation on
public benchmarks has not been published here, and a number that has not been
produced should not appear in a table.

[docs/evaluation.md](docs/evaluation.md) is the protocol, and the three commands
that produce a result are:

```bash
# object hallucination: POPE. One-word references, so correctness is not
# entangled with answer formatting.
python -c "from evaluation.datasets.pope import SPLITS, build_records, write_jsonl; \
           write_jsonl(build_records('/path/to/pope', 'out', splits=SPLITS), 'pope.jsonl')"
python scripts/collect_evidence.py --data pope.jsonl --out evidence_pope.jsonl \
    --model-path /path/to/llava --device cuda:1 \
    --second-model-path /path/to/qwen2.5-vl --second-provider qwen --second-device cuda:0 \
    --k 3 --want-attention --max-new-tokens 32
python scripts/evaluate.py --data evidence_pope.jsonl --json
python scripts/group_report.py --data evidence_pope.jsonl --group-by split

# open-ended: TextVQA, or any VQA set with accepted answers
python scripts/collect_evidence.py --data textvqa.jsonl --out evidence.jsonl \
    --model-path /path/to/llava --k 3 --want-attention
python scripts/evaluate.py --data evidence.jsonl --json
python scripts/threshold_analysis.py --data evidence.jsonl
```

Passing `--second-model-path` turns on the two cross-model signals. The two models can
sit on different cards with `--device` / `--second-device`, which on a shared machine is
usually cheaper than packing a 13B and a 7B onto one.

**Read the per-group report, not just the pooled one.** POPE's `random`, `popular`
and `adversarial` splits draw their negative examples differently and are not
interchangeable; `group_report.py` prints each separately together with the base
rate a never-warn system scores, which is the number an accuracy has to beat.

**Check the label distribution before collecting.** A file where every item carries
the same label cannot be ranked or calibrated, and `evaluate.py` will refuse it. The
collector prints the label distribution and says so when there is only one class, so
a handful of items is enough to find out rather than an hour of GPU time.

`--provider` selects the architecture (`llava` or `qwen`); both produce the same
signals, and nothing else in the pipeline knows which one ran.

The report states which signals ran, the ECE gain with its binning, AUROC and the
threshold metrics.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The licence covers this repository's code, not any third-party model weights or
datasets it can be pointed at. LLaVA-1.5 checkpoints derive from Meta's Llama-2
family and carry the Llama 2 Community License; Qwen2.5-VL carries its own terms.
