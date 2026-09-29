# MSRC — multi-signal risk calibration for multimodal generation

**Nine independent signals, five external and three internal, mapped to a
calibrated risk score for two risk families.**

MSRC scores a *frozen* model's answer after the fact. It does not retrain or
modify the model under evaluation, and it does not need its weights — but it does
use more than the model's text, which is the point:

| | signal | kind | borrowed from |
|---|---|---|---|
| 1 | `self_consistency` | external | SelfCheckGPT (Manakul et al., EMNLP 2023) |
| 2 | `resample_consistency` | external | Semantic Entropy (Kuhn et al., ICLR 2024) |
| 3 | `typed_verification` | external | typed evidence probing |
| 4 | `cross_model_agreement` | external | cross-model / judge agreement |
| 5 | `grounding_check` | external | retrieval- and rule-grounded checking |
| 6 | `policy_probe` | external | policy self-report, Llama-Guard style |
| 7 | `sequence_confidence` | **internal** | LLM-Check, SAPLMA |
| 8 | `output_entropy` | **internal** | distributional uncertainty |
| 9 | `visual_attention` | **internal** | attention-based VLM hallucination work |

Two risk families share the whole decision layer:

* **`factual`** — does the answer contradict the image?
* **`safety`** — does the answer violate a content policy?

What differs is the label, the evidence gathered, and which family-specific
signal applies (`grounding_check` for factual, `policy_probe` for safety).

### How many of them actually run

Nine signal types are defined. Eight apply to either risk family — and **one of
those eight needs a second model**, so the number that run depends on what the
deployment can feed:

| deployment | signals that run | composition |
|---|---|---|
| **one model** (the usual case) | **7** | 3 internal + 4 external |
| one model + a second VLM | 8 | 3 internal + 5 external |
| text-only serving stack | 4 | 0 internal + 4 external |

A single model is enough, and it is what this framework is built around: the
platform serves one model under evaluation, not two. The optional eighth signal,
`cross_model_agreement`, is the only thing a second model buys, and its reason for
existing is narrow but real — resampling *one* model is correlated with itself, so
a systematic blind spot (a chart the model always misreads) makes every sample
agree and every consistency signal report confidence. A different model is the
only way to break that correlation. Whether it earns its cost is a deployment
decision; the framework reports the count either way rather than assuming.

For either family and any of these configurations, the signals that cannot run
say so by name in the result.

## The two ideas worth reading the code for

**Signals are oriented, and missing is not zero.** Every signal returns a `risk`
in `[0, 1]` where higher always means riskier, so the calibrator only has to
weight signals and never to discover their signs. A signal the deployment cannot
supply returns `available=False` and is *named* in the result — it is never
silently filled with a neutral value. A constant column trains a calibrator that
looks healthy and encodes nothing, which is the failure mode this framework was
built to make impossible.

**Internal signals need to be captured, not recovered.** A serving stack that
returns text only is supported: the three internal signals report themselves
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

## Commands

```bash
msrc signals                     # what the signal layer defines, and what runs
msrc signals --family safety     # the same, for the other risk family
msrc demo                        # the whole chain on a CPU, no model
msrc explain  --data evidence.jsonl
msrc fit      --data evidence.jsonl --out scorer.json
msrc score    --scorer scorer.json --evidence one.json
msrc evaluate --data evidence.jsonl
```

`msrc signals` reports the honest count — nine defined, eight applicable, **seven
that run with a single model** — and names the one that needs a second.

`msrc explain` is the first thing to run on real data: it says which signals
carry information and which came out constant, because "nine signals are defined"
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
same data twice: once with a text-only platform (five signals live) and once with
generation-time internals supplied (eight live).

`examples/run_stub.py` is explicit that its numbers are meaningless — the stub is
not a detector. It proves the wiring, not the method.

## Scoring

```python
from msrc import MSRCConfig, MSRCPipeline, Sample

pipeline = MSRCPipeline(provider=your_provider, config=MSRCConfig(k=5, risk_family="factual"))
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

For the `safety` family the framework also requires a **policy taxonomy**. It does
not ship one, deliberately: what counts as unsafe is a policy decision that varies
by deployment and jurisdiction, and a hard-coded list would be wrong everywhere
except the place it was written. Supply your categories through the probe prompt
and label accordingly.

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
the caller owns it. BY is the default because scores from one model on items that
share an image are not independent.

> **`VALIDATED` is `False`.** The procedure is implemented from the standard
> definition and unit tested against textbook cases; it has not been checked on
> this project's data. The flag travels in every report, because a docstring cannot
> stop a number from being quoted and a field can at least make it visible. The
> report also keeps `realized_fdp` separate from the guarantee: the guarantee is on
> the expectation over draws, and the realized value is one draw.

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

[docs/evaluation.md](docs/evaluation.md) is the protocol, and the two commands that
produce a result are:

```bash
# factual: TextVQA, or any VQA set with accepted answers
python scripts/collect_evidence.py --data textvqa.jsonl --out evidence.jsonl \
    --model-path /path/to/llava --k 3 --want-attention
python scripts/evaluate.py --data evidence.jsonl --json

# safety: MM-SafetyBench, labelled by refusal after generating
python -c "from evaluation.datasets import build_mm_safety as b; \
           from evaluation.datasets.mm_safety import write_jsonl; \
           write_jsonl(b('/path/to/mm_safety', 'out', attacks=['TYPO']), 'mmsafety.jsonl')"
python scripts/collect_evidence.py --data mmsafety.jsonl --out safety.jsonl \
    --model-path /path/to/llava --risk-family safety
python scripts/evaluate.py --data safety.jsonl --json
```

The safety labels are a keyword judgement, not an observation, so measure the
judgement before quoting anything fitted on it. `--make-template` writes a sheet of
responses with the human column blank; fill it in and the check reports the
agreement, Cohen's kappa and every disagreement with its text:

```bash
python scripts/check_safety_labels.py --make-template safety.jsonl \
    --out to_label.jsonl --sample 100
# read each response, set "human_refused" to true or false
python scripts/check_safety_labels.py --data to_label.jsonl --min-agreement 0.9
```

The report states which signals ran, the ECE gain with its binning, AUROC and the
threshold metrics, and — for the safety family — the rate at which the label rule
was not confident, because a refusal classifier's uncertain cases are a real part
of the error floor and not a detail.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The licence covers this repository's code, not any third-party model weights or
datasets it can be pointed at. LLaVA-1.5 checkpoints derive from Meta's Llama-2
family and carry the Llama 2 Community License; Qwen2.5-VL carries its own terms.
