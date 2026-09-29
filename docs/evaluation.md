# Evaluation protocol

How a number from this repository is produced, in enough detail that someone else
can produce the same one. If a claim cannot be traced to a section here, treat it
as unverified.

## What is measured

Three quantities, reported together because each is uninterpretable alone:

| quantity | answers | where |
|---|---|---|
| **AUROC** | can a risky item be ranked above a safe one? | `evaluation.metrics.auroc` |
| **ECE** | is the stated probability honest? | `evaluation.metrics.ece` |
| **threshold metrics** | at a fixed operating point, what is the warning accuracy? | `evaluation.metrics.threshold_metrics` |

AUROC says nothing about whether the probabilities are calibrated. ECE says nothing
about whether the classes are separated — **a predictor that outputs the base rate
for every item has an ECE of exactly 0 and an AUROC of 0.5**. Any claim resting on
an ECE gain alone can be beaten by that predictor, which is why
`evaluation.metrics.summarise` returns all three and never one.

## The two labels

| family | label | source |
|---|---|---|
| `factual` | the answer contradicts the image | dataset ground truth, by the VQA agreement rule |
| `safety` | the answer did not refuse a harmful request | `msrc.judge`, keyword refusal detection |

The safety label needs two things that are easy to confuse with each other, and
only one of them is the label:

- **The label** is whether the model refused. There is no ground-truth answer to
  compare a response against — there is a harmful request, and what is judged is
  whether the model went along with it. `msrc.judge` decides this from the response
  text alone, which is what makes it reproducible; it is also a judgement, not an
  observation, so its disagreement rate against hand labels is a number that
  belongs beside any result (`scripts/check_safety_labels.py` measures it).
- **The taxonomy** is what `policy_probe` asks the model to judge a response
  against, and it has no default. What counts as unsafe is a policy decision that
  varies by deployment and jurisdiction, and a hard-coded list would be wrong
  everywhere except the place it was written.

A run that does not state both is not reproducible, and should not be reported as
one. The two are independent: changing the taxonomy changes one signal, and the
label stays the refusal rule.

## Protocol

### 1. Split by image, not by row

Several items can share an image. A row-wise split puts near-duplicates of the
training set into the evaluation set and inflates every metric. `evaluate.py`
splits on the `image` field and refuses to continue if either side ends up single
class.

The split is seeded (`--seed`, default `20260920`) and reported.

### 2. Generate the answer under evaluation, capturing its internals

This is the step that decides how many signals run, and it cannot be recovered
afterwards:

```bash
python scripts/collect_evidence.py \
    --data textvqa.jsonl --out evidence.jsonl \
    --model-path /path/to/llava --k 3
```

`textvqa.jsonl` carries `question`, `image` and `gold_answers`. The collector
generates the answer, records the token logprobs, entropy and attention **of that
generation**, and labels it by the standard VQA agreement rule.

`--provider` names the architecture (`llava` or `qwen`). It changes nothing about
the signals — the same seven run either way — and it exists because the safety
family's label needs a model that refuses some harmful requests and complies with
others. Check that before collecting a set: the collector prints the label
distribution, and a single-class file is refused by `evaluate.py` because there is
nothing in it to rank.

If the answers already exist — a published run, a production log — pass them as
`answer` with a `label` instead. That is a legitimate use, and the three internal
signals will report themselves unavailable, because the logprobs of *that*
generation were never recorded and re-running the model produces a different one.

### 3. Cache the evidence

Evidence costs four re-phrasings, K resamples and three probes per item, all
needing the model. It is written as JSONL and re-read by every later step, so
threshold selection, ablations and the report are CPU jobs. It also makes the
readings auditable: each signal's value, its raw reading and a one-line
justification are in the file.

Interrupted runs resume: items already carrying evidence are copied through.

### 4. Fit and report

```bash
python scripts/evaluate.py --data evidence.jsonl --json
```

The report states, in this order:

1. **how many signals produced a value**, split internal and external, with the
   constant ones named. "Nine signals are defined" and "six ran on this data" are
   different claims and only the second one is a result;
2. **the ECE gain** against the strongest single signal, with the value across
   several binnings beside it, because ECE moves with the binning and a quoted
   figure has to name its setting;
3. **AUROC and the threshold metrics**, because the project's indicator says
   准确率 — a threshold quantity — while the natural thing to quote is AUROC, and
   the two can disagree about whether a bar was cleared.

## Reporting rules

- **Quote the binning with any ECE.** `n_bins=15, uniform` unless stated.
- **Quote Brier or AUROC with any ECE gain.** See the constant predictor above.
- **Say which operating point an accuracy is at, and whether it was tuned.**
  `evaluate.py` reports at `--threshold 0.5` and does not tune it. A threshold
  chosen to maximise accuracy is a fitted parameter: pick it on the dev split, and
  say so. Accuracy at the default and accuracy at a tuned threshold answer different
  questions, and a tuned one quoted without the rule reads as a property of the
  score.
- **Say which signals ran.** Eight apply to either family; four run on a text-only
  serving stack, seven with one model at full access, eight with a second model
  attached. The count in the report is the one that happened, and the signals that
  could not run are named there.
- **Say what the label is.** A factual-hallucination number is not a content-safety
  number, and the indicator's wording ("多模态生成内容风险") could be read either way.
  For `safety`, also give the rate at which the refusal rule was unsure of itself.
- **Report the split sizes and the seed**, so the number can be reproduced rather
  than merely believed.

## Reproducing a published run

The evidence file plus the seed is the whole state. Given both:

```bash
python scripts/evaluate.py --data evidence.jsonl --seed 20260920
```

reproduces the report exactly, with no model and no GPU.
