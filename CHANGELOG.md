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

- **Evaluation layer** (`evaluation/metrics.py`) — AUROC with correct tie
  handling, Brier, ECE with a documented binning dependence, threshold metrics,
  and `summarise()` which returns all of them together so a favourable subset
  cannot be quoted by accident.

- **`examples/run_stub.py`** — the whole chain on a CPU with no model and no
  weights, run twice: once as a text-only platform and once with generation-time
  internals supplied, to show which channels a deployment can actually feed.

- CI on Python 3.9–3.12, plus a job that asserts the decision layer still works
  with NumPy alone and a job that checks the declared signal inventory against the
  code rather than against the prose.

### Notes

- The `safety` family's detection quality depends entirely on the policy taxonomy
  and labels supplied to it. This release provides the mechanism and the
  measurement, not a validated safety classifier.
- ECE is gameable: a constant predictor at the base rate scores exactly 0. The
  reporting helper prints Brier and AUROC beside it, and a test pins the property.
