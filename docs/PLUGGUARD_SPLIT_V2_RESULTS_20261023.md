# PlugGuard Split-v2 Full-run Results

The corrected split completed on 2026-10-23. This is an experimental result,
not a third-party certification or a guarantee about generated-answer safety.

## Scope and Identity

- Frozen inference/split source: `7a05a1cd124f0707ba5917fb6dfac5aa4a213f9b`.
- Development-only recovery implementation: `67c1d1afdd5c99dad2620a50a3d37fd16ca296f6`.
- Existing isolated dependency environment reused read-only; before/after pins,
  dependency freeze, and editable/path fingerprints matched.
- All 6,720 safety inputs were reprocessed. Original groups and exact duplicate
  image-content/question components did not cross development/test or
  fit/calibration boundaries; all four overlap checks were zero.
- Development 3,360 = fit 2,688 + calibration 672; held-out test 3,360.
  Base weights stayed frozen; the risk head was fitted for one epoch.
- A resource interruption occurred before fitting. The continuation verified
  3,154 same-experiment development feature files and generated the remaining
  206. Head fitting, calibration, all test inference, and answer judging were
  newly executed. This was not an uninterrupted 6,720-case single-process run.

## Held-out Request-risk Accuracy

| Source | n | TP | TN | FP | FN | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SD | 840 | 840 | 0 | 0 | 0 | 100% |
| SD_TYPO | 840 | 839 | 0 | 0 | 1 | 99.880952% |
| TYPO | 840 | 840 | 0 | 0 | 0 | 100% |
| Benign | 840 | 0 | 840 | 0 | 0 | 100% |
| Overall | 3,360 | 2,519 | 840 | 0 | 1 | 99.970238% |

The calibration-only threshold was 0.9725, frozen before testing. The head SHA256
was `9540ac28c85fef9dc951e073df0f255ee53be8f2aab889338f12a2405d911c40`.
The request-label accuracy exceeds both the 75% milestone and 85% completion
reference. The all-alarm baseline is 75%; it is not the achieved classifier.

The original safety checker, recovery feature hashes, frozen engineering files,
all 3,360 answer judgments, and 97,913 causal token events passed the final
independent checks. Six factual cells were reviewed from historical full
evidence, not newly collected in this run: minimum six actual participating
signals and ECE reductions of 32.45%, 81.35%, 38.20%, 56.44%, 64.65%, and 68.28%.
This does not constitute three freshly inferred metrics under one new run ID.

## Matched Historical Comparison

Image-content hashes, exact question hashes, and labels matched for all inputs.
Twelve fit/test records moved in each direction to repair the split; calibration
membership stayed unchanged. Only the 3,348 common held-out IDs were paired:

| Scope | Historical | Corrected split |
| --- | ---: | ---: |
| Common held-out correct / n | 3,346 / 3,348 | 3,347 / 3,348 |
| Common held-out accuracy | 99.940263% | 99.970131% |
| Common held-out FP / FN | 2 / 0 | 0 / 1 |

Two false alarms were corrected, but one new false negative appeared. The
common 3,348 development answers and generation lengths matched exactly; this
does not assert bit-identical hidden-state tensors or equal trained heads.
The historical threshold 0.764 is not interchangeable with the new threshold.

## Timing and Answer-safety Limitations

Historical warnings all triggered at token 1 with no earlier answer-token
release. The new run had two warnings at tokens 77 and 92, after 60 and 75 answer
tokens had been released. Each released token had been scored first, but it is
incorrect to claim that every warning preceded the first answer output.

Supplementary answer judging returned Safe 3,338, Controversial 14, and Unsafe
8. All 22 non-Safe judgments were on released answers and require human review.
These are automatic judgments, not confirmed harmfulness labels. The primary
metric uses request labels; it is not a 99.97% safe-answer rate or proof of zero
harmful-output leakage. The false-negative request's answer was judged Safe,
but that does not convert its request-risk miss into a correct decision.

The runtime evidence describes controlled local buffering, not a production
HTTP/streaming integration. Bounded GPU sharing is cooperative, not hardware
isolation. Frozen outputs were archived privately; no raw prompts, model head,
or generated answers are published here. No test-driven parameter changes were
made. See [test flow](PLUGGUARD_TEST_FLOW.md) and
[split protocol](SAFETY_SPLIT_V2.md) for execution requirements.
