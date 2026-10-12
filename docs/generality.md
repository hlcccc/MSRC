# What "one general method" has to mean here

The requirement: a single procedure that takes `(image, question, answer)` plus
whatever the serving stack exposes, and returns a hallucination risk — on a benchmark
it has never seen, and in a daily setting that is not a benchmark at all.

This note records where the current design fails that requirement, and what the
protocol has to be for the claim to mean anything. It is written before the work, so
that the work can be judged against it rather than the other way round.

## Three ways the current design is tailored, with the numbers that show it

### 1. A signal built on one benchmark's ontology

`object_support` reads the objects an answer names and checks whether they survive a
second look. The mechanism is general. The **lexicon is not**: it is the official
CHAIR synonym list, which is COCO's 80 categories. On CHAIR it is worth +0.2593 AUROC.
Its median contribution on the other six cells is +0.0002.

The failure is not abstract. COCO's eighty categories contain `bench`, `toilet` and
`snowboard`. They do not contain `receipt`, `whiteboard`, `prescription label`,
`bus timetable` or `menu` — which is to say, they do not contain most of what a
camera pointed at daily life actually sees. A signal keyed to them works on the
benchmark and is absent in deployment.

`scripts/signal_audit.py` flags it: concentration +0.2591, against a noise floor of
0.004.

### 2. A calibrator fitted per benchmark

Every cell in the reported table fits its **own** weights on its own dev half. That
is the normal way to report a benchmark number and it is still a form of tailoring:
the weights that matter on a chart benchmark are not the weights that matter on
object questions, and fitting them separately means the table describes seven
fitted models rather than one method.

A method that claims to generalise has to be fitted somewhere and **applied**
elsewhere. Nothing in this repository has been evaluated that way.

### 3. Signals keyed to answer format

`decision_margin` reads the first-step distribution. On yes/no benchmarks that is the
decision. On a free-form caption the first token is `The`, and the reading is close to
meaningless — it happens to be unavailable on the cached evidence, so it has not yet
done harm, but its design assumes a short-answer format.

## The protocol that would make the claim testable

**Leave-one-dataset-out.** Fit the fusion on every dataset except one, freeze it, and
apply it unchanged to the held-out dataset. Its own dev half is used only for the
threshold, never for weights. Report that number, next to the per-dataset-fit number,
and treat the gap between them as the price of generality.

This is the audit mechanism, and it is not optional: any signal whose contribution
collapses under leave-one-dataset-out was doing per-dataset work.

Two further constraints follow from the same requirement:

- **Resources must be dataset-independent.** A general object vocabulary rather than
  one benchmark's ontology; OCR; a second model; an entailment model; the model's own
  internals. Anything whose vocabulary or label rule comes from a benchmark is out.
- **The daily-scenario test.** If a change helps only when the input looks like the
  benchmark — same question phrasing, same object inventory, same answer length — it
  is not a general improvement. The honest check is whether the signal is even
  *available* on an arbitrary photograph, which the availability table in
  `signal_audit.py` already reports per cell.

## What this costs, stated now rather than discovered later

**The numbers will go down.** Per-dataset fitting is strictly more flexible than one
shared fit, so leave-one-dataset-out can only be equal or worse on every cell. Some
cells currently at 89% will not stay there.

That is not a reason to keep the tailored version. A number produced by seven
separately fitted models does not support the claim "our method works", and the
requirement here is the claim, not the number.

## What survives this filter

| keep | because |
|---|---|
| the model's own internals — confidence, entropy, first-step margin | no external resource at all |
| OCR grounding | reads whatever text is in the picture, no vocabulary |
| a second model's opinion | no vocabulary |
| resampling and re-phrasing consistency | no vocabulary |
| entailment-based clustering of samples | a general NLI model, not a benchmark ontology |
| a general object vocabulary | still a lexicon, but not one benchmark's |

| drop or rebuild | because |
|---|---|
| `object_support` on the CHAIR/COCO lexicon | one benchmark's ontology |
| per-dataset calibrator fits in the headline table | tailoring by another route |
| any answer-format-specific reading | does not survive a change of format |
