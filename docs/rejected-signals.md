# Signals that were tried and did not work

Kept because a negative result costs the same to obtain as a positive one and
saves the next person from repeating it. Each entry says what was measured, on
what, and what came back. Where a candidate was *not* adopted, that is stated
plainly rather than left implied by its absence from `msrc/signals.py`.

Every number here was taken on the **dev half only**. The discipline for anything
that reaches a reported table is in `docs/evaluation.md`: thresholds are chosen on
dev, reported on test once, and a threshold ceiling is printed beside every result
so that "the operating point is wrong" stays distinguishable from "the score
cannot do better".

---

## Minimum instead of mean aggregation

**Idea.** `self_consistency` averages agreement across re-phrasings;
`resample_consistency` averages entropy across resamples. The object-level signal
gets its power from taking the **weakest** reading rather than the average, so the
same substitution was tried on the continuous signals: weakest view agreement,
weakest resample agreement, weakest cross-model agreement, weakest pairwise view
agreement, and the share of the answer's content words appearing in no view.

**Measured.** Six cells, five candidate features, added one at a time to the
nine-signal fusion, dev half only.

**Result: no gain.**

| cell | best increment | worst |
|---|---|---|
| LLaVA POPE | +0.0022 | −0.0009 |
| LLaVA TextVQA | +0.0000 | −0.0014 |
| LLaVA HallusionBench | +0.0134 | +0.0006 |
| LLaVA CHAIR | +0.0044 | −0.0001 |
| Qwen POPE | +0.0042 | −0.0045 |
| Qwen CHAIR | +0.0026 | +0.0000 |

Every increment sits inside the noise of an AUROC at these sample sizes, and
several are negative.

**Why, and what it implies.** The object-level signal works because objects have
*discrete identity*: "bike" in one resample and "bicycle" in another are the same
object returning, and an object either comes back or it does not. Free-text
agreement has no such unit — it is a continuous quantity, and the mean already
extracts what is there. Taking a minimum over a continuous quantity mostly selects
noise. The mechanism does **not** generalise to continuous readings, and the
negative result is the reason to say so rather than to keep trying variants.

---

## Claims-level minimum support

**Idea.** The entry above concludes that minimum-over-continuous-text fails because
free text has no discrete unit. Claims do: a proposition either recurs when the
model looks again or it does not. So the answer was split into sentences as a cheap
stand-in for claims, each sentence was scored by its *best* agreement against any
re-phrasing, resample or second-model answer, and the signal was the **weakest**
sentence.

**Measured.** Six cells, dev half only.

**Result: no gain.**

| cell | availability (needs ≥2 sentences) | standalone AUROC | increment |
|---|---|---|---|
| LLaVA CHAIR | 100% | 0.5967 | **−0.0031** |
| LLaVA POPE | 46.6% | 0.5122 | +0.0002 |
| LLaVA HallusionBench | 34.8% | 0.4950 | +0.0012 |
| Qwen CHAIR | 94.9% | 0.5201 | +0.0006 |
| Qwen HallusionBench | 60.9% | 0.5121 | +0.0004 |
| LLaVA TextVQA | 0.0% | — | — |

Every increment is inside the noise and one is negative.

**What this says about the earlier explanation.** The object-level signal was
credited with working because objects are discrete. That was half the story. The
other half is that an object's identity is *checkable*: "bike" and "bicycle" resolve
to one unit by a lexicon, without judging meaning. A sentence has no such resolution
— deciding that two sentences say the same thing is a semantic judgement, and the
agreement function here is string overlap, which is too blunt to carry it. Discrete
units are necessary but not sufficient; they also have to be *identifiable cheaply*.

Together with the entry above, this closes the "go finer-grained" direction for the
signals available from cached evidence. TextVQA answers are single phrases, so the
signal cannot even be taken there.

---

## Numeric grounding

**Idea.** HallusionBench is charts, tables, maps and diagrams, where hallucination
is mostly a number read wrong — 42 on a bar chart reported as 24. The existing
`grounding_check` overlaps the answer's content words against OCR text, in which
digits are a tiny fraction, and reads AUROC 0.4988 there. So: extract the numbers
the answer states, and ask of each whether it appears in the text read off the
image. Thousands separators are normalised first, because `1,280` / `1 280` / `1280`
are the same number on the page and typography is not information.

**Measured.** Six cells, dev half only.

**Result: no gain where it applies, and it barely applies.**

| cell | availability | standalone AUROC | increment to the fusion |
|---|---|---|---|
| LLaVA HallusionBench | 26.4% | 0.5597 | +0.0004 |
| LLaVA TextVQA | 19.6% | **0.8093** | −0.0001 |
| LLaVA CHAIR | 0.8% | — | — |
| LLaVA POPE | 0.1% | — | — |
| Qwen HallusionBench | 27.8% | 0.5373 | −0.0003 |
| Qwen POPE | 0.6% | — | — |

**The one interesting reading.** On the fifth of TextVQA items where it applies,
the signal is strong on its own and stratifies cleanly: answers whose numbers are
all absent from the image text are 84.1% hallucinated, against 21.2% when every
number is on the page. It still adds nothing to the fusion, because
`sequence_confidence` (0.8695 standalone there) and `grounding_check` already carry
that information. A signal that is strong alone and redundant in company is worth
recording as exactly that.

Not adopted.

---

## Object-level support on POPE

**Idea.** Point `object_support` at POPE, where it should catch the "Yes" answered
about an object the model never actually saw.

**Result: the signal is present and constant.** Availability is 99.8%, AUROC
0.5004. The reason is structural, not a bug: the primary answer is a full sentence
("Yes, there is a snowboarder in the image, riding down a snow-covered slope") while
the four re-phrasings are one-word "Yes"/"No", so no object ever returns and every
item scores the same. The constant-column check drops it, which is the designed
behaviour.

**What would make it work.** The re-readings have to be descriptions, not one-word
answers. That is what the free-description pilot collected.

### The free-description pilot: stopped early, and why

600 POPE images were re-asked four free-description questions each ("Describe this
image in detail", "List the objects and people visible", and two paraphrases), and
the queried object's mention rate across the four was the candidate signal.

**Partial read at 120 images: AUROC 0.6368, standard error 0.084, 95% interval
[0.472, 0.802].** The interval crosses 0.5, so it is not evidence of anything yet.

The reason to stop rather than collect the rest is structural, not statistical:
**76% of items (91 of 120) fell into a single bucket — the queried object was never
mentioned in any of the four descriptions.** A short free description of a COCO image
names a handful of salient objects, and the object POPE asks about is frequently not
among them, so "not mentioned" carries almost no information and the signal sits near
constant. Collecting 2,400 instead of 480 would not remove that; the probe's recall is
the problem, not the sample size. Runs that stayed landed at 15.4% hallucinated in the
never-mentioned bucket against 0% in the two buckets where it was mentioned — the
direction is right and the separation is real, but the mass is in the wrong bucket.

The collected 533 rows are kept at `p1_pope_desc.jsonl`; nothing was deleted. The
GPU it held was reassigned to the first-step margin pilot, which is a different class
of reading and had not been measured at all.

**The design fix, if this is ever revisited.** Ask for high recall rather than a
description — "list every object you can see" — so that absence becomes informative.
That is a different collection, not more of this one.

---

## Image contrast (VCD) on the chart benchmark

**Idea.** Visual Contrastive Decoding (Leng et al., CVPR 2024) contrasts the
next-token distribution with the image against the same distribution without it. Read
as a detection signal: if removing the picture barely changes what the model was
about to say, the answer came from the language prior. HallusionBench is charts,
tables, maps and diagrams, where a model has strong expectations about what such a
figure usually says, so this was the mechanism chosen to attack it.

**Measured.** Collected on all 951 HallusionBench items with LLaVA-1.5-13B, one
ablated forward per item with a same-size grey stand-in, Jensen-Shannon divergence in
bits, dev half only.

**Result: no evidence.**

| reading | dev AUROC |
|---|---|
| nine-signal baseline | 0.6968 |
| plus image contrast | 0.7079 |
| increment | **+0.0111** |
| bootstrap 95% interval | **[−0.0058, +0.0293]** |

The interval crosses zero. Test-half accuracy at the dev-chosen threshold was 65.6%,
against a never-warn baseline of 43.2% -- far from the bar and not moved by the new
reading.

**What this rules out, and what it does not.** It rules out "the answer did not
depend on the image" as a *sufficient* description of the chart failure. The
diagnosis it implies is narrower than the disease: a model that reads a bar chart
wrongly has looked at the image quite carefully and still produced a wrong number, so
its first-step distribution is not the distribution of a model answering blind.
Chart hallucination here is misreading, not ignoring.

It does not rule out image contrast elsewhere. The signal is cheap and the mechanism
is sound where the failure really is prior-driven; it simply is not the failure on
this benchmark. Nothing was adopted.

---
## Caption length and surface features

**Idea.** Longer captions have more chances to name an object that is not there.

**Result: it works and was rejected anyway.** Length, word-count and sentence-count
features lift CHAIR AUROC to 0.765 from 0.6256. The lift is mechanical — it encodes
"more claims, more chances to be wrong" rather than anything about the image — and a
method whose gain comes from a surface statistic would be reporting the statistic.
Not adopted. (The object-level signal reaches 0.8199 on the same cell by reading the
objects, which is the reason the shortcut was unnecessary.)
