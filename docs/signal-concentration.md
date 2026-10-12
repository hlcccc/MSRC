# Signal concentration: a diagnostic, not a verdict

## Status of the requirement that produced this file

An earlier round treated "one method that must work unchanged on any dataset and in
daily use" as a requirement, and this file argued the design violated it. **That
requirement was withdrawn.** Signals may be specialised to a benchmark; nothing here
is a violation, and no rebuild is pending on this account.

What survives is a measurement worth having when reading the table.

## The measurement

`scripts/signal_audit.py` reports, per signal, its availability and its leave-one-out
contribution on each cell, then ranks signals by how concentrated their contribution
is — the largest contribution minus the median of the rest, in AUROC units. Noise
floor 0.004. Dev half only.

| signal | largest contribution | median elsewhere | concentration |
|---|---|---|---|
| `object_support` | **+0.2593** (CHAIR) | +0.0002 | +0.2591 |
| `cross_model_agreement` | **+0.1112** (HallusionBench) | +0.0009 | +0.1104 |
| `self_consistency` | +0.0461 (HallusionBench) | +0.0011 | +0.0450 |
| `visual_attention` | +0.0156 | −0.0000 | +0.0156 |
| `resample_consistency` | +0.0159 | +0.0017 | +0.0142 |
| `grounding_check` | +0.0119 | +0.0021 | +0.0099 |
| `typed_verification` | +0.0090 | +0.0018 | +0.0071 |
| `sequence_confidence` | +0.0034 | +0.0003 | — |
| `output_entropy` | +0.0030 | +0.0007 | — |
| `decision_margin` | +0.0000 | +0.0000 | — (not yet collected) |

## Why the two concentrated signals are concentrated

Both are structural facts about the benchmark, and knowing them prevents a misreading
of the table.

**`object_support` is tied to COCO's ontology.** Its lexicon is the official CHAIR
synonym list, which is COCO's 80 categories. CHAIR's labels are also built from those
categories. The mechanism — does each object the answer names survive a second look —
is independent of the lexicon, and treating the lexicon as a plain noun filter with no
synonym merging still reaches AUROC 0.78 against 0.82 with merging, so the ontology
supplies part of the strength and the mechanism the rest. On the other cells the
lexicon mostly does not apply: COCO's categories contain `bench` and `toilet`, not
`receipt` or `whiteboard`.

**`cross_model_agreement` matters where the task is hard.** It is worth +0.111 on
HallusionBench and +0.0477 on POPE, and near zero on CHAIR and TextVQA where other
signals already carry the information. That is a statement about the benchmarks'
difficulty, not about the signal being tuned to one of them — the same reading is
computed identically everywhere, and it is the only signal whose absence is a stated
deployment cost (attaching a second model buys exactly this one signal).

## How to read the table with this in hand

A cell whose accuracy leans on a concentrated signal is leaning on something
benchmark-shaped. That is legitimate and it should be stated when the number is
quoted, because a reader who takes the cell at face value will over-generalise it.
Two examples already in the repository, both stated in the CHANGELOG rather than
buried here: CHAIR's headline gain comes mostly from `object_support`, and the
HallusionBench cells are carried by the second model.

## What this file is not

It is not a rule that signals must be uniform across datasets, and the ranking above
is not a defect list. It is a per-signal account of where the method's strength
actually sits, produced automatically so that the account is not a matter of memory.
