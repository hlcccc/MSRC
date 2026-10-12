# How much does the split move the number?

Every per-cell figure in this repository comes from one split: half the images to
choose the threshold, half to report on. That is the right protocol, and it is also
a single draw. This note measures how much the draw matters, because a gap to the
bar is only meaningful next to the noise floor of the quantity being compared.

Method: `scripts/balanced_report.py` re-run with eight split seeds (0–7). Within each
seed the discipline is unchanged — threshold chosen on that seed's dev half, reported
on that seed's test half. **The distribution is reported, not a chosen seed**;
selecting the seed that passes is threshold-tuning wearing a different hat.

## Balanced accuracy across eight splits

| model | dataset | min | max | median | spread | gap to 85% (by median) |
|---|---|---|---|---|---|---|
| LLaVA-1.5-13B | POPE | 79.4% | 82.3% | **80.7%** | 2.8 | **+4.3** |
| LLaVA-1.5-13B | TextVQA | 79.6% | 81.6% | **80.5%** | 2.0 | **+4.5** |
| LLaVA-1.5-13B | HallusionBench | 64.5% | 69.6% | **67.4%** | 5.0 | +17.6 |
| LLaVA-1.5-13B | CHAIR | 73.3% | 76.6% | **75.7%** | 3.4 | +9.3 |
| Qwen2.5-VL-7B | POPE | 75.1% | 77.6% | **76.6%** | 2.5 | +8.4 |
| Qwen2.5-VL-7B | HallusionBench | 60.6% | 65.9% | **63.1%** | 5.3 | +21.9 |
| Qwen2.5-VL-7B | CHAIR | 72.4% | 80.1% | **76.0%** | 7.7 | +9.0 |

AUROC moves too, and by more than is comfortable: LLaVA POPE spans 0.8412–0.8751 and
Qwen CHAIR spans 0.8082–0.8586.

## What this changes

**Quoting a single split overstates precision.** The default seed is not neutral —
it happened to sit 1.1 points above the median on LLaVA POPE and 1.4 above on Qwen
POPE. Gaps to the bar quoted from it were optimistic on those cells and pessimistic
on TextVQA. The medians above are what the numbers are, and the earlier single-split
figures should be read as one draw from those ranges.

**Most cells have a spread comparable to the gap.** A 2.0–3.4 point spread against a
4.3–4.5 point gap means a candidate that claims two or three points is claiming
something the protocol cannot resolve at this sample size. The two cells where the
gap clearly exceeds the spread are HallusionBench and CHAIR, and there the gap is
large enough that the conclusion is not in doubt either way.

## The rule that follows

Any future claimed improvement has to clear the spread for that cell, measured the
same way, or it is not an improvement — it is a different draw. When a change is
assessed, the comparison should be **paired across the same seeds**, not two
single-split numbers subtracted.

## The parsing bug this note started from

The first version of the sweep read column 8 of the printed table and called it
balanced accuracy. Column 7 is balanced accuracy; column 8 is raw accuracy on the
dataset's own proportions. On POPE, where hallucinations are 15% of items, raw
accuracy runs about five points above balanced accuracy — so the bug produced a
cheerful table showing LLaVA POPE at "85.4% mean, half a point from the bar" for a
metric that was not the acceptance metric.

The failure mode is worth naming: the two columns sit next to each other, both are
percentages of similar magnitude, and the wrong one reads as *better* news. The sweep
now parses `--json` and never touches column indices, and the header in the printed
table carries both names so the mistake is visible to a reader too.
