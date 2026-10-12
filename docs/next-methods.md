# What each cell would need, and which methods can supply it

Written after the acceptance metric was corrected to plain accuracy on the
benchmark's own proportions. The point of the table below is to separate cells that
are a nudge away from cells that need a different approach, **before** spending GPU
time on methods that cannot close the gap.

## The gap, in the units that matter

Accuracy is a thresholded quantity, so the honest unit is AUROC: to raise accuracy
you have to improve the ranking, and the threshold is already chosen on dev to match
the metric.

| cell | accuracy (median, 8 seeds) | never warn | gap to 85% | AUROC now | AUROC needed | verdict |
|---|---|---|---|---|---|---|
| LLaVA POPE | 89.1% | 85.4% | **met** | 0.867 | — | done |
| Qwen POPE | 88.8% | 83.9% | **met** | 0.837 | — | done |
| Qwen CHAIR | 84.6% | 83.7% | −0.4 | 0.82 | ~0.83 | **a nudge** |
| LLaVA TextVQA | 80.7% | 52.0% | −4.3 | 0.880 | ~0.93 | stretch |
| LLaVA CHAIR | 76.1% | 57.2% | −8.9 | 0.82 | ~0.90 | hard |
| Qwen HallusionBench | 69.0% | 63.2% | −16.0 | 0.69 | ~0.93 | different approach |
| LLaVA HallusionBench | 66.5% | 43.5% | −18.5 | 0.685 | ~0.94 | different approach |

The never-warn column is not decoration. On LLaVA HallusionBench more than half the
items are hallucinated, so 85% accuracy means beating a coin flip by 35 points; on
POPE it means beating "never warn" by nothing at all, which the current numbers
already do by 3.7 points. **A single 85% bar means very different things on these
two cells**, and that is worth saying out loud before more work goes into either.

Incremental signal engineering has delivered roughly +0.02 AUROC per new reading
(the object-level signal +0.19 on CHAIR, the first-step margin +0.023 on POPE). At
that rate HallusionBench needs several such readings, not one.

## Methods worth trying, in order of expected value per GPU hour

### 1. Image-contrast evidence (VCD, CVPR 2024)

[Visual Contrastive Decoding](https://www.openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html)
contrasts the next-token distribution **with** the image against the same
distribution **without** it. The insight transfers directly from decoding to
detection: if removing the image barely changes what the model was about to say, the
answer came from the language prior rather than from the picture, and that is a
hallucination risk.

It targets exactly the failure mode of the chart benchmark. A model answering a
question about a bar chart has a strong prior about what such a chart usually says;
a model that never looked will answer from that prior and barely move when the image
goes away.

**Signal.** Divergence between the two first-step distributions — a forward pass with
the image and one with it replaced by a blank of the same size.
**Cost.** One extra forward per item, no generation, no weights.
**Why first.** Cheapest of the three, published at a top venue, and aimed at the two
cells that are furthest out.

### 2. Semantic entropy with entailment clustering (Nature 2024)

`resample_consistency` currently clusters samples by normalised string equality. The
published method ([Farquhar et al.](https://aclanthology.org/2025.emnlp-main.1239.pdf))
clusters by **bidirectional entailment**, which is what makes it work on free text:
"a white toilet" and "the toilet is white" are one cluster, not two disagreements.
Our version is a strictly weaker proxy and it shows — on CHAIR it reads 0.5166.

**Cost.** An NLI model plus a pass over existing samples. No new generations.
**Why second.** It upgrades a signal that is already collected and currently weak,
and it applies to every caption-style cell.

### 3. Per-head lookback ratio (Lookback Lens, EMNLP 2024)

[Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/) detects hallucination
from the ratio of attention on newly generated tokens to attention on the context,
**per head**. We collect attention already but reduce it to a single mean mass, which
is dead everywhere in this project (AUROC 0.46–0.57 across every cell).

That the scalar is dead is consistent with the paper's own finding: the signal lives
in the spread across heads, and averaging over heads is exactly the operation that
destroys it.

**Cost.** Re-collection, since per-head statistics were never stored.
**Why third.** Highest re-collection cost, but it is the only one of the three that
attacks the internal side, and the current scalar's failure is evidence that we are
throwing the signal away rather than that it is absent.

## What is not on this list, and why

- **Anything requiring test-set choices.** No threshold, split, sample or signal
  combination gets picked on the test half. This holds regardless of how far a cell is
  from the bar.
- **Retraining or fine-tuning the evaluated models.** It changes the thing being
  measured, so the comparison stops meaning anything.
- **Reported SOTA numbers from other papers.** They use different splits, prompts and
  label rules; placing one next to ours would be comparing two different experiments.

## The honest summary

Three cells are within reach of incremental work: Qwen CHAIR is 0.4 points away and
the first-step margin already has a measured, bootstrap-significant gain queued for
full scale. TextVQA and LLaVA CHAIR are stretches. **The two HallusionBench cells are
not reachable by adding signals at the rate signals have been landing** — they need
one of the methods above to work substantially better than expected, or a different
formulation of the task, and saying so now is cheaper than discovering it after
another day of GPU time.
