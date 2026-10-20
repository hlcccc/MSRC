# Safety Backend Versions And Rollback

## Keep The Two Methods Separate

| Version | Git reference | Safety method |
|---|---|---|
| Original | `singguard-before-plugguard-20261019` | SingGuard request screening before base generation |
| New | `codex/plugguard-acceptance-20261019` | PlugGuard adaptation: causal scoring and output gating during generation |

The original GitHub baseline is commit
`e0f712f55da1baf8b3c39c9dd65c7edc4a0aadc5`. The tag preserves that commit.
The already-completed server experiment has its own separately retained exact
source snapshot, configuration, and results. That snapshot takes precedence when
reproducing the historical experiment; the GitHub baseline tag is not a substitute
for its machine-specific model/data package.

Do not delete the original project or its artifacts. Do not switch code beneath
an active job. Each backend uses its own checkout, run IDs, trained state, and
report directory. No automatic fallback exists in the new step 04.

## Return To SingGuard

Create a separate checkout rather than overwriting the PlugGuard project:

```bash
git clone --branch singguard-before-plugguard-20261019 \
  https://github.com/hlcccc/MSRC.git MSRC-SingGuard
cd MSRC-SingGuard
```

Then use that version's five entry points, verified asset package and environment,
and a new run ID. On the original server, prefer the retained original project
with its matching configuration. Do not copy a report from the other backend.

Rollback means restoring the whole safety path, not just changing a model name:
the entry script, scoring/gating logic, threshold selection, delivery, answer
judgment, and report contract must all belong to SingGuard. In the retained
experiment, its operating point was 0.270; the PlugGuard trial used 0.764.
These thresholds and head parameters are not interchangeable. Select or load
the operating point according to the chosen version's development-only protocol.
The eight factual signals and their calibration are independent of this choice.

## Evidence Boundaries

The 2026-10-20 [split-v2 correction](SAFETY_SPLIT_V2.md) additionally groups
cross-ID identical image/question requests. It requires a new head fit and
threshold selection. The measurements below predate that correction; they must
not be relabeled as results of a v2 full run. Original rollback references and
historical experiment artifacts remain unchanged.

The retained SingGuard full trial reported 87.053571% on 3,360 held-out requests.
The separately completed PlugGuard adaptation reported 99.940476%
(TP 2,520 / TN 838 / FP 2 / FN 0) on the same held-out request identities.
Both primary metrics use original request-risk labels, not answer-judge labels.
Do not average or splice these two experiments.

The new script integration was also checked with 450 CPU tests (three optional
dependency skips), 39 focused tests on the server, and a new four-case GPU smoke
run. All four smoke cases matched the prior PlugGuard trial's warnings, answers,
and risk scores. These are integration checks, not another full 3,360-case run.
The merged GitHub publication passed 464 CPU tests with three optional-dependency
skips. That test suite is separate from the GPU experiment results above.

Git stores source, tests, documentation, and paired base-model fingerprints.
It does not store model weights, the trained head, custom benign images, raw
answers, or previous per-sample test scores. Keep those in the separately
verified delivery package. See the [PlugGuard flow](PLUGGUARD_TEST_FLOW.md).
