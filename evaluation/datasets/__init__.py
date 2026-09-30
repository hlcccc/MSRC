"""Dataset adapters.

Each one turns a public benchmark into the JSONL shape
:mod:`scripts.collect_evidence` reads, and each one records the label rule it
applied so a result can be traced back to it.

| adapter | family | label |
|---|---|---|
| :mod:`evaluation.datasets.mm_safety` | safety | refusal vs compliance, by :mod:`msrc.judge` |
| :mod:`evaluation.datasets.pope` | factual | the model's yes/no answer against the reference, by `parse_yes_no` |

Importing this package does not require pandas; the adapters that need it import it
themselves, after their path and argument checks, so a wrong root reports the root
rather than a missing optional dependency. Nothing here imports torch at all.
"""

from evaluation.datasets.mm_safety import build_records as build_mm_safety
from evaluation.datasets.pope import build_records as build_pope
from evaluation.datasets.pope import parse_yes_no, write_jsonl

__all__ = ["build_mm_safety", "build_pope", "parse_yes_no", "write_jsonl"]
