"""Dataset adapters.

Each one turns a public benchmark into the JSONL shape
:mod:`scripts.collect_evidence` reads, and each one records the label rule it
applied so a result can be traced back to it.

| adapter | benchmark | label |
|---|---|---|
| :mod:`evaluation.datasets.pope` | POPE (factual hallucination) | the model's yes/no answer against the reference, by `parse_yes_no` |

Importing this package does not require pandas; the adapters that need it import it
themselves, after their path and argument checks, so a wrong root reports the root
rather than a missing optional dependency. Nothing here imports torch at all.
"""

from evaluation.datasets.pope import build_records as build_pope
from evaluation.datasets.pope import parse_yes_no, write_jsonl

__all__ = ["build_pope", "parse_yes_no", "write_jsonl"]
