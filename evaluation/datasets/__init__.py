"""Dataset adapters.

Each one turns a public benchmark into the JSONL shape
:mod:`scripts.collect_evidence` reads, and each one records the label rule it
applied so a result can be traced back to it.

| adapter | family | label |
|---|---|---|
| :mod:`evaluation.datasets.mm_safety` | safety | refusal vs compliance, by :mod:`msrc.judge` |

Importing this package does not require pandas; the adapters that need it import
it themselves so the decision layer stays NumPy-only.
"""

from evaluation.datasets.mm_safety import build_records as build_mm_safety

__all__ = ["build_mm_safety"]
