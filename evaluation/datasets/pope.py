"""POPE -> MSRC input records.

POPE (Li et al., 2023) asks a yes/no question about an object in a COCO image --
"Is there a snowboard in the image?" -- where half the questions name something
that is there and half name something that is not. It is the standard benchmark for
**object hallucination** in vision-language models, and it is used here for a reason
that matters more than its fame: correctness is unambiguous.

TextVQA's answers have to be matched against ten annotators' phrasings, so a
model that answers "6th" for a reference of "sixth" is scored wrong for a reason
that has nothing to do with hallucination. On POPE the reference is one word, "yes"
or "no", and the model's answer is parsed the same way the benchmark parses it. The
label is then the thing the framework is supposed to predict, with no matching noise
on top.

Three splits ship, and they are not interchangeable:

| split | negative examples are | why it matters |
|---|---|---|
| `random` | random absent objects | the easy one |
| `popular` | the most frequent objects | checks a frequency bias |
| `adversarial` | objects chosen to co-occur with what is present | the hard one |

A number from `random` alone says very little, because a model that is biased toward
"yes" scores well on it. ``adversarial`` is the split worth quoting.

What the adapter does *not* do
------------------------------

It does not label anything. The label is "did the model answer correctly", which
depends on the model's response, so it cannot be known before the response exists.
Records carry the reference answer and ``collect_evidence.py`` labels them with
``--label-rule yes_no`` after generating.

Record shape produced
---------------------

    {"id": ..., "question": ..., "image": ..., "dataset": "POPE",
     "split": ..., "gold_answers": ["yes"], "answer_format": "yes_no"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evaluation.datasets.images import write_image

__all__ = ["SPLITS", "build_records", "parse_yes_no", "parse_yes_no_official", "write_jsonl"]

#: The three released splits, easiest first.
SPLITS = ("random", "popular", "adversarial")

#: Column names the release has used. The HF release stores the reference in
#: ``answer``; older JSON releases used ``label``. Several are tried rather than
#: one assumed.
REFERENCE_KEYS = ("answer", "label", "gt_answer")
QUESTION_KEYS = ("question", "text", "prompt")
IMAGE_KEYS = ("image", "img", "picture")


def _first_present(row: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def parse_yes_no(response: str) -> Optional[str]:
    """``"yes"``, ``"no"``, or ``None`` if the response says neither.

    Legacy MSRC parsing, NOT the released POPE scorer: the **first** yes/no decides,
    case-insensitively. "Yes, there is a snowboard" is a yes; "No, there is no
    snowboard" is a no -- the second word there is also "no", so a rule that
    scanned for either word anywhere would get the first case right and the second
    one backwards depending on which it looked for first. Taking the first
    occurrence is what makes both work.

    "I cannot tell" contains neither and returns ``None`` rather than a guess,
    because a guess here becomes a fabricated label.
    """
    import re

    text = str(response or "").lower()
    match = re.search(r"\b(yes|no)\b", text)
    return match.group(1) if match else None


def parse_yes_no_official(response: str) -> str:
    """Reproduce RUCAIBox/POPE evaluate.py at commit 08d957b917e5.

    Only the first sentence is inspected. After commas are removed, any exact
    ``No``, ``no`` or ``not`` token means no; otherwise the scorer returns yes.
    Case and the empty-response default are deliberate compatibility details.
    """
    first_sentence = str(response or "").split(".", 1)[0].replace(",", "")
    words = first_sentence.split(" ")
    return "no" if any(token in words for token in ("No", "no", "not")) else "yes"


def build_records(
    root: str | Path,
    out_dir: str | Path,
    *,
    splits: Iterable[str] = ("adversarial",),
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read the POPE parquet release and return MSRC input records.

    Images are written under ``out_dir/images/`` so the collector can read them by
    path. The HF release embeds the JPEG bytes in the parquet, so no separate COCO
    download is needed.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")

    wanted = [s for s in splits]
    unknown = sorted(set(wanted) - set(SPLITS))
    if unknown:
        raise ValueError(
            f"unknown POPE split(s) {unknown}; the release ships {list(SPLITS)}"
        )

    parquets = sorted(root.glob("**/*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"no parquet files under {root}. The HF release keeps them in Full/, "
            "one file per split."
        )

    # Everything above is argument checking and directory listing, and it happens
    # before pandas is imported on purpose: a misspelled split name or a wrong path
    # should not fail with "install pandas" when the real answer is that the path
    # is wrong. The optional dependency is only needed to read the files.
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reading POPE needs pandas:\n  pip install pandas"
        ) from exc

    # pandas reads parquet only through an engine it does not install; without
    # this the failure surfaces deep inside pandas and never says what was being
    # read.
    try:
        pd.read_parquet(parquets[0], columns=[])
    except ImportError as exc:
        raise ImportError(
            "reading POPE needs a parquet engine, which pandas does not install:\n"
            "  pip install pyarrow\n"
            f"(tried to read {parquets[0]})"
        ) from exc
    except Exception:
        pass

    image_dir = Path(out_dir).expanduser() / "images"
    records: List[Dict[str, Any]] = []

    for parquet in parquets:
        stem = parquet.stem.split("-")[0]  # "random-00000-of-00001" -> "random"
        if stem not in wanted:
            continue
        frame = pd.read_parquet(parquet)
        for index, row in enumerate(frame.to_dict("records")):
            question = _first_present(row, QUESTION_KEYS)
            reference = _first_present(row, REFERENCE_KEYS)
            if not question or reference is None:
                continue
            reference_text = str(reference).strip().lower()
            if reference_text not in ("yes", "no"):
                continue

            source = str(row.get("image_source") or row.get("id") or index)
            digest = source.replace("/", "_")
            image_path = write_image(
                _first_present(row, IMAGE_KEYS),
                image_dir / f"pope_{stem}_{digest}.jpg",
            )
            if image_path is None:
                continue

            records.append({
                "id": f"pope:{stem}:{source}:{parquet.name}:{index}",
                "source_image_id": source,
                "question": str(question),
                "image": image_path,
                "dataset": "POPE",
                "split": stem,
                "gold_answers": [reference_text],
                "answer_format": "yes_no",
                "answer_parser": "pope_official",
            })
            if limit is not None and len(records) >= limit:
                return records
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
