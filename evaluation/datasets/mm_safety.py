"""MM-SafetyBench -> MSRC input records.

MM-SafetyBench (Liu et al., 2023) pairs harmful instructions with images designed
to make a model comply -- typography overlays, and images generated to match the
request. It is the standard multimodal safety benchmark, which is why it is the
one wired up here.

What the adapter does *not* do
------------------------------

It does not label anything. The label for the safety family is "did the model
refuse", which depends on the model's response, so it cannot be known before the
response exists. The adapter emits ``gold_answers``-less records and
``collect_evidence.py`` labels them with :mod:`msrc.judge` after generating.

That ordering matters: labelling before generating would mean importing a
published set of labels produced by a *different* model, and a refusal benchmark is
only meaningful against the model actually under test.

Record shape produced
---------------------

    {"id": ..., "question": ..., "image": ..., "dataset": "MM-SafetyBench",
     "category": ..., "attack": ...}

which is the input shape `collect_evidence.py` accepts when no answer is supplied.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "build_records",
    "stratified_sample",
    "ATTACK_KINDS",
    "write_jsonl",
    "PARQUET_ENGINES",
]

#: pandas reads parquet only through one of these, and installs neither.
PARQUET_ENGINES = ("pyarrow", "fastparquet")

#: The image variants MM-SafetyBench ships per category. ``Text_only`` carries no
#: image, so it is kept as a separate option rather than silently mixed in -- a
#: deployment reporting a multimodal number should not have text-only rows in it.
ATTACK_KINDS = ("SD", "SD_TYPO", "TYPO", "Text_only")

#: Column names the release has used for the instruction and the image. Real
#: releases are not consistent, so several are tried rather than one assumed.
QUESTION_KEYS = ("question", "prompt", "instruction", "query", "text")
IMAGE_KEYS = ("image", "img", "image_bytes", "picture")


def _first_present(row: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _write_image(payload: Any, destination: Path) -> Optional[str]:
    """Persist one image from whatever the parquet stored."""
    if payload is None:
        return None
    if isinstance(payload, dict) and "bytes" in payload:
        payload = payload["bytes"]
    if isinstance(payload, (bytes, bytearray)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(bytes(payload))
        return str(destination)
    if isinstance(payload, str) and payload:
        # Some releases store a path or a base64 blob rather than raw bytes.
        candidate = Path(payload)
        if candidate.is_file():
            return str(candidate)
        import base64

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(base64.b64decode(payload))
            return str(destination)
        except Exception:
            return None
    return None


def build_records(
    root: str | Path,
    out_dir: str | Path,
    *,
    attacks: Iterable[str] = ("TYPO",),
    categories: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read the parquet release and return MSRC input records.

    Images are written under ``out_dir/images/`` so the collector can read them by
    path. Text-only rows are skipped unless ``"Text_only"`` is explicitly asked
    for, and when they are, ``image`` is left empty and the grounding signal will
    report itself unavailable for them -- which is the honest outcome rather than
    a fabricated OCR reading.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reading MM-SafetyBench needs pandas:\n  pip install pandas"
        ) from exc

    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")

    data_dir = root / "data" if (root / "data").is_dir() else root
    parquets = sorted(data_dir.glob("*/*.parquet"))
    if parquets:
        # pandas reads parquet only through an engine, and pandas does not install
        # one. Without this check the failure surfaces deep inside pandas as a
        # message about pyarrow that never mentions what was actually being done.
        try:
            pd.read_parquet(parquets[0], columns=[])
        except ImportError as exc:
            raise ImportError(
                "reading MM-SafetyBench needs a parquet engine, which pandas does "
                "not install:\n  pip install pyarrow\n"
                f"(tried to read {parquets[0]})"
            ) from exc
        except Exception:
            # Anything else is the file's problem, not the engine's; let the real
            # read below report it in context.
            pass

    image_dir = Path(out_dir).expanduser() / "images"

    records: List[Dict[str, Any]] = []
    for parquet in sorted(data_dir.glob("*/*.parquet")):
        category = parquet.parent.name
        attack = parquet.stem
        if attack not in attacks:
            continue
        if categories is not None and category not in set(categories):
            continue

        frame = pd.read_parquet(parquet)
        for index, row in frame.iterrows():
            payload = row.to_dict()
            question = _first_present(payload, QUESTION_KEYS)
            if not question:
                continue

            image_path = None
            if attack != "Text_only":
                raw = _first_present(payload, IMAGE_KEYS)
                digest = hashlib.sha1(
                    f"{category}|{attack}|{index}|{str(question)[:64]}".encode("utf-8")
                ).hexdigest()[:16]
                image_path = _write_image(raw, image_dir / f"{category}_{attack}_{digest}.png")
                if image_path is None:
                    continue

            records.append(
                {
                    "id": f"mmsafety:{category}:{attack}:{index}",
                    "question": str(question),
                    "image": image_path or "",
                    "dataset": "MM-SafetyBench",
                    "category": category,
                    "attack": attack,
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def stratified_sample(
    records: Iterable[Dict[str, Any]],
    *,
    per_category: int = 20,
    seed: int = 20260920,
) -> List[Dict[str, Any]]:
    """Up to ``per_category`` records from each category, as a balanced set.

    A category-balanced sample rather than the first N records: taking the head of
    the file would put the alphabetically early categories into the run and leave
    the late ones out, and the result would then describe those categories rather
    than the benchmark.

    The RNG is seeded **per category**, not once for the whole loop. Seeding once
    makes every category's draw depend on how many categories came before it, so a
    release that gains a category silently re-draws the sample for all the others --
    and evidence already collected for the old sample is thrown away. With
    per-category seeding, adding a category only adds its own items.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record.get("category", "")), []).append(record)

    sample: List[Dict[str, Any]] = []
    for category in sorted(buckets):
        pool = buckets[category]
        take = min(int(per_category), len(pool))
        rng = random.Random(f"{seed}:{category}")
        sample.extend(rng.sample(pool, take))
    # Shuffled so a run is not ordered by category, which would make the first
    # hour of collection cover only the first few categories.
    random.Random(seed).shuffle(sample)
    return sample


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
