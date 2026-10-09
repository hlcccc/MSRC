"""Stable image identities for held-out evaluation, independent of local names."""

from __future__ import annotations

import re
from typing import Any, Mapping

_COCO_ID = re.compile(r"COCO_(?:train|val)\d{4}_\d+", re.IGNORECASE)


def image_group_id(record: Mapping[str, Any]) -> str:
    """Keep copies of one COCO image together across all POPE settings.

    The adapter writes ``pope_random_COCO_...`` and ``pope_popular_COCO_...``
    to different paths. Those are not independent images. Explicit source IDs
    also cover datasets whose filenames do not encode their image identity.
    """
    explicit = record.get("source_image_id") or record.get("image_id")
    candidates = [explicit, record.get("image_source"), record.get("image"),
                  record.get("id")]
    is_pope = str(record.get("dataset", "")).lower() == "pope"
    is_pope = is_pope or str(record.get("id", "")).lower().startswith("pope:")
    if is_pope:
        for value in candidates:
            match = _COCO_ID.search(str(value or ""))
            if match:
                return "coco:" + match.group(0).lower()
    if explicit is not None and str(explicit).strip():
        return f"{str(record.get('dataset', 'image')).lower()}:{explicit}"
    fallback = record.get("image") or record.get("id")
    if not fallback:
        raise ValueError("record has no image identity; held-out split is undefined")
    return str(fallback)
