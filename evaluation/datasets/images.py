"""Pulling an image out of a parquet cell.

Releases store images in more than one way -- raw bytes, a ``{"bytes": ...}``
mapping (the HuggingFace datasets convention), a base64 string, or a path -- and the
POPE adapter has to cope with all of them. Kept as its own module so the reading of
those conventions sits in one place rather than inline in the adapter.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

__all__ = ["write_image"]


def write_image(payload: Any, destination: Path) -> Optional[str]:
    """Persist one image from whatever a parquet stored, or ``None`` if unusable.

    ``None`` rather than a placeholder path: a caller that cannot get an image
    should skip the record, and a made-up path would turn a missing image into a
    read error much further downstream.
    """
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
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(base64.b64decode(payload))
            return str(destination)
        except Exception:
            return None
    return None
