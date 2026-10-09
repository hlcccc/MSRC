"""Device utilities shared by the collection scripts.

`cap_gpu_memory` started life inside ``scripts/collect_evidence.py``. When the guard
runner needed it too, importing it from there meant importing the whole collector --
its provider stack, its dataset adapters, its argument parser -- to get fifteen
lines. Worse, it only worked because ``scripts`` happens to be a namespace package,
so the dependency would break silently the first time the scripts were laid out
differently.

It belongs here: it is about the device, not about collecting evidence.
"""

from __future__ import annotations

__all__ = ["cap_gpu_memory"]


def cap_gpu_memory(fraction: float, *devices: str) -> None:
    """Cap this process on each device it will use.

    On a GPU that several people share, an uncapped job does not fail by itself
    when it over-allocates -- it takes the memory somebody else's job is holding,
    and the somebody else is the one whose run dies. Capping inverts that: our
    allocation raises, our run stops, and the other job is untouched. It is the
    difference between a wasted afternoon and somebody's ruined week, so it is
    applied before a single weight is loaded.

    ``fraction`` of 0 means no cap and is the default, so nothing changes for a run
    that owns its device.
    """
    if fraction < 0 or fraction > 1:
        raise SystemExit("--gpu-memory-fraction must be 0 (no cap) or within (0, 1]")
    if fraction == 0:
        return

    import torch

    for device in devices:
        name = str(device or "")
        if not name.startswith("cuda"):
            continue
        index = int(name.split(":", 1)[1]) if ":" in name else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        total = torch.cuda.get_device_properties(index).total_memory
        print(
            f"[collect] device cuda:{index} capped at {fraction:.0%} of "
            f"{total / 2**30:.1f} GiB = {fraction * total / 2**30:.1f} GiB"
        )
