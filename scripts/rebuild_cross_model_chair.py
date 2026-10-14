#!/usr/bin/env python3
"""Rebuild the CHAIR cross-model channel under one protocol.

The channel was grown by appending and de-duplicated on the answer *string* alone,
so a cell could end up carrying three readings -- two of them from the same donor
at different generation budgets -- with nothing recording which model either came
from. The three cells therefore did not share a protocol even though the report
compares them side by side.

This rebuilds the channel from scratch: every cell carries exactly the *other two*
models' current primary answers, in a fixed order, each tagged with its provider
and the digest of the file it came from. Two readings per cell in all three cells.

The originals are not touched. Output goes to a new directory with a receipt that
records the inputs, their digests and the row counts.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

RUN = "/mnt/data/HLC/msrc_run"
CROSS = os.path.join(RUN, "crossmodel")
OUT = os.path.join(CROSS, "rebuilt_20261013")

# Order is fixed here and used to order each cell's donors, so the channel is
# reproducible rather than dependent on the order files happened to be passed in.
#
# Target and donor are separate roles and for one model they are not the same file:
# Qwen's labelled cell carries the re-collected answers, but its *primary* readings
# are in the unlabelled re-collection file -- the labelled one was produced by
# appending donors to it earlier, so its `primary` is fine but its file is not the
# one the rebuild should treat as the model's own output.
TARGETS = [
    ("LLaVA-1.5-13B", "cell_LLaVA-1.5-13B_chair_3model.jsonl"),
    ("Qwen2.5-VL-7B", "cell_Qwen2.5-VL-7B_chair_128_3model.jsonl"),
    ("InternVL3-8B", "cell_InternVL3-8B_chair_3model.jsonl"),
]

DONORS = {
    "LLaVA-1.5-13B": "cell_LLaVA-1.5-13B_chair_3model.jsonl",
    "Qwen2.5-VL-7B": "cell_Qwen2.5-VL-7B_chair_128.jsonl",
    "InternVL3-8B": "cell_InternVL3-8B_chair_3model.jsonl",
}

#: Fields a target row must carry for the headline table to read it.
REQUIRED = ("id", "image", "question", "answer", "label")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: str):
    with open(path, encoding="utf-8-sig") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def basename(value) -> str:
    return os.path.basename(str(value or ""))


def main() -> int:
    os.makedirs(OUT, exist_ok=True)

    tables = {}
    digests = {}
    for model, fname in DONORS.items():
        path = os.path.join(CROSS, fname)
        if not os.path.exists(path):
            raise SystemExit("missing donor source: %s" % path)
        rows = load(path)
        digests[model] = {"file": fname, "sha256": sha256(path), "n": len(rows)}
        tables[model] = {r["id"]: r for r in rows}
        print("%-16s donor  %-46s n=%d sha256=%s"
              % (model, fname, len(rows), digests[model]["sha256"][:12]))

    receipt = {"built": "2026-10-13", "protocol": "two donors per cell, provider-tagged",
               "sources": digests, "cells": {}}

    for target, fname in TARGETS:
        tpath = os.path.join(CROSS, fname)
        rows = load(tpath)
        missing = [f for f in REQUIRED if f not in rows[0]]
        if missing:
            raise SystemExit("%s is missing %s -- that is not a labelled cell"
                             % (fname, missing))
        donors = [m for m, _ in TARGETS if m != target]
        out_rows = []
        joined = {d: 0 for d in donors}
        skipped = {d: 0 for d in donors}

        for row in rows:
            item = row["id"]
            evidence = row.get("evidence")
            if not isinstance(evidence, dict):
                raise SystemExit("row %r has no evidence object" % item)
            channel = []
            for donor in donors:
                src = tables[donor].get(item)
                if src is None:
                    skipped[donor] += 1
                    continue
                if basename(src.get("image")) != basename(row.get("image")):
                    skipped[donor] += 1
                    continue
                if str(src.get("question", "")) != str(row.get("question", "")):
                    skipped[donor] += 1
                    continue
                primary = dict(src["evidence"]["primary"])
                # Provenance goes in `meta`, which Evidence serialises -- a bare
                # top-level key would be dropped the first time the file is read
                # and written back, and the whole point of tagging a donor is that
                # the tag survives.
                primary["meta"] = {
                    "provider": donor,
                    "provider_file": digests[donor]["file"],
                    "provider_sha256": digests[donor]["sha256"],
                }
                channel.append(primary)
                joined[donor] += 1

            if len(channel) != len(donors):
                raise SystemExit(
                    "row %r got %d of %d donors; a half-filled channel would look "
                    "like a cell with fewer readings rather than a broken join"
                    % (item, len(channel), len(donors)))
            evidence["cross_model"] = channel
            out_rows.append(row)

        out_name = "cell_%s_chair_2model.jsonl" % target
        out_path = os.path.join(OUT, out_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in out_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        receipt["cells"][target] = {
            "source": fname, "source_sha256": digests[target]["sha256"],
            "out": out_name, "out_sha256": sha256(out_path), "n": len(out_rows),
            "donors": donors, "joined": joined, "skipped": skipped,
        }
        print("%-16s -> %-42s n=%d donors=%s" % (target, out_name, len(out_rows),
                                                 donors))

    with open(os.path.join(OUT, "receipt.json"), "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, ensure_ascii=False, indent=2)
    print("\nreceipt:", os.path.join(OUT, "receipt.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
