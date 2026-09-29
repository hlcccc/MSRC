#!/usr/bin/env python
"""Collect MSRC evidence for a labelled dataset and cache it as JSONL.

Evidence is the expensive part: every item costs four re-phrasings, K resamples
and three verification probes, all of which need the model. Collecting it once and
caching it is what makes the evaluation re-runnable -- fitting, threshold
selection, ablations and the report all read the same readings, on a CPU, without
touching a GPU again.

It is also what makes the readings auditable. A reviewer can open the JSONL and
see exactly what each signal was computed from, rather than trusting a summary.

Input: JSONL, one object per line, with at least

    {"question": ..., "answer": ..., "image": ..., "label": 0|1}

Output: the same records with an ``evidence`` field added.

    python scripts/collect_evidence.py --data chair.jsonl --out evidence.jsonl \\
        --model-path /path/to/llava --k 3

Already-collected items are skipped on re-run, so an interrupted job resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msrc.provider import PromptSet, gather_evidence  # noqa: E402
from msrc.signals import Evidence  # noqa: E402
from msrc.types import RISK_FACTUAL  # noqa: E402


def load_jsonl(path: Path) -> list:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno} is not valid JSON: {exc.msg}") from exc
    if not rows:
        raise SystemExit(f"{path} holds no records")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="input JSONL with labels")
    parser.add_argument("--out", required=True, help="output JSONL with evidence")
    parser.add_argument("--model-path", required=True, help="HF checkpoint directory")
    parser.add_argument("--k", type=int, default=3, help="resamples per item")
    parser.add_argument("--limit", type=int, default=0, help="stop after N items (0 = all)")
    parser.add_argument("--risk-family", default=RISK_FACTUAL)
    parser.add_argument("--no-ocr", action="store_true", help="skip the OCR channel")
    parser.add_argument(
        "--want-attention",
        action="store_true",
        help="compute the visual-attention signal; keeps every layer's attention "
             "alive and is the most expensive reading here",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    from msrc.providers import HFLLaVAProvider, HFOCRProvider

    source = Path(args.data).expanduser()
    if not source.is_file():
        raise SystemExit(f"input not found: {args.data}")
    records = load_jsonl(source)
    if args.limit:
        records = records[: args.limit]

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if out_path.exists():
        for row in load_jsonl(out_path):
            if row.get("evidence"):
                done[row.get("id", row.get("image", ""))] = row
        print(f"[collect] resuming: {len(done)} items already have evidence")

    ocr_provider = None if args.no_ocr else HFOCRProvider()
    provider = HFLLaVAProvider(
        model_path=args.model_path,
        ocr_provider=ocr_provider,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        want_attention=args.want_attention,
    )

    print(f"[collect] {len(records)} items, k={args.k}, attention={args.want_attention}")
    print(f"[collect] loading {args.model_path} ...")
    started = time.time()
    provider.load()
    print(f"[collect] loaded in {time.time() - started:.1f}s; calls begin")

    written = 0
    t0 = time.time()
    handle = out_path.open("w", encoding="utf-8")
    try:
        for index, record in enumerate(records):
            key = record.get("id", record.get("image", str(index)))
            if key in done:
                handle.write(json.dumps(done[key], ensure_ascii=False) + "\n")
                written += 1
                continue

            evidence, calls = gather_evidence(
                provider,
                str(record.get("question", "")),
                str(record.get("answer", "")),
                str(record.get("image", "")),
                risk_family=args.risk_family,
                k=args.k,
                prompts=PromptSet(),
            )
            record = dict(record)
            record["id"] = key
            record["evidence"] = evidence.to_dict()
            record["model_calls"] = calls
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written % args.save_every == 0:
                handle.flush()
                rate = (time.time() - t0) / max(written - len(done), 1)
                left = (len(records) - written) * rate
                print(
                    f"[collect] {written}/{len(records)}  "
                    f"{rate:.1f}s/item  eta {left/60:.0f} min  "
                    f"calls={provider.calls}",
                    flush=True,
                )
    finally:
        handle.close()

    print(f"[collect] wrote {written} records -> {out_path}")
    print(f"[collect] total generations: {provider.calls}")
    if provider.want_attention:
        print(f"[collect] of which attention passes: {provider.attention_calls}")

    # A quick sanity read: which signals actually produced a value on this data.
    rows = load_jsonl(out_path)
    from msrc.signals import build_report

    live, dead = [], []
    for row in rows[:200]:
        report = build_report(Evidence.from_dict(row["evidence"]))
        for signal in report.signals:
            (live if signal.available else dead).append(signal.name)
    from collections import Counter

    print("[collect] signals that produced a value (first 200 items):")
    for name, count in Counter(live).most_common():
        print(f"    {name:26} {count}")
    if dead:
        print("[collect] signals that never produced a value:")
        for name, count in Counter(dead).most_common():
            print(f"    {name:26} {count}  <- check the serving stack supplies it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
