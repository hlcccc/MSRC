#!/usr/bin/env python
"""Routine audit of the evidence pipeline: is this data authoritative, and is the
setup the mainstream one?

This is a check to run every time evidence is collected, not a one-off. It looks for
the failures that produce a *plausible* number rather than an error:

- **Labelled by a rule nobody can point at.** A benchmark scored by an archived rule
  when an official one exists still produces a number; it is just not the benchmark's.
- **A short file.** A run that stops early leaves a smaller, easier or harder subset
  and every metric downstream silently describes that subset.
- **Image leakage across the split.** One photograph on both sides of the split makes
  the test half partly a memory test. The reported metric goes up and nothing warns.
- **A missing evidence channel.** TextVQA was measured for weeks with `cross_model`
  empty on every item -- the single strongest signal on the one dataset where it was
  checked, absent everywhere it was not.
- **Duplicate items.** The same question counted twice weights it twice.

    python scripts/pipeline_audit.py --cell "LLaVA|POPE=path.jsonl@9000@official" ...

``@expected_n`` and ``@label_rule`` are optional; when given they are checked.

Exit status is non-zero when anything fails, so this can gate a queue.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.groups import image_group_id  # noqa: E402
from evaluation.split import DEFAULT_DEV_FRACTION, DEFAULT_SEED, grouped_split  # noqa: E402
from msrc.signals import Evidence  # noqa: E402

CHANNELS = ("views", "resamples", "verifications", "cross_model", "ocr_texts")

#: Rules that come from the benchmark's own released scorer. A label produced any
#: other way is not automatically wrong, but it is not the benchmark's number either.
OFFICIAL_RULES = {"official", "chairs-official", "pope-official", "m4c-official"}


def parse_spec(spec: str):
    head, raw = spec.split("=", 1)
    model, dataset = head.split("|", 1)
    parts = raw.split("@")
    return model, dataset, Path(parts[0]).expanduser(), parts[1:]


def audit(model, dataset, path, extras):
    problems, notes = [], []
    if not path.is_file():
        return None, [f"{model} | {dataset}: 文件不存在 {path}"], []
    rows = [json.loads(l) for l in
            path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if not rows:
        return None, [f"{model} | {dataset}: 文件为空"], []
    n = len(rows)

    expected_n = int(extras[0]) if extras and extras[0].isdigit() else None
    label_rule = extras[1] if len(extras) > 1 else None

    # -- size against the published split
    if expected_n is not None and n != expected_n:
        problems.append(f"条数 {n} != 官方 {expected_n}（跑早了，或者被截断过）")
    elif expected_n is not None:
        notes.append(f"条数与官方一致（{n}）")

    # -- label provenance
    sources = collections.Counter(
        str(r.get("label_source") or r.get("label_rule") or "") for r in rows)
    unnamed = sources.get("", 0)
    if unnamed:
        problems.append(f"{unnamed} 条没有写明标签来源（label_source / label_rule 为空）")
    else:
        notes.append("每条都写了标签来源：" + "、".join(sorted(sources)))
    if label_rule and label_rule not in OFFICIAL_RULES and label_rule != "any":
        problems.append(f"标签规则 {label_rule!r} 不是基准官方规则")

    # -- duplicates
    ids = [str(r.get("id", "")) for r in rows]
    dup_ids = [k for k, c in collections.Counter(ids).items() if c > 1]
    if dup_ids:
        problems.append(f"重复 id {len(dup_ids)} 个，例如 {dup_ids[:3]}")

    # -- image leakage across the split
    groups = [image_group_id(r) for r in rows]
    dev, test = grouped_split(groups, dev_fraction=DEFAULT_DEV_FRACTION,
                              seed=DEFAULT_SEED)
    dev_images, test_images = set(g for g, m in zip(groups, dev) if m), \
                              set(g for g, m in zip(groups, test) if m)
    overlap = dev_images & test_images
    if overlap:
        problems.append(f"开发/测试半区共用 {len(overlap)} 张图 —— 测试半区部分是背答案")
    else:
        notes.append(f"切分按真实图片身份，两半无共用图（dev {len(dev_images)} / "
                     f"test {len(test_images)} 张）")

    # -- evidence channels
    missing = []
    avail = {}
    for ch in CHANNELS:
        have = sum(1 for r in rows
                   if getattr(Evidence.from_dict(r["evidence"]), ch))
        avail[ch] = have / n
        if have == 0:
            missing.append(ch)
    if missing:
        problems.append("整列缺失的证据通道：" + "、".join(missing)
                        + "（对应的信号在这些条目上全部不可用）")

    labels = [int(r.get("label", -1)) for r in rows]
    if any(l not in (0, 1) for l in labels):
        problems.append("存在不是 0/1 的标签")
    stats = {
        "n": n, "positive_rate": sum(labels) / n,
        "channels": avail,
        "label_sources": dict(sources),
    }
    return stats, problems, notes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", action="append", required=True,
                        metavar="MODEL|DATASET=PATH[@官方条数][@标签规则]")
    args = parser.parse_args(argv)

    print("=" * 100)
    print("数据流程审计：权不权威、公不公平 —— 建议每次采完都跑一次")
    print("=" * 100)
    failed = 0
    for spec in args.cell:
        model, dataset, path, extras = parse_spec(spec)
        stats, problems, notes = audit(model, dataset, path, extras)
        print()
        print(f"### {model} | {dataset}")
        if stats is None:
            print("   ✗ " + "; ".join(problems))
            failed += 1
            continue
        print("   n=%d  正类率 %.1f%%" % (stats["n"], 100 * stats["positive_rate"]))
        print("   证据通道可用率：" + "  ".join(
            "%s=%.0f%%" % (k, 100 * v) for k, v in stats["channels"].items()))
        for x in notes:
            print("   ✓ " + x)
        for x in problems:
            print("   ✗ " + x)
        failed += len(problems)
    print()
    print("=" * 100)
    if failed:
        print(f"发现 {failed} 处问题。这些问题不会让程序报错，只会让数字看起来是对的。")
    else:
        print("没有发现问题。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
