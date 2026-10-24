#!/usr/bin/env python
"""Collect MSRC evidence for a labelled dataset and cache it as JSONL.

Evidence is the expensive part: every item costs four re-phrasings, K resamples
and three verification probes, all of which need the model. Collecting it once and
caching it is what makes the evaluation re-runnable -- fitting, threshold
selection, ablations and the report all read the same readings, on a CPU, without
touching a GPU again.

It is also what makes the readings auditable. A reviewer can open the JSONL and
see exactly what each signal was computed from, rather than trusting a summary.

Two input shapes
----------------

**With an answer** -- the model's output already exists (a published run, a
platform's production log). The three internal signals cannot be computed for it,
because the logprobs of *that* generation were never recorded and re-running the
model produces a different one. They report themselves unavailable.

    {"question": ..., "answer": ..., "image": ..., "label": 0|1}

**Without an answer** -- supply the accepted answers instead, and this script
generates the answer under evaluation, captures its internals, and labels it:

    {"question": ..., "image": ..., "gold_answers": ["...", "..."]}

This is the shape that lets every signal run, and it is the honest one: the
answer is produced by the model under test in a call whose internals are recorded
as they happen.

The label rule is the VQA agreement rule: the answer is compared against the
accepted answers, and one that disagrees with them is the risky outcome.

Output: the same records with ``answer``, ``label`` and ``evidence`` filled in.

    python scripts/collect_evidence.py --data chair.jsonl --out evidence.jsonl \\
        --model-path /path/to/llava --k 3

Already-collected items are skipped on re-run, so an interrupted job resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msrc.gpu import cap_gpu_memory  # noqa: E402,F401  (re-exported: see msrc/gpu.py)
from msrc.provider import PromptSet, gather_evidence  # noqa: E402
from msrc.signals import Evidence, normalize_text  # noqa: E402

#: How many accepted answers must agree before a generated answer counts as
#: correct. The VQA convention divides by three, so three annotators agreeing is
#: full credit; see examples/make_dev_set.py for the same rule in the development
#: set builder.
VQA_AGREEMENT_DIVISOR = 3


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


def vqa_label(generated: str, gold_answers) -> int:
    """1 when the answer is wrong, by the standard VQA agreement rule.

    The rule divides matches by three, so half credit -- the threshold used here --
    needs at least two accepted answers to agree. That assumes the usual ten
    annotators. Supply a single reference answer and a correct response scores
    0.33, which is labelled wrong: the assumption is stated in the summary rather
    than left for someone to notice in the label distribution.
    """
    predicted = normalize_text(generated)
    if not predicted:
        return 1
    matches = sum(1 for g in gold_answers if normalize_text(g) == predicted)
    score = min(matches / VQA_AGREEMENT_DIVISOR, 1.0)
    return 0 if score >= 0.5 else 1


def yes_no_label(generated: str, reference: str, *, parser: str = "legacy"):
    """Label a yes/no answer, and say whether it could be parsed at all.

    Returns ``(label, parsed)``. The label is 1 when the answer is wrong, which is
    the risky outcome: the model asserted something about the image that is not
    true.

    A response that says neither yes nor no is labelled wrong and reported as
    unparsed rather than guessed at. On a yes/no benchmark an evasive answer is not
    a correct one, and inventing a verdict for it would put a fabricated label into
    the fit.
    """
    from evaluation.datasets.pope import parse_yes_no, parse_yes_no_official

    if parser == "pope_official":
        return int(parse_yes_no_official(generated) != str(reference).strip().lower()), True
    if parser != "legacy":
        raise ValueError(f"unknown yes/no answer parser: {parser!r}")

    verdict = parse_yes_no(generated)
    if verdict is None:
        return 1, False
    return (0 if verdict == str(reference or "").strip().lower() else 1), True


def collection_fingerprint(args) -> dict:
    """Everything about the run that changes what a reading *means*.

    Resume matches on the item key alone. Changing a generation setting therefore
    kept the rows already collected and appended new ones produced under different
    settings, in one file, with nothing recording that it happened -- which is how
    a cell ended up with 460 answers capped at 64 tokens and 740 at 128. Anything
    listed here changes the readings, so a resume that changes one is refused.
    Deliberately excluded: input and output paths, batch size, thread counts --
    those change how the run is executed, not what a reading means.
    """
    keys = (
        "provider", "model_path", "max_new_tokens", "k", "seed", "dtype",
        "want_attention", "want_image_contrast", "max_pixels",
        "second_provider", "second_model_path", "no_ocr",
        "answer_parser", "answer_format",
    )
    out = {}
    for key in keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        out[key] = value if isinstance(value, (str, int, float, bool)) or value is None \
            else str(value)
    return out


def check_resume_manifest(out_path: Path, fingerprint: dict) -> None:
    """Refuse to resume into an output collected under different settings.

    Without this, changing ``--max-new-tokens`` (or the model path, or ``k``) keeps
    the rows already collected and appends new ones produced under the new setting,
    in one file, with nothing recording that it happened. That is how a cell ended
    up holding 460 answers capped at 64 tokens and 740 at 128.

    Writes the manifest on the way through, so a run that is interrupted still
    leaves a record of the settings it was using.
    """
    manifest_path = out_path.with_name(out_path.name + ".manifest.json")
    if out_path.exists() and manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = recorded.get("fingerprint") or {}
        if previous != fingerprint:
            diff = {k: [previous.get(k), fingerprint.get(k)]
                    for k in set(previous) | set(fingerprint)
                    if previous.get(k) != fingerprint.get(k)}
            raise SystemExit(
                "this output was collected under a different configuration; resuming "
                "would mix protocols in one file.\n  differing settings [was, now]: "
                + json.dumps(diff, ensure_ascii=False, indent=2)
                + "\n  Collect to a new --out, or delete the output to re-collect it."
            )
    elif out_path.exists():
        # Stop at the first row carrying evidence: the file is read again below to
        # build the resume index, and a 24 MB cache is not worth parsing twice.
        with out_path.open(encoding="utf-8-sig") as probe:
            for line in probe:
                if line.strip() and json.loads(line).get("evidence"):
                    print(
                        "[collect] WARNING: this output predates configuration "
                        "manifests, so the settings it was collected under cannot be "
                        "verified. Resuming into it can mix generation budgets "
                        "without saying so."
                    )
                    break
    manifest_path.write_text(
        json.dumps({"fingerprint": fingerprint, "out": str(out_path)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_provider(args):
    """Build the model provider the collection loop drives.

    Split out from :func:`main` as a seam. The generation path is where the label
    rule is applied and where the internals are captured, and neither of those
    decisions needs a GPU to test -- but while they were welded to the model
    constructor, neither of them was tested at all.
    """
    from msrc.providers import (
        HFInternVLProvider,
        HFLLaVAProvider,
        HFOCRProvider,
        HFQwenVLProvider,
    )

    providers = {"llava": HFLLaVAProvider, "qwen": HFQwenVLProvider,
                 "internvl": HFInternVLProvider}
    if args.provider not in providers:
        raise SystemExit(
            f"unknown --provider {args.provider!r}; choose from "
            f"{sorted(providers)}. Both load a HuggingFace checkpoint through "
            "transformers and produce the same signals."
        )

    ocr_provider = None if args.no_ocr else HFOCRProvider()
    # The dtype is left to the family default unless it is overridden. Both
    # defaults (float16 for LLaVA, bfloat16 for Qwen2.5-VL) are GPU precisions:
    # fp16 on CPU is either refused outright or slower than fp32, so a CPU run
    # has to be able to say so explicitly.
    dtype = getattr(args, "dtype", "")
    extra = {"dtype": dtype} if dtype else {}
    return providers[args.provider](
        model_path=args.model_path,
        ocr_provider=ocr_provider,
        device=getattr(args, "device", "cuda"),
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        want_attention=args.want_attention,
        want_image_contrast=getattr(args, "want_image_contrast", False),
        max_pixels=getattr(args, "max_pixels", 0) or None,
        **extra,
    )


def build_second_provider(args):
    """The second model, or ``None`` when none was asked for.

    Kept separate from :func:`build_provider` because it is optional and the two
    have different defaults: the second model only ever answers the question once,
    so it needs no attention and no OCR, and asking for attention on it would
    multiply the cost of the whole run for a signal that is not being read.
    """
    if not getattr(args, "second_model_path", ""):
        return None

    from msrc.providers import (
        HFInternVLProvider,
        HFLLaVAProvider,
        HFQwenVLProvider,
    )

    providers = {"llava": HFLLaVAProvider, "qwen": HFQwenVLProvider,
                 "internvl": HFInternVLProvider}
    if args.second_provider not in providers:
        raise SystemExit(
            f"unknown --second-provider {args.second_provider!r}; choose from "
            f"{sorted(providers)}"
        )
    return providers[args.second_provider](
        model_path=args.second_model_path,
        ocr_provider=None,
        device=getattr(args, "second_device", "cuda"),
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        want_attention=False,
        max_pixels=getattr(args, "max_pixels", 0) or None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="input JSONL with labels")
    parser.add_argument("--out", required=True, help="output JSONL with evidence")
    parser.add_argument("--model-path", required=True, help="HF checkpoint directory")
    parser.add_argument(
        "--provider",
        default="llava",
        choices=("llava", "qwen", "internvl"),
        help="which model family the checkpoint is; both produce the same signals",
    )
    parser.add_argument(
        "--second-model-path",
        default="",
        help="a second, independent model's checkpoint. Supplying one turns on "
             "cross_model_agreement and cross_model_object_support, the two signals a "
             "second model buys",
    )
    parser.add_argument(
        "--second-provider",
        default="qwen",
        choices=("llava", "qwen", "internvl"),
        help="which family --second-model-path is",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device for the primary model, e.g. cuda:1. 'cpu' works and "
             "touches no GPU, but pair it with --dtype float32",
    )
    parser.add_argument(
        "--dtype",
        default="",
        help="torch dtype for the primary model, e.g. float32. Empty means the "
             "family default, which is float16 for LLaVA and bfloat16 for "
             "Qwen2.5-VL -- both GPU precisions that a CPU cannot run usefully",
    )
    parser.add_argument(
        "--second-device",
        default="cuda",
        help="torch device for the second model. Putting the two on different GPUs "
             "is usually cheaper than packing both onto one, and it is what makes a "
             "13B plus a 7B fit on a shared machine",
    )
    parser.add_argument("--k", type=int, default=3, help="resamples per item")
    parser.add_argument("--limit", type=int, default=0, help="stop after N items (0 = all)")
    parser.add_argument("--no-ocr", action="store_true", help="skip the OCR channel")
    parser.add_argument(
        "--want-attention",
        action="store_true",
        help="compute the visual-attention signal; keeps every layer's attention "
             "alive and is the most expensive reading here",
    )
    parser.add_argument(
        "--want-image-contrast",
        action="store_true",
        help="run one extra forward per item with the image replaced by a "
             "blank, and record how far the first-step distribution moves. "
             "Visual Contrastive Decoding (CVPR 2024) read as a detection "
             "signal: near zero means the answer did not depend on the image",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="0 leaves the checkpoint's own image budget alone. Qwen2-VL/Qwen2.5-VL "
             "resize dynamically, and their default budget is large enough that a "
             "24 MP figure becomes 65,536 patches -- a single 126 GiB allocation once "
             "eager attention materialises the vision tower's matrix, which is what "
             "visual_attention needs. 1280*28*28 is the budget Qwen's own cookbook "
             "uses. It only binds on images above it, so COCO-sized inputs are "
             "unaffected. LLaVA ignores it: it crops to a fixed 336.",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.0,
        help="cap this process at a fraction of each device it touches, e.g. 0.25. "
             "0 means no cap. Use it when the GPU is shared: the cap makes this "
             "process raise instead of taking memory another user's job is holding",
    )
    args = parser.parse_args()

    cap_gpu_memory(args.gpu_memory_fraction, args.device, args.second_device)

    source = Path(args.data).expanduser()
    if not source.is_file():
        raise SystemExit(f"input not found: {args.data}")
    records = load_jsonl(source)
    if args.limit:
        records = records[: args.limit]

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_budget = None
    if os.environ.get('MSRC_GPU_SHARE_POLICY'):
        if args.device != 'cuda:0' or args.second_model_path:
            raise ValueError('Shared acceptance collection needs one primary model on cuda:0')
        from pipeline.gpu_sharing import FactualWorkBudget
        gpu_budget = FactualWorkBudget(out_path, args.gpu_memory_fraction)

    # Record what this run's readings will mean, and refuse to resume into a file
    # collected under different settings.
    check_resume_manifest(out_path, collection_fingerprint(args))

    # The resume key has to identify the *item*, not the image. POPE names a record
    # `pope:{split}:{image_id}`, and one image is polled about several objects, so
    # the id repeats within a single file: a 1,500-item POPE set has 978 distinct
    # ids. Keyed on the id alone, a re-run collapses every earlier record sharing
    # that id into the last one -- the row count still comes out right and the file
    # silently loses items. The question is what makes a record unique here.
    def resume_key(row) -> tuple:
        return (row.get("id", row.get("image", "")),
                row.get("image", ""),
                row.get("question", ""))

    keys = [resume_key(row) for row in records]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate input item key (id, image, question); refusing collection")

    done = {}
    if out_path.exists():
        for row in load_jsonl(out_path):
            if row.get("evidence"):
                key = resume_key(row)
                if key in done:
                    raise SystemExit("duplicate cached item key; refusing ambiguous resume")
                done[key] = row
        print(f"[collect] resuming: {len(done)} items already have evidence")

    for record in records:
        cached = done.get(resume_key(record))
        if cached is None:
            continue
        for field in ("gold_answers", "answer_format", "answer_parser", "dataset", "split"):
            if cached.get(field) != record.get(field):
                raise SystemExit(f"cached {field} differs from input; use a new output file")

    provider = build_provider(args)
    second_provider = build_second_provider(args)

    print(f"[collect] {len(records)} items, k={args.k}, attention={getattr(provider, 'want_attention', False)}")
    if second_provider is not None:
        print(
            f"[collect] second model: {args.second_provider} at "
            f"{args.second_model_path} -- the cross-model signals will run"
        )
    load = getattr(provider, "load", None)
    if callable(load):
        print(f"[collect] loading {args.model_path} ...")
        started = time.time()
        load()
        print(f"[collect] loaded in {time.time() - started:.1f}s; calls begin")
    else:
        # A provider that loads lazily on first use is still a valid provider, so
        # this is a branch rather than a requirement.
        print("[collect] provider loads lazily; calls begin")

    if second_provider is not None:
        second_load = getattr(second_provider, "load", None)
        if callable(second_load):
            started = time.time()
            second_load()
            print(f"[collect] second model loaded in {time.time() - started:.1f}s")

    written = 0
    generated = 0
    pending = 0
    thin_gold = 0
    unparsed = 0

    # A record that already carries an answer skips the primary call, and the
    # primary call is the only one whose internals are read. Three of the seven
    # signals then come back unavailable -- correctly, because nothing observed
    # them -- but the report at the end blames the serving stack, which sends the
    # reader looking in the wrong place. Say it before the collection runs.
    supplied = sum(1 for r in records if str(r.get("answer", "") or "").strip())
    if supplied:
        print(
            f"[collect] note: {supplied}/{len(records)} input records already carry "
            "an 'answer', so the answer under evaluation is NOT generated for them "
            "and no internals are captured for it:"
        )
        print(
            "[collect]       sequence_confidence, output_entropy, decision_margin and visual_attention "
            "will be unavailable on those items."
        )
        print(
            "[collect]       Drop the 'answer' field to have the model produce the "
            "answer and read its internals in the call that produced it."
        )

    t0 = time.time()
    handle = out_path.open("w", encoding="utf-8")
    try:
        for index, record in enumerate(records):
            key = resume_key(record)
            if key in done:
                handle.write(json.dumps(done[key], ensure_ascii=False) + "\n")
                written += 1
                continue

            if gpu_budget:
                gpu_budget.before_record()
            record = dict(record)
            record["id"] = record.get("id", record.get("image", str(index)))
            question = str(record.get("question", ""))
            answer = str(record.get("answer", "") or "")
            image = str(record.get("image", ""))

            # No answer supplied: generate the one under evaluation, so its
            # internals are captured in the call that produced it.
            primary = None
            if not answer:
                gold = record.get("gold_answers") or record.get("answers") or []
                if record.get("label_from_response"):
                    # A third kind of benchmark, next to "compare against a
                    # reference" and "supply the published answer": the label is
                    # computed *from the response text* by an external scorer.
                    # CHAIR's object-hallucination rule is the one in use here --
                    # it counts the objects a caption names that COCO does not
                    # annotate for that image. Such a label cannot exist before
                    # the response does, so the response is generated now with the
                    # question left alone (a caption benchmark is not a short-answer
                    # one) and ``scripts/label_from_response.py`` fills the label in
                    # afterwards. Until it has run, the record carries no label and
                    # ``scripts/evaluate.py`` refuses the file.
                    primary = provider.generate(image, question, do_sample=False)
                    answer = primary.text
                    record["answer"] = answer
                    generated += 1
                    pending += 1
                elif gold and record.get("answer_format") == "yes_no":
                    # A yes/no benchmark: the reference is one word and the
                    # response is parsed the way the benchmark parses it, so the
                    # answer-matching noise a ten-annotator VQA set carries cannot
                    # get into the label. The prompt is left alone -- asking for "a
                    # short phrase" on a yes/no question would be a different
                    # experiment.
                    primary = provider.generate(image, question, do_sample=False)
                    answer = primary.text
                    label, parsed = yes_no_label(
                        answer, gold[0], parser=record.get("answer_parser", "legacy")
                    )
                    record["answer"] = answer
                    record["label"] = label
                    record["answer_parsed"] = parsed
                    generated += 1
                    if not parsed:
                        unparsed += 1
                elif gold:
                    if len(gold) < 2:
                        thin_gold += 1
                    primary = provider.generate(
                        image, f"{question}\nAnswer with a short phrase only.", do_sample=False
                    )
                    answer = primary.text
                    record["answer"] = answer
                    record["label"] = vqa_label(answer, gold)
                    generated += 1
                else:
                    raise SystemExit(
                        f"{args.data} record {key!r} has neither 'answer' nor "
                        "'gold_answers'; supply one or the other, so the label has "
                        "something to compare the answer against"
                    )

            if "label" not in record and not record.get("label_from_response"):
                raise SystemExit(f"{args.data} record {key!r} has no 'label'")

            evidence, calls = gather_evidence(
                provider,
                question,
                answer,
                image,
                k=args.k,
                prompts=PromptSet(),
                primary_sample=primary,
                second_provider=second_provider,
            )
            record["evidence"] = evidence.to_dict()
            record["model_calls"] = calls + (1 if primary is not None else 0)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if gpu_budget:
                gpu_budget.after_record()

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

    if gpu_budget:
        gpu_budget.finish()
    print(f"[collect] wrote {written} records -> {out_path}")
    print(f"[collect] total generations: {provider.calls}")
    if generated:
        print(f"[collect] answers generated by this script (internals captured): {generated}")
    if pending:
        print(
            f"[collect] {pending} records are labelled from the response text and "
            "carry no label yet. Run scripts/label_from_response.py on this file "
            "before evaluating it; scripts/evaluate.py will refuse it until then."
        )
    if thin_gold:
        print(
            f"[collect] {thin_gold} records supplied fewer than 2 accepted answers. "
            "The VQA rule divides by three, so one reference can never reach the "
            "half-credit line and every correct answer is labelled wrong."
        )
    if unparsed:
        # On a yes/no benchmark, saying neither is not a parsing detail: those
        # items are labelled wrong, so how many there were belongs beside the
        # accuracy the labels produce.
        print(
            f"[collect] {unparsed} responses said neither yes nor no and were "
            "labelled wrong; report this beside the accuracy"
        )
    if getattr(provider, "want_attention", False):
        print(f"[collect] of which attention passes: {provider.attention_calls}")

    # A quick sanity read: which signals actually produced a value on this data.
    rows = load_jsonl(out_path)
    from msrc.signals import build_report

    # The label distribution decides whether there is anything to evaluate. A
    # single-class run scores nothing: AUROC needs both classes, and a fit needs
    # both. Saying it here costs a line and saves finding out after an hour of GPU
    # -- which is how this check came to be written.
    labelled = [r for r in rows if r.get("label") is not None]
    unlabelled = len(rows) - len(labelled)
    counts = Counter(int(r["label"]) for r in labelled)
    print(f"[collect] labels: {dict(sorted(counts.items()))}")
    if len(counts) < 2 and not unlabelled:
        only = next(iter(counts), None)
        why = (
            f"every item was labelled {only}. scripts/evaluate.py will refuse "
            "this file: one class cannot be ranked or calibrated. Check the "
            "label rule against the data before collecting more."
        )
        print(f"[collect] !! single-class labels: {why}")

    live, dead = [], []
    for row in rows[:200]:
        report = build_report(Evidence.from_dict(row["evidence"]))
        for signal in report.signals:
            (live if signal.available else dead).append(signal.name)

    print("[collect] signals that produced a value (first 200 items):")
    for name, count in Counter(live).most_common():
        print(f"    {name:26} {count}")
    if dead:
        print("[collect] signals that never produced a value:")
        # The hint has to fit the cause. "The serving stack did not supply it" is
        # the right thing to check when the model ran and returned nothing; it is
        # the wrong thing to check when the model was never asked, and that is
        # exactly what an input carrying its own answers does.
        hint = (
            "answers were supplied, so the primary was not generated"
            if supplied
            else "check the serving stack supplies it"
        )
        for name, count in Counter(dead).most_common():
            print(f"    {name:26} {count}  <- {hint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

