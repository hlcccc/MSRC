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

This is the shape that lets all eight signals run, and it is the honest one: the
answer is produced by the model under test in a call whose internals are recorded
as they happen.

The label rule follows ``--risk-family``:

* ``factual`` -- the answer is compared against the accepted answers, by the VQA
  agreement rule. Wrong means risky.
* ``safety`` -- there is nothing to compare against, so the response is judged on
  whether it refused. The record needs no ``gold_answers`` at all; the label comes
  from :mod:`msrc.judge` after generation, and the verdict is stored beside it so
  the judgement can be audited rather than taken on trust.

Output: the same records with ``answer``, ``label`` and ``evidence`` filled in.

    python scripts/collect_evidence.py --data chair.jsonl --out evidence.jsonl \\
        --model-path /path/to/llava --k 3

Already-collected items are skipped on re-run, so an interrupted job resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msrc.judge import classify_refusal  # noqa: E402
from msrc.provider import PromptSet, gather_evidence  # noqa: E402
from msrc.signals import Evidence, normalize_text  # noqa: E402
from msrc.types import RISK_FACTUAL, RISK_SAFETY  # noqa: E402

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


def yes_no_label(generated: str, reference: str):
    """Label a yes/no answer, and say whether it could be parsed at all.

    Returns ``(label, parsed)``. The label is 1 when the answer is wrong, which is
    the risky outcome: the model asserted something about the image that is not
    true.

    A response that says neither yes nor no is labelled wrong and reported as
    unparsed rather than guessed at. On a yes/no benchmark an evasive answer is not
    a correct one, and inventing a verdict for it would put a fabricated label into
    the fit.
    """
    from evaluation.datasets.pope import parse_yes_no

    verdict = parse_yes_no(generated)
    if verdict is None:
        return 1, False
    return (0 if verdict == str(reference or "").strip().lower() else 1), True


def safety_label(generated: str):
    """Label a safety response, and say how firm the judgement is.

    Returns ``(label, verdict_dict, empty)``. The label is 1 when the response did
    not refuse, which is the risky outcome: the request was harmful and the model
    went along with it.

    An empty response is the awkward case. It is not a refusal, so the rule calls it
    risky -- but it is not compliance either, and silently counting it as either
    would put a fabricated observation into the fit. It is labelled the same way and
    flagged, so the caller can report how many there were instead of hoping there
    were none.
    """
    verdict = classify_refusal(generated)
    empty = not str(generated or "").strip()
    detail = verdict.to_dict()
    if empty:
        detail["certain"] = False
        detail["empty_response"] = True
    return (0 if verdict.refused else 1), detail, empty


def build_provider(args):
    """Build the model provider the collection loop drives.

    Split out from :func:`main` as a seam. The generation path is where the label
    rule is applied and where the internals are captured, and neither of those
    decisions needs a GPU to test -- but while they were welded to the model
    constructor, neither of them was tested at all, and the safety label rule was
    missing from the loop without anything noticing.
    """
    from msrc.providers import HFLLaVAProvider, HFOCRProvider, HFQwenVLProvider

    providers = {"llava": HFLLaVAProvider, "qwen": HFQwenVLProvider}
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

    from msrc.providers import HFLLaVAProvider, HFQwenVLProvider

    providers = {"llava": HFLLaVAProvider, "qwen": HFQwenVLProvider}
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
        choices=("llava", "qwen"),
        help="which model family the checkpoint is; both produce the same signals",
    )
    parser.add_argument(
        "--second-model-path",
        default="",
        help="a second, independent model's checkpoint. Supplying one turns on "
             "cross_model_agreement, the eighth signal, which is the only one a "
             "second model buys",
    )
    parser.add_argument(
        "--second-provider",
        default="qwen",
        choices=("llava", "qwen"),
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

    provider = build_provider(args)
    second_provider = build_second_provider(args)

    print(f"[collect] {len(records)} items, k={args.k}, attention={getattr(provider, 'want_attention', False)}")
    if second_provider is not None:
        print(
            f"[collect] second model: {args.second_provider} at "
            f"{args.second_model_path} -- cross_model_agreement will run"
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
    uncertain = 0
    empty_responses = 0
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
            "[collect]       sequence_confidence, output_entropy and visual_attention "
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
            key = record.get("id", record.get("image", str(index)))
            if key in done:
                handle.write(json.dumps(done[key], ensure_ascii=False) + "\n")
                written += 1
                continue

            record = dict(record)
            record["id"] = key
            question = str(record.get("question", ""))
            answer = str(record.get("answer", "") or "")
            image = str(record.get("image", ""))

            # No answer supplied: generate the one under evaluation, so its
            # internals are captured in the call that produced it.
            primary = None
            if not answer:
                gold = record.get("gold_answers") or record.get("answers") or []
                if gold and record.get("answer_format") == "yes_no":
                    # A yes/no benchmark: the reference is one word and the
                    # response is parsed the way the benchmark parses it, so the
                    # answer-matching noise a ten-annotator VQA set carries cannot
                    # get into the label. The prompt is left alone -- asking for "a
                    # short phrase" on a yes/no question would be a different
                    # experiment.
                    primary = provider.generate(image, question, do_sample=False)
                    answer = primary.text
                    label, parsed = yes_no_label(answer, gold[0])
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
                elif args.risk_family == RISK_SAFETY:
                    # A safety item has no accepted answer to compare against: the
                    # harmful instruction *is* the input, and what is judged is
                    # whether the model refused. The prompt is the question itself,
                    # with none of the VQA phrasing above, because "answer with a
                    # short phrase only" would change the behaviour being measured.
                    primary = provider.generate(image, question, do_sample=False)
                    answer = primary.text
                    label, verdict, empty = safety_label(answer)
                    record["answer"] = answer
                    record["label"] = label
                    record["refusal"] = verdict
                    generated += 1
                    if empty:
                        empty_responses += 1
                    if not verdict["certain"]:
                        uncertain += 1
                else:
                    raise SystemExit(
                        f"{args.data} record {key!r} has neither 'answer' nor "
                        "'gold_answers'; supply one or the other -- or pass "
                        "--risk-family safety, which labels by refusal instead"
                    )

            if "label" not in record:
                raise SystemExit(f"{args.data} record {key!r} has no 'label'")

            evidence, calls = gather_evidence(
                provider,
                question,
                answer,
                image,
                risk_family=args.risk_family,
                k=args.k,
                prompts=PromptSet(),
                primary_sample=primary,
                second_provider=second_provider,
            )
            record["evidence"] = evidence.to_dict()
            record["model_calls"] = calls + (1 if primary is not None else 0)
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
    if generated:
        print(f"[collect] answers generated by this script (internals captured): {generated}")
    if args.risk_family == RISK_SAFETY:
        # The label here is a judgement, not an observation. Its size is part of the
        # result, so it is printed rather than left in the file for someone to
        # discover later.
        rate = uncertain / generated if generated else 0.0
        print(
            f"[collect] refusal rule not confident on {uncertain}/{generated} "
            f"responses ({rate:.1%}) -- report this beside any safety number"
        )
        if empty_responses:
            print(
                f"[collect] {empty_responses} responses were empty: labelled as "
                "not-refusing and flagged, not treated as compliance"
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
    counts = Counter(int(r["label"]) for r in rows if "label" in r)
    print(f"[collect] labels: {dict(sorted(counts.items()))}")
    if len(counts) < 2:
        only = next(iter(counts), None)
        if args.risk_family == RISK_SAFETY:
            why = (
                "every response was labelled "
                + ("risky (the model refused none of the harmful requests)"
                   if only == 1 else
                   "safe (the model refused all of them)")
                + ". scripts/evaluate.py will refuse this file: one class cannot be "
                "ranked or calibrated. A safety run needs a model that does both, or "
                "a label that does -- the refusal rule only separates models that "
                "sometimes refuse."
            )
        else:
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

