#!/usr/bin/env python3
"""Re-label the TextVQA cells with the official VQA evaluator's rules.

The in-repo label function compares answers after this project's own
``normalize_text`` and applies ``min(matches/3, 1)``. The scoring shape is the
official one, but the *normalisation* is not: the official evaluator lowercases,
strips punctuation under its own rules, maps number words to digits, drops
articles and expands contractions. Those differences move a small number of binary
risk labels, so "the labels equal the official evaluator's" was not a claim the
code could support.

This recomputes every label with the official normalisation and records both the
official soft score and whether the label moved. The originals are not touched;
the new files go to their own directory with a receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

RUN = "/mnt/data/HLC/msrc_run"
OUT = os.path.join(RUN, "official_textvqa_20261013")

CELLS = [
    ("LLaVA-1.5-13B", "evidence_textvqa_2model.jsonl"),
    ("Qwen2.5-VL-7B", "crossmodel/cell_Qwen2.5-VL-7B_textvqa.jsonl"),
    ("InternVL3-8B", "crossmodel/cell_InternVL3-8B_textvqa.jsonl"),
]

# ---------------------------------------------------------------------------
# Official VQA normalisation (vqaEval.py, VQA v2 release), reproduced verbatim
# apart from formatting. Kept as one block so it can be diffed against upstream.
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
    "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}

MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
              "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
              "nine": "9", "ten": "10"}

ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']


def process_punctuation(text: str) -> str:
    out = text
    for p in PUNCT:
        if (p + " " in text or " " + p in text) or (COMMA_STRIP.search(text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return PERIOD_STRIP.sub("", out, re.UNICODE)


def process_digit_article(text: str) -> str:
    out = []
    for word in text.lower().split():
        word = MANUAL_MAP.get(word, word)
        if word not in ARTICLES:
            out.append(word)
    for i, word in enumerate(out):
        if word in CONTRACTIONS:
            out[i] = CONTRACTIONS[word]
    return " ".join(out)


def official_normalise(text: str) -> str:
    return process_digit_article(process_punctuation(str(text or "")))


def official_score(answer: str, gold_answers) -> float:
    """VQA accuracy: min(annotators agreeing / 3, 1)."""
    res = official_normalise(answer)
    if not res:
        return 0.0
    matches = sum(1 for g in gold_answers if official_normalise(g) == res)
    return min(1.0, matches / 3.0)


# ---------------------------------------------------------------------------
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    receipt = {"built": "2026-10-13",
               "rule": "official VQA normalisation; label=0 iff score>=0.5",
               "cells": {}}

    # Not every cell cached its references. Build a question-keyed table from the
    # cells that did, and join the ones that did not onto it -- verifying the
    # question text and the image file name so a join cannot silently cross items.
    reference = {}
    for _model, rel in CELLS:
        path = os.path.join(RUN, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                gold = row.get("gold_answers")
                if not gold:
                    continue
                reference[row["id"]] = (list(gold), str(row.get("question", "")),
                                        os.path.basename(str(row.get("image", ""))))
    if not reference:
        raise SystemExit("no cell carries gold_answers; nothing to join onto")
    print("reference table built from %d items\n" % len(reference))

    for model, rel in CELLS:
        path = os.path.join(RUN, rel)
        if not os.path.exists(path):
            print("MISSING %s" % path)
            continue
        with open(path, encoding="utf-8-sig") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]

        moved = 0
        joined = own = unjoined = 0
        scores = []
        out_rows = []
        for row in rows:
            gold = row.get("gold_answers")
            if gold:
                own += 1
            else:
                ref = reference.get(row["id"])
                if ref is None:
                    unjoined += 1
                    continue
                gold, ref_q, ref_img = ref
                if ref_q != str(row.get("question", "")):
                    raise SystemExit("join crossed items: %s question differs" % row["id"])
                if ref_img and os.path.basename(str(row.get("image", ""))) != ref_img:
                    raise SystemExit("join crossed items: %s image differs" % row["id"])
                row["gold_answers"] = list(gold)
                joined += 1

            score = official_score(row.get("answer", ""), gold)
            new_label = 0 if score >= 0.5 else 1
            old_label = int(row.get("label", new_label))
            row["label_original_rule"] = old_label
            row["official_vqa_score"] = round(score, 6)
            row["label"] = new_label
            scores.append(score)
            if new_label != old_label:
                moved += 1
            out_rows.append(row)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        share_moved = moved / max(len(out_rows), 1)
        # A mean official score at the floor means every answer missed, which in
        # practice means the references were absent rather than that the model got
        # everything wrong. Refuse to write labels in that case.
        if mean_score == 0.0 or share_moved > 0.25:
            raise SystemExit(
                "%s: mean official score %.3f and %.1f%% of labels moved -- that is "
                "a broken reference join, not a relabelling. Nothing written."
                % (model, mean_score, 100 * share_moved))

        out_name = "cell_%s_textvqa_official.jsonl" % model
        out_path = os.path.join(OUT, out_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in out_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        receipt["cells"][model] = {
            "source": rel, "source_sha256": sha256(path), "n": len(rows),
            "written": len(out_rows), "gold_own": own, "gold_joined": joined,
            "unjoined": unjoined,
            "labels_moved": moved, "share_moved": round(share_moved, 6),
            "mean_official_score": round(mean_score, 6),
            "out": out_name, "out_sha256": sha256(out_path),
        }
        print("%-16s n=%4d written=%4d gold(own/joined)=%d/%d  moved=%3d (%.2f%%)"
              "  mean score=%.3f  -> %s"
              % (model, len(rows), len(out_rows), own, joined, moved,
                 100 * share_moved, mean_score, out_name))

    with open(os.path.join(OUT, "receipt.json"), "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, ensure_ascii=False, indent=2)
    print("\nreceipt:", os.path.join(OUT, "receipt.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
