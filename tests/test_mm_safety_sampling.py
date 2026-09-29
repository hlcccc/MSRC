"""Building a category-balanced safety set, and keeping it stable.

Two properties matter and only one of them is obvious. Balanced coverage is the
obvious one: take the head of the file and the run describes whichever categories
sort first. The other is stability -- adding a category to the release must not
re-draw the sample for the categories already there, because evidence for the old
sample has already been paid for on a GPU.

That second property is why this has its own file: it needs no pandas, so it should
not be skipped wherever pandas is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.datasets.mm_safety import stratified_sample  # noqa: E402


def _records(counts: dict, *, prefix: str = "TYPO") -> list:
    out = []
    for category, n in counts.items():
        for index in range(n):
            out.append({
                "id": f"mmsafety:{category}:{prefix}:{index}",
                "category": category,
                "attack": prefix,
                "question": f"q{index}",
            })
    return out


def test_it_takes_at_most_per_category_from_each_category():
    records = _records({"Fraud": 50, "Sex": 50, "HateSpeech": 50})
    sample = stratified_sample(records, per_category=20, seed=1)
    assert len(sample) == 60
    counts = {}
    for record in sample:
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    assert counts == {"Fraud": 20, "Sex": 20, "HateSpeech": 20}


def test_a_small_category_contributes_all_it_has():
    """Malware_Generation has 44 TYPO items; a category with 3 must not crash."""
    records = _records({"Fraud": 50, "Tiny": 3})
    sample = stratified_sample(records, per_category=20, seed=1)
    counts = {}
    for record in sample:
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    assert counts == {"Fraud": 20, "Tiny": 3}


def test_it_is_deterministic():
    records = _records({"Fraud": 50, "Sex": 50, "Gov_Decision": 50})
    first = stratified_sample(records, per_category=10, seed=7)
    second = stratified_sample(records, per_category=10, seed=7)
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_a_different_seed_gives_a_different_sample():
    records = _records({"Fraud": 200})
    assert (
        [r["id"] for r in stratified_sample(records, per_category=20, seed=1)]
        != [r["id"] for r in stratified_sample(records, per_category=20, seed=2)]
    )


def test_adding_a_category_does_not_change_the_others():
    """The property the per-category seeding exists for.

    This is not hypothetical: the first safety set was built with a single RNG
    seeded once, and when two categories finished downloading, every category's
    draw moved and 182 already-collected items fell out of the sample.
    """
    before = _records({"Fraud": 60, "Sex": 60, "HateSpeech": 60})
    after = before + _records({"Political_Lobbying": 60})

    old_sample = stratified_sample(before, per_category=20, seed=42)
    new_sample = stratified_sample(after, per_category=20, seed=42)

    old_ids = {r["id"] for r in old_sample}
    new_ids = {r["id"] for r in new_sample}

    assert old_ids <= new_ids, "every previously sampled item is still sampled"
    added = new_ids - old_ids
    assert len(added) == 20
    assert {r["category"] for r in new_sample if r["id"] in added} == {"Political_Lobbying"}


def test_the_sample_is_not_ordered_by_category():
    """Otherwise the first hour of a run covers only the first few categories."""
    records = _records({name: 30 for name in
                        ("Fraud", "Sex", "HateSpeech", "Gov_Decision", "Physical_Harm")})
    sample = stratified_sample(records, per_category=10, seed=42)
    order = [r["category"] for r in sample]
    assert order != sorted(order)


def test_an_empty_input_gives_an_empty_sample():
    assert stratified_sample([], per_category=20, seed=1) == []
