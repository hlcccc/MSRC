"""Image identities for the held-out split.

A held-out split is only as honest as its grouping key. POPE polls the same 500
COCO images under all three settings and the adapter writes them to three different
paths, so grouping by path puts the same photograph on both sides of the boundary:
87.3% of a 1,500-item POPE run landed on images that appeared in both halves, and
the dev and test files shared 217 images. A model that has seen a photograph once
has effectively seen every question asked about it.

These tests pin the key that fixes that, and the refusal to invent one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.groups import image_group_id  # noqa: E402


def _pope(split: str, path: str, coco: str = "COCO_val2014_000000153865") -> dict:
    return {
        "id": f"pope:{split}:{coco}",
        "dataset": "POPE",
        "split": split,
        "image": path,
    }


def test_one_photograph_under_three_settings_is_one_group():
    """The regression this file exists for."""
    random = _pope("random", "/data/pope_random/COCO_val2014_000000153865.jpg")
    popular = _pope("popular", "/data/pope_popular/COCO_val2014_000000153865.jpg")
    adversarial = _pope("adversarial", "/data/pope_adversarial/COCO_val2014_000000153865.jpg")

    assert image_group_id(random) == image_group_id(popular) == image_group_id(adversarial)


def test_two_photographs_stay_two_groups():
    a = _pope("random", "/data/pope_random/COCO_val2014_000000153865.jpg")
    b = _pope(
        "random",
        "/data/pope_random/COCO_val2014_000000470699.jpg",
        coco="COCO_val2014_000000470699",
    )

    assert image_group_id(a) != image_group_id(b)


def test_the_group_name_does_not_depend_on_where_the_file_was_written():
    """Two runs that stage the same image in different directories must agree.

    A key derived from the path would silently produce a different split for a
    differently-staged copy of the same data, and neither split would be wrong in a
    way anyone could see.
    """
    here = _pope("random", "/tmp/run-a/COCO_val2014_000000153865.jpg")
    there = _pope("random", "/mnt/data/HLC/pope_images/COCO_val2014_000000153865.jpg")

    assert image_group_id(here) == image_group_id(there)


def test_the_identity_is_recognised_regardless_of_case():
    row = {"dataset": "pope", "id": "q1", "image": "/x/coco_val2014_000000153865.jpg"}

    assert image_group_id(row) == "coco:coco_val2014_000000153865"


def test_the_identity_is_recognised_from_the_id_alone():
    """The adapter writes the COCO name into the id, not only into the path."""
    row = {"id": "pope:popular:COCO_val2014_000000153865", "image": "/x/renamed.jpg"}

    assert image_group_id(row) == "coco:coco_val2014_000000153865"


def test_an_explicit_source_id_is_preferred_over_the_path():
    row = _pope("random", "/x/COCO_val2014_000000470699.jpg")
    row["source_image_id"] = "COCO_val2014_000000153865"

    assert image_group_id(row) == "coco:coco_val2014_000000153865"


def test_a_non_pope_record_uses_its_declared_image_id():
    row = {"dataset": "MM-SafetyBench", "image_id": "SD_TYPO/1.jpg", "image": "/a/1.jpg"}

    assert image_group_id(row) == "mm-safetybench:SD_TYPO/1.jpg"


def test_a_non_pope_record_without_an_explicit_id_falls_back_to_its_path():
    row = {"dataset": "HallusionBench", "image": "/a/figure.png"}

    assert image_group_id(row) == "/a/figure.png"


def test_a_pope_record_with_no_coco_name_falls_back_rather_than_inventing_one():
    """Falling back keeps the item; it does not silently merge it with anything."""
    row = {"dataset": "POPE", "id": "pope:random:weird", "image": "/x/weird.jpg"}

    assert image_group_id(row) == "/x/weird.jpg"


def test_a_record_with_no_identity_at_all_is_refused():
    """Every alternative here is a guess, and a guessed group is a leaky split."""
    with pytest.raises(ValueError, match="no image identity"):
        image_group_id({"question": "is there a car?"})
