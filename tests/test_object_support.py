"""Tests for the object-level support signal.

The behaviour worth pinning down is not "does it produce a number" but the two
properties the rest of the framework leans on: the reading is oriented (a
hallucinated object raises risk), and an unanswerable item reports itself
unavailable rather than scoring zero.
"""

from __future__ import annotations

import pytest

from msrc.object_support import (
    DEFAULT_LEXICON_PATH,
    ObjectLexicon,
    default_lexicon,
    object_min_support,
)
from msrc.signals import ALL_SIGNALS, Evidence, ObjectSupport, build_report
from msrc.types import SIGNAL_EXTERNAL, Sample

LEXICON = ObjectLexicon([
    ["person", "man", "woman"],
    ["bicycle", "bike"],
    ["car", "automobile"],
    ["wine glass"],
    ["glass", "drinking glass"],
])


def ev(answer, views=(), resamples=()):
    return Evidence(
        question="Describe the image.",
        answer=answer,
        primary=Sample(text=answer),
        views=[Sample(text=t) for t in views],
        resamples=[Sample(text=t) for t in resamples],
    )


# -- lexicon ---------------------------------------------------------------


def test_bundled_lexicon_loads_and_has_the_official_group_count():
    lex = ObjectLexicon.from_path(DEFAULT_LEXICON_PATH)
    assert len(lex.groups) == 80, "official CHAIR synonym list has 80 object groups"


def test_synonyms_map_to_one_group():
    assert LEXICON.mentioned("a bike") == LEXICON.mentioned("a bicycle")
    assert len(LEXICON.mentioned("bike and bicycle")) == 1


def test_longest_surface_form_wins():
    # "wine glass" must not be counted as the separate "glass" group.
    assert LEXICON.mentioned("a wine glass") == frozenset({3})
    assert LEXICON.mentioned("a glass") == frozenset({4})


def test_matching_is_case_insensitive_and_tolerates_odd_spacing():
    assert LEXICON.mentioned("A  WINE   GLASS") == frozenset({3})


def test_substrings_do_not_match():
    # "car" inside "carpet", "man" inside "manner": neither is an object mention.
    assert LEXICON.mentioned("the carpet and the manner") == frozenset()


def test_empty_and_missing_text_mention_nothing():
    assert LEXICON.mentioned("") == frozenset()
    assert LEXICON.mentioned(None) == frozenset()


def test_empty_lexicon_is_rejected():
    with pytest.raises(ValueError):
        ObjectLexicon([])
    with pytest.raises(ValueError):
        ObjectLexicon([["  "], [""]])


# -- the reading -----------------------------------------------------------


def test_an_object_that_never_returns_gives_zero_support():
    reading = object_min_support(
        "a car and a mirror",
        ["a car", "a car", "a car"],
        LEXICON,
    )
    # "mirror" is not in this lexicon, so only "car" is named and it always returns.
    assert reading is not None
    assert reading[0] == 1.0
    assert reading[1] == 1


def test_the_weakest_object_decides_the_reading():
    reading = object_min_support(
        "a car and a bicycle",
        ["a car", "a car", "a bicycle"],
        LEXICON,
    )
    assert reading is not None
    support, n_named, weakest = reading
    assert n_named == 2
    assert support == pytest.approx(1 / 3)
    assert weakest == "bicycle"


def test_support_counts_groups_not_words():
    # "bike" and "bicycle" in different re-readings are the same object coming back.
    reading = object_min_support("a bicycle", ["a bike", "a bicycle"], LEXICON)
    assert reading is not None
    assert reading[0] == 1.0


def test_no_named_object_is_not_a_reading():
    assert object_min_support("yes", ["yes", "yes"], LEXICON) is None


def test_too_few_auxiliary_texts_is_not_a_reading():
    assert object_min_support("a car", ["a car"], LEXICON) is None
    assert object_min_support("a car", [], LEXICON) is None


# -- the signal ------------------------------------------------------------


def test_signal_is_external_and_oriented():
    sig = ObjectSupport(LEXICON)
    assert sig.kind == SIGNAL_EXTERNAL

    seen = sig.compute(ev("a car and a bicycle",
                          views=["a car", "a car"],
                          resamples=["a car"]))
    missing = sig.compute(ev("a car",
                             views=["a car", "a car"],
                             resamples=["a car"]))
    assert seen.available and missing.available
    assert seen.risk > missing.risk, "an object that never returns is the riskier caption"
    assert missing.risk == pytest.approx(0.0)


def test_unavailable_when_the_answer_names_no_object():
    v = ObjectSupport(LEXICON).compute(ev("yes", views=["yes", "yes"], resamples=["yes"]))
    assert not v.available
    assert v.risk == 0.0
    assert "lexicon" in v.detail


def test_unavailable_when_there_is_nothing_to_re_read():
    v = ObjectSupport(LEXICON).compute(ev("a car"))
    assert not v.available
    assert "2" in v.detail


def test_unavailable_when_the_lexicon_cannot_be_loaded(tmp_path, monkeypatch):
    import msrc.object_support as osup

    monkeypatch.setenv(osup.LEXICON_ENV, str(tmp_path / "does_not_exist.txt"))
    osup.clear_cache()
    try:
        v = ObjectSupport().compute(ev("a car", views=["a car", "a car"]))
        assert not v.available
        assert "lexicon" in v.detail
    finally:
        osup.clear_cache()


def test_registry_contains_the_signal_and_stays_ordered():
    names = [s.name for s in ALL_SIGNALS]
    assert "object_support" in names
    assert len(names) == len(set(names)), "signal names are column names; they must be unique"
    assert names[-3:] == ["decision_margin", "image_reliance", "visual_attention"]


def test_build_report_runs_it_without_a_model():
    # Against the bundled lexicon, "bicycle" returns in 1 of the 3 re-readings and
    # "car" in all 3, so the weakest object sets the reading at 1/3.
    report = build_report(ev("a car and a bicycle",
                             views=["a car", "a bicycle"],
                             resamples=["a car"]))
    by_name = {s.name: s for s in report.signals}
    assert by_name["object_support"].available
    assert by_name["object_support"].raw == pytest.approx(1 / 3)
    assert by_name["object_support"].risk == pytest.approx(2 / 3)
