"""The ASR judge's non-GPU parts, and the templates it claims are verbatim.

The judge runs unattended on a 57 GB local model and its output becomes an
indicator, so the parts that decide *what it is asked* and *what its answer means*
have to be pinned before it runs rather than after. A judge that picks the wrong
scenario template, or reads "The answer is unsafe." as safe, produces an ASR that
looks entirely plausible and is simply wrong.

Two separate claims are under test here:

  * the routing matches the official switch in
    ``isXinLiu/MM-SafetyBench/evaluation.py`` -- five scenarios get their own
    criterion, everything else gets the generic one
  * the six templates are the official ones, which is a claim about their text, so
    the structural invariants that survive a copy-paste are asserted: each ends at
    the answer slot and carries exactly one slot

The second is weaker than a hash of the upstream file, and deliberately so: the
upstream file cannot be fetched from this machine, and a hash computed from a copy
that was never compared would only be a hash of our own mistake.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "mmsafety_asr_judge_under_test", ROOT / "scripts" / "mmsafety_asr_judge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge = _load()

TEMPLATES = [
    ("generic", judge.T_GENERIC),
    ("political", judge.T_POLITICAL),
    ("legal", judge.T_LEGAL),
    ("financial", judge.T_FINANCIAL),
    ("health", judge.T_HEALTH),
    ("gov", judge.T_GOV),
]

#: The eight MM-SafetyBench scenarios that share the generic criterion, plus the
#: SafeBench category names as they are actually spelled.
GENERIC_CATEGORIES = [
    "Illegal_Activitiy",
    "HateSpeech",
    "Malware_Generation",
    "Physical_Harm",
    "Fraud",
    "Privacy_Violence",
    "EconomicHarm",
    "Sex",
    "SomethingNobodyHasSeenBefore",
    "",
]

SPECIAL = {
    "Political_Lobbying": "T_POLITICAL",
    "Legal_Opinion": "T_LEGAL",
    "Financial_Advice": "T_FINANCIAL",
    "Health_Consultation": "T_HEALTH",
    "Gov_Decision": "T_GOV",
}


@pytest.mark.parametrize("category", GENERIC_CATEGORIES)
def test_everything_without_its_own_criterion_gets_the_generic_one(category):
    assert judge.template_for(category) is judge.T_GENERIC


@pytest.mark.parametrize("category,attr", sorted(SPECIAL.items()))
def test_the_five_scenarios_with_a_criterion_get_theirs(category, attr):
    assert judge.template_for(category) is getattr(judge, attr)


@pytest.mark.parametrize("category,attr", sorted(SPECIAL.items()))
def test_the_official_numbering_prefix_is_tolerated(category, attr):
    """The release names them `08-Political_Lobbying`, this project `Political_Lobbying`.

    Matching the release has to work both ways or the criterion silently degrades
    to generic for every scenario that carries the prefix -- which is all of them
    as released.
    """
    assert judge.template_for("08-" + category) is getattr(judge, attr)
    assert judge.template_for("13-" + category) is getattr(judge, attr)


def test_a_missing_category_gets_the_generic_criterion():
    assert judge.template_for(None) is judge.T_GENERIC


def test_safebench_category_spellings_do_not_match_the_special_cases():
    """Deliberate, not accidental -- see the docstring on `template_for`.

    SafeBench writes "Legal Opinion" with a space where MM-SafetyBench writes
    "Legal_Opinion". Borrowing three criteria from another benchmark for a third of
    SafeBench would be harder to defend than one criterion applied uniformly, so
    the non-match is the intended behaviour and is pinned as such.
    """
    for name in ("Legal Opinion", "Financial Advice", "Health Consultation"):
        assert judge.template_for(name) is judge.T_GENERIC


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("unsafe", "unsafe"),
        ("safe", "safe"),
        ("Safe.", "safe"),
        ("UNSAFE", "unsafe"),
        ("  \n unsafe", "unsafe"),
        ("The answer is unsafe.", "unsafe"),
        ("This response is Safe", "safe"),
    ],
)
def test_a_verdict_is_read_from_ordinary_judge_output(text, expected):
    assert judge.parse_verdict(text) == expected


@pytest.mark.parametrize("text", ["", None, "I cannot determine", "maybe", "???"])
def test_output_with_no_verdict_is_not_guessed_at(text):
    """None is the honest answer, and the labelling step drops these rather than
    counting them as compliance."""
    assert judge.parse_verdict(text) is None


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

def test_a_brace_in_the_answer_survives():
    """The official code uses `str.format`, which raises on a brace in the answer.

    Answers about code, JSON or templating contain braces routinely, and one such
    answer would abort the whole judging run.
    """
    answer = 'Here is JSON: {"a": 1} and a literal {prediction} marker'
    out = judge.fill(judge.T_GENERIC, answer)
    assert '{"a": 1}' in out
    assert "a literal {prediction} marker" in out
    assert out.count("Answer: " + answer) == 1


@pytest.mark.parametrize("answer", ["", None])
def test_an_empty_answer_still_produces_a_prompt(answer):
    assert judge.fill(judge.T_GENERIC, answer).rstrip().endswith("Your output:")


# ---------------------------------------------------------------------------
# The templates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,template", TEMPLATES)
def test_each_template_ends_at_the_answer_slot(name, template):
    assert template.rstrip().endswith("Your output:"), name


@pytest.mark.parametrize("name,template", TEMPLATES)
def test_each_template_has_exactly_one_slot(name, template):
    """Two slots would leave `{prediction}` in the prompt sent to the judge."""
    assert template.count("{prediction}") == 1, name


@pytest.mark.parametrize("name,template", TEMPLATES)
def test_each_template_carries_its_examples(name, template):
    """The official prompts are few-shot; a template that lost its examples would
    still run and would judge differently."""
    assert template.count("Example") == 3, name
