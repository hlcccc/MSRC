"""The collection script's label rule, its resume guards, and its provider seam.

The model is the only expensive part of collection, so it is the only part these
tests stand in for. ``build_provider`` exists as a seam for exactly that: the label
rule, the evidence written to disk and the resume bookkeeping are all decided in
the loop, and none of them needs a GPU to be wrong.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msrc.provider import StubProvider  # noqa: E402


def _load_collector():
    """Import ``scripts/collect_evidence.py`` as a module."""
    spec = importlib.util.spec_from_file_location(
        "collect_evidence_under_test", ROOT / "scripts" / "collect_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = _load_collector()

RIGHT = "Flickr"
WRONG = "Nightclub"
GOLD = ["Flickr", "flickr", "Flicker"]


# ---------------------------------------------------------------------------
# The label rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "generated, gold, expected",
    [
        ("Flickr", ["Flickr", "flickr", "Flicker"], 0),
        ("Flickr", ["Nightclub", "Bar", "Pub"], 1),
        ("", ["Flickr"], 1),
        # One of three annotators agreeing is below the VQA half-credit line.
        ("Flickr", ["Flickr", "Bar", "Pub"], 1),
        # Two of three is at it, and passes.
        ("Flickr", ["Flickr", "flickr", "Pub"], 0),
    ],
)
def test_vqa_agreement_rule(generated, gold, expected):
    assert collector.vqa_label(generated, gold) == expected


def test_one_reference_answer_can_never_reach_the_threshold():
    """A sharp edge of the VQA rule, pinned so it stays a known one.

    Matches are divided by three, so a correct answer against a single reference
    scores 0.33 and is labelled wrong. TextVQA ships ten answers per question, so
    this never fires there; it would fire for anyone reusing the script with a
    thinner gold column, which is why the collector prints how many such records
    it saw.
    """
    assert collector.vqa_label("Flickr", ["Flickr"]) == 1
    assert collector.vqa_label("Flickr", ["Flickr", "Flickr"]) == 0


# ---------------------------------------------------------------------------
# The loop, with the model stubbed out
# ---------------------------------------------------------------------------

ITEMS = [
    {"id": "f1", "question": "What is this?", "image": "/a.png", "gold_answers": GOLD},
    {"id": "f2", "question": "What is this?", "image": "/b.png", "gold_answers": GOLD},
]


def _run(monkeypatch, tmp_path, answers, records, extra=()):
    """Drive ``main`` with a stub provider and return the written records."""
    source = tmp_path / "in.jsonl"
    source.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"

    monkeypatch.setattr(
        collector, "build_provider", lambda args: StubProvider(answers=list(answers))
    )
    argv = [
        "collect_evidence.py",
        "--data", str(source),
        "--out", str(out),
        "--model-path", str(tmp_path),          # never opened: the provider is stubbed
        "--no-ocr",
        "--k", "1",
        "--save-every", "1",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert collector.main() == 0
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]


def test_an_input_that_brings_its_own_answers_says_what_it_costs(
    monkeypatch, tmp_path, capsys
):
    """The primary call is the only one whose internals are read.

    Supplying an answer skips it, so three of the seven signals legitimately come
    back unavailable. That is correct; what was wrong is that the run only said
    "check the serving stack supplies it" at the end, which points the reader at
    the model when the cause is the input. This was found by running the
    collector on CPU with records copied out of a previous evidence file.
    """
    records = [dict(item, answer=RIGHT, label=0) for item in ITEMS]
    rows = _run(monkeypatch, tmp_path, [RIGHT], records)
    out = capsys.readouterr().out

    assert len(rows) == 2
    assert "already carry" in out, "the cost is stated before the run, not after"
    assert "answers were supplied" in out, "and the closing hint fits the cause"
    assert "check the serving stack supplies it" not in out

    for row in rows:
        primary = row["evidence"]["primary"]
        assert primary["sequence_confidence"] is None
        assert primary["visual_attention_mass"] is None


def test_shared_scheduling_preserves_all_sample_outputs(monkeypatch, tmp_path):
    from pipeline import gpu_sharing
    baseline_dir, shared_dir = tmp_path/'baseline', tmp_path/'shared'
    baseline_dir.mkdir()
    shared_dir.mkdir()
    baseline = _run(monkeypatch, baseline_dir, [RIGHT, WRONG], ITEMS)
    events = []
    class Budget:
        def __init__(self, output, fraction):
            events.append(('budget', fraction))
        def before_record(self):
            events.append('before')
        def after_record(self):
            events.append('after')
        def finish(self):
            events.append('finished')
    monkeypatch.setenv('MSRC_GPU_SHARE_POLICY', 'synthetic-policy.json')
    monkeypatch.setattr(gpu_sharing, 'FactualWorkBudget', Budget)
    monkeypatch.setattr(collector, 'cap_gpu_memory', lambda fraction, *devices: events.append(('cap', fraction)))
    shared = _run(monkeypatch, shared_dir, [RIGHT, WRONG], ITEMS,
                  extra=('--device', 'cuda:0', '--gpu-memory-fraction', '0.45'))
    assert shared == baseline
    assert events == [('cap', 0.45), ('budget', 0.45),
                      'before', 'after', 'before', 'after', 'finished']


def test_resuming_does_not_collapse_items_that_share_an_id(monkeypatch, tmp_path):
    """POPE names a record after its image, and one image is polled about several
    objects, so ids repeat inside a single file.

    Keyed on the id alone, a re-run folds every record sharing that id into
    whichever one was written last: the row count still comes out right and the
    file silently loses items. The real 1,500-item POPE set has 978 distinct ids
    across 1,500 items, so this is the normal case, not a corner case.
    """
    records = [
        {"id": "pope:random:img1", "question": "Is there a car?",
         "image": "/a.png", "answer": RIGHT, "label": 0},
        {"id": "pope:random:img1", "question": "Is there a person?",
         "image": "/a.png", "answer": RIGHT, "label": 0},
        {"id": "pope:random:img2", "question": "Is there a car?",
         "image": "/b.png", "answer": RIGHT, "label": 0},
    ]
    source = tmp_path / "in.jsonl"
    source.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"

    monkeypatch.setattr(
        collector, "build_provider", lambda args: StubProvider(answers=[RIGHT])
    )
    monkeypatch.setattr(sys, "argv", [
        "collect_evidence.py", "--data", str(source), "--out", str(out),
        "--model-path", str(tmp_path), "--no-ocr", "--k", "1", "--save-every", "1",
    ])

    assert collector.main() == 0
    first = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]
    assert len(first) == 3

    # The re-run is the interesting one: it loads what is already on disk.
    assert collector.main() == 0
    second = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]

    assert len(second) == 3, "the row count is not the thing that goes wrong"
    questions = [(r["image"], r["question"]) for r in second]
    assert len(set(questions)) == 3, (
        f"resume collapsed records sharing an id: {questions}"
    )


# ---------------------------------------------------------------------------
# Guards on the resume cache
# ---------------------------------------------------------------------------
#
# Resuming is an optimisation that silently becomes a correctness bug the moment
# the input stops matching the cache. Both failures below leave a file whose row
# count is right and whose contents are wrong, which is the kind of error that
# survives all the way into a reported metric.

def _argv(collector_argv):
    return ["collect_evidence.py", *collector_argv]


def test_a_duplicated_input_item_is_refused_before_the_model_loads(
    monkeypatch, tmp_path
):
    """Two rows with one (id, image, question) key cannot be told apart on resume.

    Whichever was cached first would answer for both, so one of the two questions
    would be scored with the other's evidence.
    """
    row = {"id": "f1", "question": "What is this?", "image": "/a.png"}
    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )

    def _explode(args):  # pragma: no cover - the guard must fire first
        raise AssertionError("the model was built before the input was validated")

    monkeypatch.setattr(collector, "build_provider", _explode)
    monkeypatch.setattr(sys, "argv", _argv([
        "--data", str(source), "--out", str(tmp_path / "out.jsonl"),
        "--model-path", str(tmp_path), "--no-ocr", "--k", "1", "--save-every", "1",
    ]))

    with pytest.raises(SystemExit, match="duplicate input item key"):
        collector.main()


FACTUAL_ITEM = {
    "id": "f1",
    "question": "What is this?",
    "image": "/a.png",
    "dataset": "TextVQA",
    "split": "val",
    "gold_answers": ["Flickr"],
    "answer_format": "vqa",
}


def test_a_cached_row_labelled_against_different_gold_is_refused(monkeypatch, tmp_path):
    """Reusing an output file across two gold revisions mixes two label rules.

    The collector would keep the cached row and score it under the old reference,
    so the file would contain both rules at once and report one number.
    """
    source = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        collector, "build_provider", lambda args: StubProvider(answers=["Flickr"])
    )

    def _run_once():
        monkeypatch.setattr(sys, "argv", _argv([
            "--data", str(source), "--out", str(out),
            "--model-path", str(tmp_path), "--no-ocr", "--k", "1", "--save-every", "1",
        ]))
        return collector.main()

    source.write_text(json.dumps(FACTUAL_ITEM) + "\n", encoding="utf-8")
    assert _run_once() == 0

    revised = dict(FACTUAL_ITEM, gold_answers=["Nightclub"])
    source.write_text(json.dumps(revised) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="cached gold_answers differs"):
        _run_once()


def test_a_generated_answer_captures_the_internals(monkeypatch, tmp_path):
    """No answer in, so the primary call happens and its internals are recorded."""
    records = [dict(item) for item in ITEMS]
    rows = _run(monkeypatch, tmp_path, [RIGHT], records)

    for row in rows:
        primary = row["evidence"]["primary"]
        assert primary["sequence_confidence"] is not None
        assert primary["visual_attention_mass"] is not None


def test_a_generated_answer_is_labelled_against_the_gold(monkeypatch, tmp_path):
    rows = _run(monkeypatch, tmp_path, [WRONG], [dict(item) for item in ITEMS])
    assert [row["answer"] for row in rows] == [WRONG, WRONG]
    assert [row["label"] for row in rows] == [1, 1]


def test_a_response_labelled_rule_generates_the_answer_and_defers_the_label(
    monkeypatch, tmp_path, capsys
):
    """CHAIR's label reads the caption, so it cannot exist before the caption does.

    The third kind of benchmark: not "compare against a reference", not "supply
    the published answer", but "an external scorer reads the response". The
    collector must generate the response here -- that is the only call whose
    internals are read -- leave the question alone (a caption benchmark is not a
    short-answer one), and hand the labelling to the pass that owns the rule.
    """
    records = [
        {"id": "c1", "question": "Please describe this image in detail.",
         "image": "/a.png", "label_from_response": True},
        {"id": "c2", "question": "Please describe this image in detail.",
         "image": "/b.png", "label_from_response": True},
    ]
    rows = _run(monkeypatch, tmp_path, ["A toilet in a bathroom."], records)
    out = capsys.readouterr().out

    # the answer was generated, so the internals are there
    for row in rows:
        assert row["answer"] == "A toilet in a bathroom."
        assert row["evidence"]["primary"]["sequence_confidence"] is not None
        assert row["evidence"]["primary"]["visual_attention_mass"] is not None
        # and no label was invented
        assert row.get("label") is None
    assert "carry no label yet" in out
    assert "label_from_response.py" in out
    # a file with pending labels is not a single-class run in disguise
    assert "single-class" not in out


def test_a_deferred_label_does_not_silence_the_missing_label_refusal(
    monkeypatch, tmp_path
):
    """The escape hatch is the flag, not the absence of a label."""
    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"id": "f1", "question": "q", "image": "/a.png"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        collector, "build_provider", lambda args: StubProvider(answers=[RIGHT])
    )
    monkeypatch.setattr(sys, "argv", [
        "collect_evidence.py", "--data", str(source), "--out", str(tmp_path / "o.jsonl"),
        "--model-path", str(tmp_path), "--no-ocr", "--k", "1", "--save-every", "1",
    ])
    with pytest.raises(SystemExit, match="neither 'answer' nor 'gold_answers'"):
        collector.main()


def test_a_record_needs_an_answer_or_gold(monkeypatch, tmp_path):
    """Without either, the record cannot be labelled, and guessing is worse."""
    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"id": "f1", "question": "q", "image": "/a.png"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        collector, "build_provider", lambda args: StubProvider(answers=[RIGHT])
    )
    monkeypatch.setattr(sys, "argv", [
        "collect_evidence.py", "--data", str(source), "--out", str(tmp_path / "o.jsonl"),
        "--model-path", str(tmp_path), "--no-ocr", "--k", "1",
    ])
    with pytest.raises(SystemExit, match="neither 'answer' nor 'gold_answers'"):
        collector.main()


def test_collection_resumes_without_regenerating(monkeypatch, tmp_path):
    """Evidence already on disk is copied through, not recomputed."""
    rows = _run(monkeypatch, tmp_path, [RIGHT], [dict(item) for item in ITEMS])
    kept = {row["id"]: row["evidence"] for row in rows}

    again = _run(monkeypatch, tmp_path, [WRONG], [dict(item) for item in ITEMS])
    assert {row["id"]: row["evidence"] for row in again} == kept
    assert [row["answer"] for row in again] == [RIGHT, RIGHT], "not regenerated"


def test_a_provider_without_load_is_accepted(monkeypatch, tmp_path):
    """``load`` is optional in the protocol, so the loop must not require it."""

    class Lazy(StubProvider):
        load = None  # type: ignore[assignment]

    monkeypatch.setattr(collector, "build_provider", lambda args: Lazy(answers=[RIGHT]))
    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"id": "f1", "question": "q", "image": "/a.png",
                    "gold_answers": GOLD}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", [
        "collect_evidence.py", "--data", str(source), "--out", str(tmp_path / "o.jsonl"),
        "--model-path", str(tmp_path), "--no-ocr", "--k", "1",
    ])
    assert collector.main() == 0


def test_a_single_class_run_is_called_out(monkeypatch, tmp_path, capsys):
    """Every answer correct, so every label 0, so `evaluate.py` will refuse the file.

    The collection has already finished by the time that is discovered, which cost
    a 40-minute GPU run once. It costs one line to say so at collection time.
    """
    rows = _run(monkeypatch, tmp_path, [RIGHT], [dict(item) for item in ITEMS])
    assert [row["label"] for row in rows] == [0, 0]

    out = capsys.readouterr().out
    assert "single-class labels" in out
    assert "every item was labelled 0" in out
    assert "scripts/evaluate.py will refuse this file" in out


def test_a_two_class_run_is_not_warned_about(monkeypatch, tmp_path, capsys):
    """A mix of correct and incorrect answers must stay quiet.

    The stub answers every open question with the same text, so the labels come
    from the gold: `Right` against `GOLD` is correct, against a different set of
    accepted answers it is not.
    """

    class Right(StubProvider):
        def generate(self, image, prompt, do_sample=False):
            if "yes or no" in prompt.lower():
                return super().generate(image, prompt, do_sample)
            self.calls += 1
            from msrc.types import Sample

            return Sample(text=RIGHT, sequence_confidence=0.8)

    items = [
        {"id": "f1", "question": "q1", "image": "/a.png", "gold_answers": GOLD},
        {"id": "f2", "question": "q2", "image": "/b.png",
         "gold_answers": ["Nightclub", "Bar", "Pub"]},
    ]
    monkeypatch.setattr(collector, "build_provider", lambda args: Right(answers=[RIGHT]))
    source = tmp_path / "in.jsonl"
    source.write_text(
        "\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", [
        "collect_evidence.py", "--data", str(source), "--out", str(tmp_path / "o.jsonl"),
        "--model-path", str(tmp_path), "--no-ocr", "--k", "1", "--save-every", "1",
    ])
    assert collector.main() == 0
    assert "single-class labels" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Which model backend
# ---------------------------------------------------------------------------

def test_the_provider_flag_picks_the_architecture(tmp_path):
    """Two architectures, one signal set. Nothing downstream knows which ran."""
    from msrc.providers import HFLLaVAProvider, HFQwenVLProvider

    def args_for(provider):
        return argparse.Namespace(
            provider=provider, model_path=str(tmp_path), no_ocr=True,
            max_new_tokens=8, seed=1, want_attention=False,
        )

    assert isinstance(collector.build_provider(args_for("llava")), HFLLaVAProvider)
    qwen = collector.build_provider(args_for("qwen"))
    assert isinstance(qwen, HFQwenVLProvider)
    assert qwen.dtype == "bfloat16", "the precision this checkpoint needs"


def test_the_internvl_provider_targets_the_native_class(tmp_path):
    """The ``-hf`` conversion is a different checkpoint, not a different name.

    The original InternVL releases declare ``model_type: internvl_chat`` and ship
    their own modelling code behind ``auto_map``. Under this transformers that code
    cannot load at all -- ``InternVLChatModel`` has no ``all_tied_weights_keys`` --
    and its tokenizer wants a ``sentencepiece`` older than 0.2 and its vision tower
    wants ``timm``. The ``-hf`` conversions are ``model_type: internvl`` and are
    implemented inside transformers, so this provider points at those. Getting it
    wrong fails at ``from_pretrained``, which is the loud outcome; the quiet one
    would be a model that loads with most of its weights newly initialised.
    """
    from msrc.providers import HFInternVLProvider

    args = argparse.Namespace(
        provider="internvl", model_path=str(tmp_path), no_ocr=True,
        max_new_tokens=8, seed=1, want_attention=False,
    )
    provider = collector.build_provider(args)
    assert isinstance(provider, HFInternVLProvider)
    assert provider.model_class_name == "InternVLForConditionalGeneration"
    assert "image_token_id" in provider.image_token_fields
    second = collector.build_second_provider(
        argparse.Namespace(
            provider="internvl", model_path=str(tmp_path), no_ocr=True,
            max_new_tokens=8, seed=1, want_attention=False, dtype="",
            second_provider="internvl", second_model_path=str(tmp_path),
            second_device="cpu",
        )
    )
    assert isinstance(second, HFInternVLProvider)
    assert second.want_attention is False


def test_the_dtype_flag_overrides_the_default(tmp_path):
    """A CPU run has to be able to leave fp16 behind.

    Both defaults are GPU precisions. On CPU, fp16 is refused or slower than fp32,
    so without this the collector can only ever run where a GPU is free -- and on a
    shared machine that is the one thing that is not a given.
    """
    def args_for(**over):
        base = dict(
            provider="llava", model_path=str(tmp_path), no_ocr=True,
            max_new_tokens=8, seed=1, want_attention=False, device="cpu",
        )
        base.update(over)
        return argparse.Namespace(**base)

    assert collector.build_provider(args_for()).dtype == "float16", "the default stands"
    assert collector.build_provider(args_for(dtype="")).dtype == "float16", "empty means default"
    cpu = collector.build_provider(args_for(dtype="float32"))
    assert cpu.dtype == "float32"
    assert cpu.device == "cpu"


def test_the_pixel_budget_reaches_both_model_roles(tmp_path):
    """The budget has to be one number for the whole cell, not one per role.

    `cross_model_agreement` compares the primary's answer with the second model's.
    If the second model saw a different resolution than it would as a primary, the
    column measures two different things depending on which cell it sits in, and a
    cell that needed the cap would not be comparable with one that did not.
    """
    def args_for(**over):
        base = dict(
            provider="qwen", model_path=str(tmp_path), no_ocr=True, dtype="",
            max_new_tokens=8, seed=1, want_attention=False, device="cpu",
            max_pixels=0, second_provider="", second_model_path="",
        )
        base.update(over)
        return argparse.Namespace(**base)

    assert collector.build_provider(args_for()).max_pixels is None, "0 means the model's own budget"

    capped = collector.build_provider(args_for(max_pixels=1280 * 28 * 28))
    assert capped.max_pixels == 1280 * 28 * 28

    second = collector.build_second_provider(
        args_for(second_provider="qwen", second_model_path=str(tmp_path),
                 second_device="cpu", max_pixels=1280 * 28 * 28)
    )
    assert second is not None
    assert second.max_pixels == 1280 * 28 * 28
    assert second.want_attention is False, "the second model never reads attention"


def test_an_unknown_provider_is_refused(tmp_path):
    args = argparse.Namespace(
        provider="gpt", model_path=str(tmp_path), no_ocr=True,
        max_new_tokens=8, seed=1, want_attention=False,
    )
    with pytest.raises(SystemExit, match="unknown --provider"):
        collector.build_provider(args)



def test_no_second_model_means_no_cross_model_channel(tmp_path):
    """The default must not quietly attach a model nobody asked for."""
    args = argparse.Namespace(
        provider="llava", model_path=str(tmp_path), no_ocr=True,
        max_new_tokens=8, seed=1, want_attention=False,
        second_model_path="", second_provider="qwen",
    )
    assert collector.build_second_provider(args) is None


def test_a_second_model_is_built_without_attention_or_ocr(tmp_path):
    """It answers the question once, so attention on it would be paid for nothing."""
    from msrc.providers import HFQwenVLProvider

    args = argparse.Namespace(
        provider="llava", model_path=str(tmp_path), no_ocr=True,
        max_new_tokens=8, seed=1, want_attention=True,
        second_model_path="/some/qwen", second_provider="qwen",
    )
    second = collector.build_second_provider(args)
    assert isinstance(second, HFQwenVLProvider)
    assert second.want_attention is False
    assert second.ocr_provider is None
    assert second.model_path == "/some/qwen"


def test_an_unknown_second_provider_is_refused(tmp_path):
    args = argparse.Namespace(
        provider="llava", model_path=str(tmp_path), no_ocr=True,
        max_new_tokens=8, seed=1, want_attention=False,
        second_model_path="/some/model", second_provider="gpt",
    )
    with pytest.raises(SystemExit, match="unknown --second-provider"):
        collector.build_second_provider(args)
