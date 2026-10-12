"""The headline table has to be reproducible from inside the repository.

The per-cell numbers behind the acceptance claim used to come from one-off shell
scripts kept outside the repo. A number that can be read but not re-run is a claim,
not a result, so the report now lives in `scripts/balanced_report.py` and these
tests pin the parts of it that would fail silently.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "balanced_report", ROOT / "scripts" / "balanced_report.py"
)
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)

from evaluation.metrics import balanced_accuracy  # noqa: E402


# -- the two rates the acceptance metric is built from ---------------------

def test_rates_are_recall_and_false_positive_rate():
    labels = [1, 1, 0, 0]
    predicted = [1, 0, 1, 0]
    tpr, fpr = br.rates(labels, predicted)
    assert tpr == pytest.approx(0.5)
    assert fpr == pytest.approx(0.5)


def test_rates_average_into_balanced_accuracy():
    labels = [1, 1, 1, 0, 0, 0]
    predicted = [1, 1, 0, 0, 1, 0]
    tpr, fpr = br.rates(labels, predicted)
    assert (tpr + 1 - fpr) / 2 == pytest.approx(
        balanced_accuracy(labels, predicted, 0.5))


def test_never_warning_scores_the_positive_rate_in_accuracy_only():
    labels = [0] * 85 + [1] * 15
    predicted = [0] * 100
    tpr, fpr = br.rates(labels, predicted)
    assert (tpr, fpr) == (0.0, 0.0)
    assert balanced_accuracy(labels, predicted, 0.5) == pytest.approx(0.5)
    assert float((np.asarray(predicted) == np.asarray(labels)).mean()) == pytest.approx(0.85)


# -- threshold selection ---------------------------------------------------

def test_best_threshold_maximises_the_rule_it_is_given():
    rng = np.random.default_rng(0)
    labels = (rng.random(400) < 0.3).astype(int)
    scores = np.clip(0.2 + 0.6 * labels + rng.normal(0, 0.2, 400), 0, 1)
    chosen = br.best_threshold(scores, labels, rule="balanced")
    achieved = balanced_accuracy(labels, scores, chosen)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert achieved >= balanced_accuracy(labels, scores, t) - 1e-12


def test_the_accuracy_rule_maximises_accuracy():
    rng = np.random.default_rng(3)
    labels = (rng.random(400) < 0.15).astype(int)
    scores = np.clip(0.2 + 0.6 * labels + rng.normal(0, 0.2, 400), 0, 1)
    chosen = br.best_threshold(scores, labels, rule="accuracy")
    achieved = float(((scores >= chosen).astype(int) == labels).mean())
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert achieved >= float(((scores >= t).astype(int) == labels).mean()) - 1e-12


def test_the_two_rules_disagree_on_an_unbalanced_benchmark():
    """This is why the rule has to match the reported metric.

    Where positives are rare, the balanced optimum warns far more often than the
    accuracy optimum, and every extra warning is a false positive subtracted from a
    number the majority class already dominates.
    """
    rng = np.random.default_rng(7)
    labels = (rng.random(600) < 0.12).astype(int)
    scores = np.clip(0.15 + 0.7 * labels + rng.normal(0, 0.22, 600), 0, 1)
    acc_t = br.best_threshold(scores, labels, rule="accuracy")
    bal_t = br.best_threshold(scores, labels, rule="balanced")
    assert bal_t < acc_t, "the balanced optimum should warn more, i.e. sit lower"


def test_an_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        br.best_threshold([0.1, 0.9], [0, 1], rule="f1")


def test_best_threshold_prefers_the_higher_of_two_ties():
    """On a tied grid the conservative warning is the one to take."""
    scores = [0.2, 0.8]
    labels = [0, 1]
    chosen = br.best_threshold(scores, labels, grid=np.array([0.2, 0.5, 0.8]))
    assert chosen == pytest.approx(0.8)


# -- the report end to end -------------------------------------------------

def _write_cell(tmp_path: Path, name: str, separable: bool, positive_rate: float = 0.3) -> Path:
    """A tiny evidence file shaped like the real ones."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(200):
        y = int(rng.random() < positive_rate)
        skill = (0.8 if y else 0.2) if separable else 0.5
        value = float(np.clip(skill + rng.normal(0, 0.15), 0, 1))
        rows.append({
            "id": f"x:{i}", "label": y, "image": f"img{i // 2}.jpg",
            "evidence": {
                "question": "q", "answer": "a", "image": f"img{i // 2}.jpg",
                "primary": {"text": "a", "sequence_confidence": value,
                            "sequence_entropy": 1 - value,
                            "visual_attention_mass": 0.5,
                            "first_token_margin": value},
                "views": [{"text": "a"}] * 4,
                "resamples": [{"text": "a"}, {"text": "b"}],
                "verifications": [{"text": "yes"}],
                "cross_model": [{"text": "a"}],
                "ocr_texts": [],
            },
        })
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_a_separable_cell_reports_a_pass(tmp_path, capsys):
    path = _write_cell(tmp_path, "good.jsonl", separable=True)
    assert br.main(["--cell", f"M|D={path}"]) == 0
    out = capsys.readouterr().out
    assert "达标" in out
    assert "上界" in out


def test_an_unseparable_cell_reports_a_failure_with_a_ceiling_beside_it(tmp_path, capsys):
    path = _write_cell(tmp_path, "flat.jsonl", separable=False)
    assert br.main(["--cell", f"M|D={path}"]) == 0
    out = capsys.readouterr().out
    assert "未达" in out
    assert "调阈值救不了" in out


def test_the_json_payload_carries_the_metric_and_the_bar(tmp_path, capsys):
    path = _write_cell(tmp_path, "one.jsonl", separable=True)
    assert br.main(["--cell", f"M|D={path}", "--json"]) == 0
    text = capsys.readouterr().out
    payload = json.loads(text[text.index("{", text.index("{")):])
    assert payload["acceptance_metric"] == "balanced_accuracy"
    assert payload["bar"] == 0.85
    assert payload["ceiling_is_selected_on_test"] is True
    assert set(payload["cells"][0]) >= {
        "tpr", "fpr", "balanced_accuracy", "accuracy", "never_warn_accuracy",
        "auroc", "balanced_accuracy_ceiling", "passes",
    }


def test_a_malformed_cell_spec_is_rejected_loudly():
    with pytest.raises(SystemExit):
        br.main(["--cell", "no-separator"])


def test_a_missing_file_is_reported_and_skipped(tmp_path, capsys):
    assert br.main(["--cell", f"M|D={tmp_path / 'nope.jsonl'}"]) == 0
    assert "文件缺失" in capsys.readouterr().out


def test_the_verdict_hangs_on_plain_accuracy_not_on_balanced(tmp_path, capsys):
    """The acceptance metric is plain accuracy; balanced accuracy is a footnote.

    The failure this guards is concrete: on POPE, warning about nothing scores
    85.4% accuracy and 50% balanced accuracy. Judging on the balanced figure would
    fail a cell that passes, and on a rare-positive cell it would pass one that
    does not.
    """
    path = _write_cell(tmp_path, "skew.jsonl", separable=True, positive_rate=0.12)
    assert br.main(["--cell", f"M|D={path}", "--json"]) == 0
    text = capsys.readouterr().out
    payload = json.loads(text[text.index("{", text.index("{")):])
    cell = payload["cells"][0]
    assert cell["passes"] == (cell["accuracy"] >= 0.85)
    assert "accuracy_ceiling" in cell