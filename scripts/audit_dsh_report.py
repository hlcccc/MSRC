#!/usr/bin/env python
"""Reproduce DSH's report and audit corrected image-disjoint POPE evaluation.

This reads archived evidence only, never runs inference or modifies source data.
Bootstrap intervals condition on the fitted model and resample entire images.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.datasets.pope import parse_yes_no_official
from evaluation.groups import image_group_id
from evaluation.metrics import auroc, brier, ece, threshold_metrics
from evaluation.split import grouped_split
from msrc.model import RiskCalibrator
from msrc.signals import Evidence, build_report

SEED = 20260920
GRID = np.round(np.linspace(.05, .95, 91), 4)


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def matrix(rows):
    reports = [build_report(Evidence.from_dict(row["evidence"])) for row in rows]
    names = reports[0].names()
    assert all(report.names() == names for report in reports)
    return names, np.asarray([report.vector() for report in reports], float)


def pick(scores, labels):
    values = [float(((scores >= threshold) == labels).mean()) for threshold in GRID]
    return float(GRID[np.flatnonzero(values == np.max(values))[-1]])


def image_ci(correct, groups, *, n_boot=5000, seed=SEED):
    unique, index = np.unique(groups, return_inverse=True)
    total = np.bincount(index).astype(float)
    hits = np.bincount(index, weights=correct.astype(float))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), (n_boot, len(unique)))
    values = hits[draws].sum(axis=1) / total[draws].sum(axis=1)
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def fit(X, labels):
    model = RiskCalibrator(l2=.05).fit(X, labels)
    if not model.success_:
        raise RuntimeError("nonconvergent calibrator; no results may be published")
    return model


def run(rows, labels, *, grouping="canonical", seed=SEED, three_way=False,
        bootstrap=True):
    names, X = matrix(rows)
    groups = np.asarray([image_group_id(row) if grouping == "canonical" else
                         str(row.get("image", row.get("id", ""))) for row in rows])
    canonical = np.asarray([image_group_id(row) for row in rows])
    dev, test = grouped_split(groups, seed=seed)
    train, threshold_dev = dev, dev
    if three_way:
        train = dev
        unique = np.unique(groups[test])
        np.random.default_rng(seed + 1).shuffle(unique)
        threshold_dev = np.isin(groups, unique[:len(unique) // 2])
        test = ~(train | threshold_dev)
    y = np.asarray(labels, int)
    for side in (train, threshold_dev, test):
        if len(np.unique(y[side])) < 2:
            raise ValueError("single-class partition")
    model = fit(X[train], y[train])
    sd = model.predict_proba(X[threshold_dev])
    st = model.predict_proba(X[test])
    threshold = pick(sd, y[threshold_dev])
    metrics = threshold_metrics(y[test], st, threshold)
    majority_label = int(y[train].mean() >= .5)
    constant = np.full(test.sum(), y[train].mean())
    per = {name: auroc(y[train], X[train, i]) for i, name in enumerate(names)
           if np.ptp(X[train, i]) > 1e-12}
    best = max(per, key=lambda name: abs(per[name] - .5))
    j = names.index(best)
    raw = X[test, j] if per[best] >= .5 else 1 - X[test, j]
    raw = np.clip(raw, 0, 1)
    single = fit(X[train, [j]][:, None], y[train])
    single_scores = single.predict_proba(X[test, [j]][:, None])
    eb = ece(y[test], raw, n_bins=15)
    ea = ece(y[test], st, n_bins=15)
    single_ece = ece(y[test], single_scores, n_bins=15)
    out = {
        "n": len(rows), "groups": len(np.unique(groups)),
        "n_fit": int(train.sum()), "n_threshold_dev": int(threshold_dev.sum()),
        "n_test": int(test.sum()), "n_test_images": len(np.unique(canonical[test])),
        "seed": seed, "grouping": grouping, "three_way": three_way,
        "fit_test_shared_images": len(set(canonical[train]) & set(canonical[test])),
        "converged": model.success_, "gradient_norm": model.grad_norm_,
        "baseline_signal": best, "ece_before": eb, "ece_after": ea,
        "ece_relative_gain": (eb - ea) / eb,
        "ece_by_bins": {str(n): {"before": ece(y[test], raw, n_bins=n),
                                  "after": ece(y[test], st, n_bins=n)} for n in (5, 10, 15, 20)},
        "auroc": auroc(y[test], st), "brier": brier(y[test], st),
        "threshold_metrics": metrics,
        "fixed_threshold_05_metrics": threshold_metrics(y[test], st, .5),
        "full_corpus_positive_rate": float(y.mean()),
        "always_no_warning_accuracy": float((y[test] == 0).mean()),
        "dev_selected_majority_accuracy": float((y[test] == majority_label).mean()),
        "constant_dev_prior": {"probability": float(y[train].mean()),
                                "ece": ece(y[test], constant),
                                "brier": brier(y[test], constant), "auroc": .5},
        "calibrated_single_control": {"ece": single_ece,
                                      "brier": brier(y[test], single_scores),
                                      "auroc": auroc(y[test], single_scores),
                                      "fusion_ece_gain": (single_ece - ea) / single_ece},
        "live_signals": [name for i, name in enumerate(names) if np.ptp(X[:, i]) > 1e-12],
        "missing_counts": dict(Counter(name for row in rows
                              for name in build_report(Evidence.from_dict(row["evidence"])).missing())),
    }
    if bootstrap:
        correct = (st >= threshold) == y[test]
        out["accuracy_image_cluster_ci95"] = image_ci(correct, canonical[test])
        # Reproduce the old row bootstrap to distinguish it from image bootstrap.
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(correct), (5000, len(correct)))
        out["accuracy_row_ci95"] = [float(v) for v in np.percentile(correct[draws].mean(1), [2.5, 97.5])]
    return out


def repeats(rows, labels, n=200):
    _, X = matrix(rows)
    groups = np.asarray([image_group_id(row) for row in rows])
    y = np.asarray(labels)
    values = []
    for seed in range(SEED, SEED + n):
        dev, test = grouped_split(groups, seed=seed)
        model = fit(X[dev], y[dev])
        threshold = pick(model.predict_proba(X[dev]), y[dev])
        values.append(float(((model.predict_proba(X[test]) >= threshold) == y[test]).mean()))
    return {"n": n, "median": float(np.median(values)), "min": min(values),
            "max": max(values), "fraction_ge_85": float(np.mean(np.asarray(values) >= .85)),
            "caveat": "overlapping resplits of one corpus, not 200 independent tests"}


def official_coverage(rows, refs):
    out = {}
    for split in ("random", "popular", "adversarial"):
        official = load(refs / f"coco_pope_{split}.json")
        lookup = {(image_group_id({"dataset": "POPE", "image": row["image"]}), row["text"]):
                  row["label"] for row in official}
        subset = [row for row in rows if row["split"] == split]
        matched, conflicts, unmatched = 0, [], []
        for row in subset:
            key = image_group_id(row), row["question"]
            if key not in lookup:
                unmatched.append({"id": row["id"], "question": row["question"]})
                continue
            matched += 1
            if row["gold_answers"][0] != lookup[key]:
                conflicts.append(row["id"])
        y_object = np.asarray([int(row["gold_answers"][0] == "yes") for row in subset])
        predicted = np.asarray([int(parse_yes_no_official(row["answer"]) == "yes") for row in subset])
        answer_metrics = threshold_metrics(y_object, predicted, .5)
        out[split] = {"n": len(subset), "paper_n_questions": len(official),
                      "paper_n_images": len({row["image"] for row in official}),
                      "fraction_questions": len(subset) / len(official),
                      "matching_official_questions": matched, "gold_conflicts": conflicts,
                      "unmatched": unmatched, "yes_gold": int(y_object.sum()),
                      "paper_answer_metrics": answer_metrics,
                      "note": "answer accuracy is not MSRC warning accuracy"}
    return out


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT.parent)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scorer-out", type=Path)
    parser.add_argument("--http-check", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    pope = workspace / "evidence/msrc_dsh_audit/original/evidence_pope_full1500.jsonl"
    refs = workspace / "evidence/reference_code"
    rows = load(pope)
    archived = np.asarray([row["label"] for row in rows], int)
    official = np.asarray([int(parse_yes_no_official(row["answer"]) != row["gold_answers"][0])
                           for row in rows], int)
    result = {"source_sha256": {str(pope): digest(pope)},
              "dsh_base_commit": "c042554", "raw_rows": len(rows),
              "distinct_paths": len({row["image"] for row in rows}),
              "actual_images": len({image_group_id(row) for row in rows}),
              "duplicate_item_keys": len(rows) - len({(r["id"], r["image"], r["question"]) for r in rows}),
              "official_parser_changed_labels": int((official != archived).sum()),
              "pope_coverage": official_coverage(rows, refs), "pope": {}}
    for name, labels, grouping in (("historical_reproduction", archived, "path"),
                                  ("image_disjoint_archived_labels", archived, "canonical"),
                                  ("image_disjoint_official_labels", official, "canonical")):
        result["pope"][name] = {}
        for split in ("ALL", "adversarial", "random", "popular"):
            indices = np.arange(len(rows)) if split == "ALL" else np.asarray(
                [i for i, row in enumerate(rows) if row["split"] == split])
            result["pope"][name][split] = run([rows[i] for i in indices], labels[indices], grouping=grouping)
        print(name, json.dumps(result["pope"][name]["ALL"]["threshold_metrics"]))
    result["pope"]["three_way_official_labels"] = run(rows, official, three_way=True)
    result["pope"]["resplit_sensitivity"] = {}
    for split in ("ALL", "adversarial"):
        indices = np.arange(len(rows)) if split == "ALL" else np.asarray(
            [i for i, row in enumerate(rows) if row["split"] == split])
        result["pope"]["resplit_sensitivity"][split] = repeats([rows[i] for i in indices], official[indices])
    spec = importlib.util.spec_from_file_location("reference_textvqa", refs / "m4c_evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator = module.TextVQAAccuracyEvaluator()
    text_path = workspace / "MSRC/artifacts/evidence_textvqa.jsonl"
    text_rows = load(text_path)
    text_scores = np.asarray([evaluator._compute_answer_scores(row["gold_answers"]).get(
        evaluator.answer_processor(row["answer"]), 0.) for row in text_rows])
    text_labels = (text_scores < .5).astype(int)
    result["source_sha256"][str(text_path)] = digest(text_path)
    result["textvqa"] = {"n": len(text_rows), "official_soft_answer_accuracy": float(text_scores.mean()),
                         "changed_labels": int((text_labels != np.asarray([r["label"] for r in text_rows])).sum()),
                         "official_softscore_risk_cutoff": .5,
                         "archived": run(text_rows, [row["label"] for row in text_rows], grouping="path"),
                         "official_scored": run(text_rows, text_labels)}
    safety_path = workspace / "MSRC/artifacts/evidence_safety_qwen.jsonl"
    safety_rows = load(safety_path)
    result["source_sha256"][str(safety_path)] = digest(safety_path)
    result["safety"] = {"label_semantics": "keyword non-refusal proxy, NOT paper ASR or verified violations",
                        "n": len(safety_rows),
                        "non_refusal_proxy_rate": float(np.mean([r["label"] for r in safety_rows])),
                        "evaluation": run(safety_rows, [r["label"] for r in safety_rows]),
                        "scenario_judge_completed": False}
    if args.scorer_out:
        from msrc.pipeline import MSRCConfig, MSRCPipeline
        names, features = matrix(rows)
        groups = np.asarray([image_group_id(row) for row in rows])
        dev, test = grouped_split(groups, seed=SEED)
        threshold = result["pope"]["image_disjoint_official_labels"]["ALL"]["threshold_metrics"]["threshold"]
        pipeline = MSRCPipeline(config=MSRCConfig(k=3, threshold=threshold))
        pipeline.fit_from_signals(features[dev], official[dev], names)
        pipeline.provenance.update({"source_sha256": digest(pope),
                                    "image_grouping": "canonical COCO image ID",
                                    "fit_seed": SEED, "label_rule": "official POPE parser",
                                    "calibrator_converged": pipeline.calibrator.success_,
                                    "gradient_norm": pipeline.calibrator.grad_norm_,
                                    "scope": "POPE 1500-question pilot, not official full scale"})
        pipeline.save(args.scorer_out)
        loaded = MSRCPipeline.load(args.scorer_out)
        predictions = loaded.calibrator.predict_proba(features[test])
        diff = float(np.max(np.abs(predictions - pipeline.calibrator.predict_proba(features[test]))))
        if diff > 1e-12:
            raise RuntimeError("saved scorer changed predictions")
        pred_path = args.out.parent / "predictions_pope_canonical.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "image_id", "split", "question",
                                    "answer", "gold", "risk_label", "risk_probability",
                                    "threshold", "warning", "correct_warning"])
            writer.writeheader()
            for index, probability in zip(np.flatnonzero(test), predictions):
                row = rows[index]
                writer.writerow({"id": row["id"], "image_id": groups[index], "split": row["split"],
                                 "question": row["question"], "answer": row["answer"],
                                 "gold": row["gold_answers"][0], "risk_label": int(official[index]),
                                 "risk_probability": float(probability), "threshold": threshold,
                                 "warning": int(probability >= threshold),
                                 "correct_warning": int((probability >= threshold) == official[index])})
        result["runtime"] = {"scorer": str(args.scorer_out), "prediction_rows": len(predictions),
                              "save_load_max_error": diff, "predictions_csv": str(pred_path)}
        if args.http_check:
            from fastapi.testclient import TestClient
            from msrc.service import create_app
            with TestClient(create_app(loaded)) as client:
                health = client.get("/health")
                item = rows[int(np.flatnonzero(test)[0])]
                response = client.post("/v1/msrc/risk", json={"evidence": item["evidence"]})
            assert health.status_code == response.status_code == 200
            assert abs(response.json()["risk_score"] - predictions[0]) <= 1e-6
            result["runtime"]["http"] = {"health_status": health.status_code,
                                         "risk_status": response.status_code,
                                         "risk_score": response.json()["risk_score"],
                                         "uses_real_cached_evidence": True,
                                         "note": "in-process HTTP contract check, no network deployment"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
