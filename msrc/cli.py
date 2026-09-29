"""Command line entry point.

    msrc signals                       what the signal layer defines, and what runs
    msrc demo                          the whole chain on a CPU, no model
    msrc explain   --data ev.jsonl     which signals carry information on this data
    msrc fit       --data ev.jsonl --out scorer.json
    msrc score     --scorer scorer.json --data item.json
    msrc evaluate  --data ev.jsonl [--json]
    msrc version

Two rules the commands follow, both learned from a framework that broke them:

**No silent provider.** Nothing here scores an item without evidence that came
from somewhere real, and anything that would produce a model-free number says so
in its output rather than in a footnote. `msrc demo` is the one exception, and it
prints that its numbers are meaningless.

**No unfitted scoring.** `msrc score` refuses to run against a scorer that has no
fitted coefficients. An uncalibrated prior is not a QACD probability, and returning
one as though it were is how a caller ends up trusting a number that means nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

__all__ = ["main", "build_parser"]

#: Bumped with the package. Reported by `msrc version` so a scorer file and a
#: code version can be matched up when something does not reproduce.
VERSION = "0.1.0"


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise SystemExit(
            f"file not found: {path}\n"
            "  Expected JSONL, one object per line. See docs/evaluation.md for the "
            "two record shapes (with an answer, or with gold_answers to generate one)."
        )
    rows: List[Dict[str, Any]] = []
    # utf-8-sig, because PowerShell's Out-File and Excel both prepend a BOM and the
    # resulting file would otherwise be rejected on its first line.
    for lineno, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msrc",
        description="Multi-signal risk calibration for multimodal generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The score is fitted on your labels, so `fit` and `evaluate` need an "
            "evidence file, not raw answers. See docs/evaluation.md."
        ),
    )
    parser.add_argument("--version", action="version", version=f"msrc {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_signals = sub.add_parser(
        "signals",
        help="list the signal layer, and how many run under a given deployment",
    )
    p_signals.add_argument(
        "--family",
        default="factual",
        choices=["factual", "safety"],
        help="which risk family to report for (they differ by one signal)",
    )
    p_signals.add_argument(
        "--second-model",
        action="store_true",
        help="assume a second model is attached, which enables one more signal",
    )

    sub.add_parser("demo", help="run the whole chain on a CPU with no model")
    sub.add_parser("version", help="print the version")

    p_explain = sub.add_parser(
        "explain",
        help="which signals carry information on a dataset, and what they weigh",
    )
    p_explain.add_argument("--data", required=True, help="evidence JSONL")

    p_fit = sub.add_parser("fit", help="fit a scorer and write it out")
    p_fit.add_argument("--data", required=True, help="evidence JSONL with labels")
    p_fit.add_argument("--out", required=True, help="destination JSON scorer")
    p_fit.add_argument("--l2", type=float, default=0.05)
    p_fit.add_argument("--dev-fraction", type=float, default=0.5)
    p_fit.add_argument("--seed", type=int, default=20260920)

    p_score = sub.add_parser("score", help="score one item from a fitted scorer")
    p_score.add_argument("--scorer", required=True, help="JSON scorer from `msrc fit`")
    p_score.add_argument(
        "--evidence",
        required=True,
        help="JSON file holding one record's evidence "
             "(the `evidence` object from a collection run)",
    )
    p_score.add_argument("--json", action="store_true")

    p_eval = sub.add_parser("evaluate", help="fit, evaluate and report with caveats")
    p_eval.add_argument("--data", required=True, help="evidence JSONL with labels")
    p_eval.add_argument("--dev-fraction", type=float, default=0.5)
    p_eval.add_argument("--seed", type=int, default=20260920)
    p_eval.add_argument("--threshold", type=float, default=0.5)
    p_eval.add_argument("--n-bins", type=int, default=15)
    p_eval.add_argument("--json", action="store_true")

    return parser


def _cmd_signals(args) -> int:
    from msrc.signals import ALL_SIGNALS
    from msrc.types import SIGNAL_INTERNAL

    applied = [s for s in ALL_SIGNALS if s.applies_to(args.family)]
    usable = applied if args.second_model else [
        s for s in applied if s.name != "cross_model_agreement"
    ]
    internal = [s for s in usable if s.kind == SIGNAL_INTERNAL]
    external = [s for s in usable if s.kind != SIGNAL_INTERNAL]

    print(f"Risk family: {args.family}")
    print(f"  defined : {len(ALL_SIGNALS)} signal types")
    print(f"  apply   : {len(applied)}")
    print(f"  run     : {len(usable)}  ({len(internal)} internal + {len(external)} external)")
    print()
    for signal in usable:
        print(f"  {signal.name:24} {signal.kind:9} {signal.description}")
    skipped = [s for s in applied if s not in usable]
    if skipped:
        print()
        print("  not run in this configuration:")
        for signal in skipped:
            print(f"  {signal.name:24} {signal.kind:9} needs a second model")
    print()
    if args.family == "safety":
        print("Note: the safety family scores a policy question, and this framework")
        print("ships no policy taxonomy. What counts as unsafe is a deployment")
        print("decision, and a result computed without one is not reproducible.")
    return 0


def _cmd_demo(_args) -> int:
    script = Path(__file__).resolve().parents[1] / "examples" / "run_stub.py"
    if not script.is_file():
        raise SystemExit(f"demo script not found at {script}")
    import runpy

    runpy.run_path(str(script), run_name="__main__")
    return 0


def _split(rows, dev_fraction: float, seed: int):
    """Split by group (image), so near-duplicates cannot straddle the boundary."""
    import numpy as np

    groups = np.asarray([str(r.get("image", r.get("id", ""))) for r in rows])
    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    cut = int(len(unique) * dev_fraction)
    dev_groups = set(unique[:cut])
    dev_mask = np.asarray([g in dev_groups for g in groups])
    return dev_mask, ~dev_mask


def _matrix(rows):
    """Signal matrix and labels from an evidence file, plus which signals are live."""
    import numpy as np

    from msrc.signals import Evidence, build_report

    names: List[str] = []
    matrix, labels = [], []
    for row in rows:
        if not row.get("evidence"):
            raise SystemExit(
                f"record {row.get('id', '?')!r} has no evidence. Run "
                "scripts/collect_evidence.py first; a scorer cannot be fitted on "
                "answers alone."
            )
        report = build_report(Evidence.from_dict(row["evidence"]))
        if not names:
            names = report.names()
        matrix.append(report.vector())
        labels.append(int(row.get("label", 0)))
    return np.asarray(matrix, dtype=float), np.asarray(labels, dtype=int), names


def _cmd_explain(args) -> int:
    import numpy as np

    from msrc.model import RiskCalibrator
    from msrc.types import SIGNAL_INTERNAL

    rows = _load_jsonl(args.data)
    X, y, names = _matrix(rows)
    if len(set(y.tolist())) < 2:
        raise SystemExit(
            "the labels contain only one class, so nothing can be fitted or "
            "explained. Check the labelling rule."
        )

    internal = set()
    from msrc.signals import ALL_SIGNALS
    for s in ALL_SIGNALS:
        if s.kind == SIGNAL_INTERNAL:
            internal.add(s.name)

    print(f"{len(rows)} items, positive rate {y.mean():.3%}")
    print()
    print("  signal                     kind       range            status")
    live = 0
    for index, name in enumerate(names):
        column = X[:, index]
        spread = float(np.ptp(column))
        kind = "internal" if name in internal else "external"
        status = "carries information" if spread > 1e-12 else "constant -- no weight"
        if spread > 1e-12:
            live += 1
        print(f"  {name:26} {kind:9} {column.min():.3f}..{column.max():.3f}   {status}")
    print()
    n_internal_live = sum(
        1 for i, n in enumerate(names)
        if n in internal and float(np.ptp(X[:, i])) > 1e-12
    )
    print(f"  {live} of {len(names)} signals carry information "
          f"({n_internal_live} internal)")
    if live < len(names):
        print("  A constant signal means the deployment cannot feed it, or the data")
        print("  does not exercise it. Either way it contributes nothing, and the")
        print("  count that matters is the live one.")

    calibrator = RiskCalibrator(l2=0.05).fit(X, y)
    print()
    print("  fitted weights on the full set (largest first):")
    for row in calibrator.weights(names):
        print(f"    {row['signal']:26} {row['weight']:+.4f}")
    return 0


def _cmd_fit(args) -> int:
    from msrc.pipeline import MSRCConfig, MSRCPipeline

    rows = _load_jsonl(args.data)
    dev_mask, _ = _split(rows, args.dev_fraction, args.seed)
    dev_rows = [r for r, keep in zip(rows, dev_mask) if keep]
    if not dev_rows:
        raise SystemExit("the split produced an empty development set")

    X, y, names = _matrix(dev_rows)
    if len(set(y.tolist())) < 2:
        raise SystemExit("the development split has a single class; check the labels")

    pipeline = MSRCPipeline(config=MSRCConfig(l2=args.l2))
    pipeline.fit_from_signals(X, y, names)

    for warning in pipeline.fit_warnings:
        print(f"  ! {warning}", file=sys.stderr)
    path = pipeline.save(args.out)
    print(f"scorer written to {path}")
    print(f"  fitted on     : {len(dev_rows)} items, positive rate {y.mean():.3%}")
    print(f"  signals       : {len(names)} in the matrix")
    print(f"  provenance    : {json.dumps(pipeline.provenance, ensure_ascii=False)}")
    return 0


def _cmd_score(args) -> int:
    from msrc.pipeline import MSRCPipeline

    scorer = Path(args.scorer).expanduser()
    if not scorer.is_file():
        raise SystemExit(f"scorer not found: {args.scorer}")

    payload = json.loads(Path(args.evidence).expanduser().read_text(encoding="utf-8"))
    if "evidence" in payload:
        payload = payload["evidence"]
    if not payload:
        raise SystemExit(f"{args.evidence} holds no evidence object")

    pipeline = MSRCPipeline.load(scorer)
    if not pipeline.fitted_:
        raise SystemExit(
            "this scorer file carries no fitted coefficients. Scoring with it would "
            "return an arbitrary number; re-fit with `msrc fit`."
        )

    result = pipeline.score("", "", "", evidence=payload)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"risk_score            : {result.risk_score:.6f}")
    print(f"calibrated_confidence : {result.calibrated_confidence:.6f}")
    print(f"is_high_risk          : {result.is_high_risk}  (threshold {result.threshold})")
    print(f"risk_family           : {result.risk_family}")
    print()
    for signal in result.signals.signals:
        flag = " " if signal.available else "!"
        print(f"  [{flag}] {signal.name:24} risk={signal.risk:.4f}  {signal.detail}")
    for warning in result.warnings:
        print(f"  ! {warning}")
    return 0


def _cmd_evaluate(args) -> int:
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    if not script.is_file():
        raise SystemExit(f"evaluate script not found at {script}")

    # Check the input here rather than letting the child process fail: a traceback
    # from a subprocess is a worse error message than the one this module already
    # knows how to produce.
    _load_jsonl(args.data)

    import subprocess

    cmd = [
        sys.executable, str(script),
        "--data", args.data,
        "--dev-fraction", str(args.dev_fraction),
        "--seed", str(args.seed),
        "--threshold", str(args.threshold),
        "--n-bins", str(args.n_bins),
    ]
    if args.json:
        cmd.append("--json")
    return subprocess.call(cmd)


def _cmd_version(_args) -> int:
    print(f"msrc {VERSION}")
    return 0


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "signals": _cmd_signals,
        "demo": _cmd_demo,
        "explain": _cmd_explain,
        "fit": _cmd_fit,
        "score": _cmd_score,
        "evaluate": _cmd_evaluate,
        "version": _cmd_version,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
