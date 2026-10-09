#!/usr/bin/env python
"""Measure the conformal guarantee instead of asserting it.

`msrc.conformal` ships with ``VALIDATED = False`` and a note saying the procedure is
implemented from the standard definition and unit tested, but has never been
checked against this project's data. This is that check.

The claim under test
--------------------

With calibration items that satisfy the null -- here, answers that are *correct* --
the split-conformal p-values are exchangeable under the null, so Benjamini-Hochberg
on them controls the expected proportion of false discoveries among the selected
items at ``alpha``. A discovery is a rejection, and a rejection means "this looks
riskier than the safe calibration items did", so a **false** discovery is a selected
item that is in fact correct.

What this does
--------------

Draws a fresh three-way group split -- fit the calibrator, calibrate the p-values,
evaluate -- many times, and reports for each ``alpha``:

* the mean realized FDP, which is the quantity the guarantee bounds;
* the share of draws where the realized FDP came in at or under ``alpha``;
* the coverage, because a procedure that selects nothing controls its FDP perfectly
  and is useless.

The honest caveat, printed with the result: calibration and test come from the same
pool here, so exchangeability holds **by construction**. That is the condition a
deployment is most likely to violate, and this validates the implementation, not
any particular deployment's calibration set.

    python scripts/validate_conformal.py --data evidence.jsonl --repeats 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.split import grouped_split  # noqa: E402
from evaluation.groups import image_group_id  # noqa: E402
from msrc.conformal import select  # noqa: E402
from msrc.model import RiskCalibrator  # noqa: E402
from msrc.signals import Evidence, build_report  # noqa: E402

__all__ = ["ALPHA_GRID", "load_evidence", "summarise_draws"]

ALPHA_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)


def load_evidence(path: Path):
    rows = []
    for line in Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    rows = [r for r in rows if r.get("evidence")]
    if not rows:
        raise SystemExit(f"{path} holds no records with evidence")

    matrix, labels, groups = [], [], []
    for row in rows:
        report = build_report(Evidence.from_dict(row["evidence"]))
        matrix.append(report.vector())
        labels.append(int(row["label"]))
        groups.append(image_group_id(row))
    return np.asarray(matrix, dtype=float), np.asarray(labels, dtype=int), np.asarray(groups)


def summarise_draws(draws) -> list:
    """Per-alpha summary from a list of ``(alpha, realized_fdp, coverage)`` draws.

    Split out from the drawing so the arithmetic can be tested without a model.

    Two different means are reported, because conflating them is how a correct
    procedure gets reported as a broken one:

    * ``mean_fdp_over_draws`` counts a draw that selected nothing as an FDP of
      **zero**, which is what the guarantee is about: the false discovery
      *proportion* is defined as 0 when there are no discoveries, and
      ``E[FDP] <= alpha`` is an average over all draws including those.
    * ``mean_fdp_when_selected`` averages only the draws that selected something.
      That is the number a user actually experiences when the procedure fires, and
      it is legitimately larger -- conditioning on "it fired" selects the draws
      with the most marginal p-values. It is not the guarantee, and reporting it as
      one makes a conservative procedure look like a violating one. An earlier
      version of this script did exactly that.

    ``realized_fdp`` is ``nan`` in the input for a draw that selected nothing.
    """
    by_alpha: dict = {}
    for alpha, fdp, coverage in draws:
        by_alpha.setdefault(alpha, []).append((fdp, coverage))

    summary = []
    for alpha in sorted(by_alpha):
        pairs = by_alpha[alpha]
        selected = [(fdp, cov) for fdp, cov in pairs if not np.isnan(fdp)]
        # Empty selections contribute zero to the quantity the guarantee bounds.
        as_zero = [fdp if not np.isnan(fdp) else 0.0 for fdp, _ in pairs]
        summary.append({
            "alpha": alpha,
            "draws": len(pairs),
            "draws_that_selected_nothing": len(pairs) - len(selected),
            "mean_fdp_over_draws": float(np.mean(as_zero)),
            "mean_fdp_when_selected": (
                float(np.mean([fdp for fdp, _ in selected])) if selected else float("nan")
            ),
            "max_fdp_when_selected": (
                float(np.max([fdp for fdp, _ in selected])) if selected else float("nan")
            ),
            "share_of_draws_within_alpha": float(np.mean([fdp <= alpha for fdp in as_zero])),
            "mean_coverage": float(np.mean([cov for _, cov in pairs])),
        })
    return summary


def verdict(summary) -> str:
    """One of ``violated``, ``holds``, ``no_power``.

    A procedure that selects nothing has not broken its guarantee -- it has no
    power, which is a different finding with a different remedy. Treating an
    undefined FDP as a failure would report the most conservative possible
    behaviour as the worst one.
    """
    if not summary:
        return "no_power"
    if all(row["mean_coverage"] == 0.0 for row in summary):
        return "no_power"
    if any(row["mean_fdp_over_draws"] > row["alpha"] for row in summary):
        return "violated"
    return "holds"


def power_condition(alpha: float, m_test: int) -> int:
    """Calibration items needed before any rejection is possible at this alpha.

    Split-conformal p-values are multiples of ``1/(n+1)``, so the smallest one the
    procedure can produce is ``1/(n+1)``. Step-up at level ``alpha`` over ``m``
    tests compares against ``alpha/(m*c_m)`` at the first rank, so nothing can ever
    be rejected unless ``1/(n+1) <= alpha/m`` roughly -- i.e. the null calibration
    set has to be larger than the test set divided by alpha. With a few hundred
    test items and alpha = 0.1 that is thousands of calibration items, which is why
    this is worth stating rather than discovering from an empty selection.
    """
    return int(np.ceil(m_test / max(alpha, 1e-12)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="evidence JSONL")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--fit-fraction", type=float, default=0.4)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.5,
        help="share of the non-fit data given to the conformal calibration set. "
             "The procedure can only reject anything once the null calibration set "
             "is larger than the test set divided by alpha, so raising this is how "
             "you get a regime where the guarantee can actually be measured rather "
             "than vacuously satisfied",
    )
    parser.add_argument("--procedure", default="BY", choices=("BH", "BY"))
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    X, y, groups = load_evidence(Path(args.data))

    draws = []
    skipped = 0
    test_sizes = []
    null_sizes = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        fit_mask, rest_mask = grouped_split(
            groups, dev_fraction=args.fit_fraction, seed=seed
        )
        if len(set(y[fit_mask])) < 2:
            skipped += 1
            continue
        # The calibrator must not see the calibration or test items, or the
        # p-values are computed against a model fitted on the very scores it is
        # judging.
        rest_groups = groups[rest_mask]
        cal_mask_rel, test_mask_rel = grouped_split(
            rest_groups, dev_fraction=args.calibration_fraction, seed=seed + 1
        )
        cal_index = np.flatnonzero(rest_mask)[cal_mask_rel]
        test_index = np.flatnonzero(rest_mask)[test_mask_rel]

        calibrator = RiskCalibrator(l2=args.l2).fit(X[fit_mask], y[fit_mask])
        cal_scores = calibrator.predict_proba(X[cal_index])
        test_scores = calibrator.predict_proba(X[test_index])
        y_cal, y_test = y[cal_index], y[test_index]

        # The null set is the *correct* answers: the p-value asks how unusual a
        # score is among items that are not risky.
        null_scores = cal_scores[y_cal == 0]
        if null_scores.size < 5:
            skipped += 1
            continue

        test_sizes.append(len(test_index))
        null_sizes.append(int(null_scores.size))

        for alpha in ALPHA_GRID:
            accepted = select(null_scores, test_scores, alpha=alpha, procedure=args.procedure)
            selected = int(accepted.sum())
            if selected == 0:
                draws.append((alpha, float("nan"), 0.0))
                continue
            fdp = float(((y_test == 0) & accepted).sum()) / selected
            draws.append((alpha, fdp, selected / len(y_test)))

    if not draws:
        raise SystemExit(
            "no usable draws: every repeat produced a single-class side or too few "
            "correct calibration items. The file is too small or too imbalanced."
        )

    summary = summarise_draws(draws)

    print("=" * 84)
    print("Conformal selective prediction: measured against the guarantee")
    print("=" * 84)
    print(f"  procedure        : {args.procedure}")
    print(f"  repeats used     : {args.repeats - skipped} of {args.repeats}"
          + (f"  ({skipped} skipped: single-class side or too few null items)" if skipped else ""))
    print(f"  n                : {len(y)}")
    mean_test = int(np.mean(test_sizes)) if test_sizes else 0
    mean_null = int(np.mean(null_sizes)) if null_sizes else 0
    print(f"  per draw         : ~{mean_test} test items, ~{mean_null} null calibration items")
    print()
    print(f"  {'alpha':>6} {'E[FDP]':>9} {'FDP|fired':>10} {'max':>7} "
          f"{'<= alpha':>9} {'coverage':>9} {'fired nothing':>14}")
    for row in summary:
        conditional = row["mean_fdp_when_selected"]
        print(f"  {row['alpha']:6.2f} {row['mean_fdp_over_draws']:9.4f} "
              f"{conditional:10.4f} {row['max_fdp_when_selected']:7.4f} "
              f"{row['share_of_draws_within_alpha']:9.1%} {row['mean_coverage']:9.1%} "
              f"{row['draws_that_selected_nothing']:14d}")
    print()
    print("  E[FDP] 是把「一个都没选」记作 0 之后的均值 —— **保证约束的就是这一列**。")
    print("  FDP|fired 是只在「选了东西」的那些次里算的均值；它天然更大，因为条件在")
    print("  「程序响了」上等于挑走了 p 值最边缘的那些次。它是用户真正会遇到的数，")
    print("  但它不是保证，拿它去比 alpha 会把一个保守的程序判成违规。")
    print()

    outcome = verdict(summary)
    print("  一个拒绝要成为可能，null 校准集需要多大（约 n_test / alpha）:")
    for row in summary:
        needed = power_condition(row["alpha"], max(mean_test, 1))
        print(f"    alpha={row['alpha']:.2f}  需要约 {needed} 个 null 校准项"
              f"（本次约 {mean_null} 个）")
    print()
    if outcome == "no_power":
        print("  结论: 选不出任何条目 —— 不是保证被违反，而是这个样本量下没有功效。")
        print("        保证在这种情况下是空真（vacuously true），一个从不选择的程序")
        print("        其 FDP 永远为 0，这是保守到没用，不是控制得好。")
    elif outcome == "violated":
        print("  结论: 存在 mean FDP 高于 alpha 的档位 —— 保证未成立，实现需要复查。")
    else:
        print("  结论: 各档 mean FDP 均不高于 alpha —— 保证在本次测量中成立。")
    print()
    print("  这个测量验证的是实现，不是任何一次部署的校准集。这里的校准集与测试集")
    print("  同源，所以可交换性是由构造保证的 —— 那恰恰是部署中最容易被破坏的前提。")
    print()

    if args.json:
        print(json.dumps({
            "procedure": args.procedure,
            "repeats_used": args.repeats - skipped,
            "repeats_skipped": skipped,
            "n_items": int(len(y)),
            "per_alpha": summary,
            "outcome": outcome,
            "guarantee_holds": outcome == "holds",
            "exchangeable_by_construction": True,
        }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
