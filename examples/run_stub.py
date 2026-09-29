#!/usr/bin/env python
"""Run the whole framework on a CPU, with no model and no weights.

This exists because the interesting question for a new deployment is not "does
the maths work" but "which of my signals are actually alive". The stub produces
plausible readings on every channel, so the script runs the identical data twice:

  * **text-only platform** -- the serving stack returns strings and nothing else.
    The three internal signals and the cross-model signal report themselves
    unavailable; five signals carry the score.
  * **with generation-time internals** -- the platform recorded the logprobs and
    attention of the call that produced the answer. All eight signals run.

The difference between the two runs is the point of the framework.

The numbers this prints are **meaningless**. The stub is not a detector: its
readings are synthetic and its "wrong" answers are marked as such in the text. It
proves the wiring and the signal accounting, nothing more. Real numbers need a
real model and real labels; see docs/README for that path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import format_summary, summarise  # noqa: E402
from msrc import MSRCConfig, MSRCPipeline, Sample, StubProvider  # noqa: E402
from msrc.provider import gather_evidence  # noqa: E402
from msrc.signals import ALL_SIGNALS  # noqa: E402
from msrc.types import RISK_FACTUAL, RISK_SAFETY  # noqa: E402

QUESTION = "what is the website that hosts this photo?"
RIGHT = "Flickr"
WRONG = "Flickr WRONG"          # the stub reads the marker and behaves like an
                                # unsure model: it drifts, resamples disagree,
                                # probes come back negative, logprobs flatten


def primary_for(answer: str, failed: int) -> Sample:
    """The generation the platform already made, with the internals it captured.

    These cannot be recovered after the fact. Re-running the model produces a
    *different* generation, whose logprobs are not the ones behind this answer, so
    a deployment that wants the internal signals has to record them at generation
    time and pass them in.
    """
    if failed:
        return Sample(text=answer, sequence_confidence=0.34, sequence_entropy=2.8,
                      visual_attention_mass=0.12)
    return Sample(text=answer, sequence_confidence=0.88, sequence_entropy=0.50,
                  visual_attention_mass=0.62)


def run(with_internals: bool, n_fit: int = 120, n_eval: int = 200) -> dict:
    provider = StubProvider(answers=(RIGHT,), ocr_text=RIGHT)
    pipeline = MSRCPipeline(provider=provider, config=MSRCConfig(k=5, risk_family=RISK_FACTUAL))

    vectors, labels = [], []
    for i in range(n_fit):
        failed = i % 2
        answer = WRONG if failed else RIGHT
        primary = primary_for(answer, failed) if with_internals else None
        evidence, _ = gather_evidence(
            provider, QUESTION, answer, f"/img{i}.jpg",
            risk_family=RISK_FACTUAL, k=5, primary_sample=primary,
        )
        vectors.append([s.risk for s in pipeline._report_for(evidence).signals])
        labels.append(failed)

    pipeline.fit_from_signals(
        np.asarray(vectors, dtype=float), np.asarray(labels, dtype=int), pipeline.signal_names
    )

    scores, truth = [], []
    for i in range(n_eval):
        failed = i % 2
        answer = WRONG if failed else RIGHT
        result = pipeline.score(
            QUESTION, answer, f"/img{i}.jpg",
            primary_sample=primary_for(answer, failed) if with_internals else None,
        )
        scores.append(result.risk_score)
        truth.append(failed)

    live = [
        name for name in pipeline.signal_names
        if not np.allclose(
            np.asarray(vectors)[:, pipeline.signal_names.index(name)],
            np.asarray(vectors)[0, pipeline.signal_names.index(name)],
        )
    ]
    return {"pipeline": pipeline, "scores": scores, "labels": truth, "live": live}


def main() -> int:
    print("=" * 84)
    print("Signal inventory")
    print("=" * 84)
    for kind, label in (("external", "external"), ("internal", "internal")):
        names = [s.name for s in ALL_SIGNALS if s.kind == kind]
        print(f"  {label:9} {len(names)}: {', '.join(names)}")
    for family in (RISK_FACTUAL, RISK_SAFETY):
        applied = [s for s in ALL_SIGNALS if s.applies_to(family)]
        internal = sum(1 for s in applied if s.kind == "internal")
        print(f"  [{family}] {len(applied)} signals apply: "
              f"{internal} internal + {len(applied) - internal} external")

    results = {}
    for with_internals in (False, True):
        title = ("WITH generation-time internals" if with_internals
                 else "TEXT-ONLY platform")
        print()
        print("=" * 84)
        print(title)
        print("=" * 84)
        outcome = run(with_internals)
        results[title] = outcome
        print(f"  signals carrying information: {len(outcome['live'])} "
              f"-> {', '.join(outcome['live'])}")
        print(format_summary(summarise(outcome["labels"], outcome["scores"])))

    print()
    print("=" * 84)
    print("One scored item, in full")
    print("=" * 84)
    pipeline = results["WITH generation-time internals"]["pipeline"]
    result = pipeline.score(QUESTION, WRONG, "/img.jpg",
                            primary_sample=primary_for(WRONG, 1))
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    print()
    print("=" * 84)
    print("These numbers are meaningless -- the stub is not a detector.")
    print("=" * 84)
    print("What they show is the signal accounting: which channels a deployment")
    print("can actually feed, and what it loses when it cannot feed them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
