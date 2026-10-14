# -*- coding: utf-8 -*-
"""Unreadable signals must not enter the design matrix as "no risk".

A column that is unavailable for *every* item is constant and the L2 penalty
drives its weight to zero -- harmless. A column that is unavailable for *part* of
the items is the dangerous case: the 0.0 that reaches the calibrator is a
low-risk reading, so "could not be read" is silently encoded as "no risk" and the
fit can learn that absence of evidence is evidence of safety. These tests pin the
two policies apart.
"""
from __future__ import annotations

import numpy as np
import pytest

from msrc.pipeline import MSRCPipeline
from msrc.types import SignalReport, SignalValue


def report_of(pairs):
    """Build a SignalReport from ``[(risk, available), ...]``."""
    return SignalReport(signals=[
        SignalValue(name="signal_%d" % i, kind="external", risk=float(risk),
                    available=bool(available))
        for i, (risk, available) in enumerate(pairs)
    ])


def test_fill_values_average_only_the_readings_that_were_available():
    reports = [
        report_of([(0.2, True), (0.9, False)]),
        report_of([(0.8, True), (0.1, True)]),
    ]
    fill = MSRCPipeline._fill_values(reports, "mean")
    # column 0 is readable twice: (0.2 + 0.8) / 2
    assert fill[0] == pytest.approx(0.5)
    # column 1 is readable once; the unavailable 0.9 must not count
    assert fill[1] == pytest.approx(0.1)


def test_zero_policy_produces_no_fill_values():
    reports = [report_of([(0.2, True), (0.9, False)])]
    assert MSRCPipeline._fill_values(reports, "zero") is None


def test_a_fully_unavailable_signal_falls_back_to_zero():
    """Nothing to average -- and the column is constant, so L2 zeroes its weight."""
    reports = [
        report_of([(0.2, True), (0.5, False)]),
        report_of([(0.4, True), (0.7, False)]),
    ]
    assert MSRCPipeline._fill_values(reports, "mean")[1] == 0.0


def test_the_vector_uses_the_fill_for_unreadable_cells_only():
    pipe = MSRCPipeline()
    report = report_of([(0.2, True), (0.9, False)])
    assert pipe._vector(report, None) == [0.2, 0.0]
    assert pipe._vector(report, np.asarray([0.5, 0.7])) == [0.2, 0.7]


def test_a_short_fill_vector_does_not_index_past_its_end():
    pipe = MSRCPipeline()
    report = report_of([(0.2, True), (0.9, False)])
    assert pipe._vector(report, np.asarray([0.5])) == [0.2, 0.0]


def test_missing_policy_survives_config_round_trip():
    from msrc.pipeline import MSRCConfig

    config = MSRCConfig(missing_policy="mean")
    assert MSRCConfig.from_dict(config.to_dict()).missing_policy == "mean"


def test_the_default_policy_is_the_historical_one():
    """Changing the default silently would rewrite every previously reported number."""
    from msrc.pipeline import MSRCConfig

    assert MSRCConfig().missing_policy == "zero"
