"""Report-only aggregates must not change individual acceptance thresholds."""
from copy import deepcopy

import pytest

from extensions.multilayer import report_artifacts as reports


def cells():
    return [dict(cell=model + '_' + dataset, n_dev=10, n_test=10,
        ece_before=before, ece_after=after, relative_improvement=(before - after) / before,
        minimum_actual_signals_per_test_sample=6)
        for model in ('llava', 'qwen', 'internvl')
        for dataset, before, after in (('pope', .1, .08), ('textvqa', .4, .04))]


def safety():
    return dict(primary=dict(temporal_hold=dict(n=100, TP=70, TN=20, FP=5, FN=5,
        accuracy=.9, always_alarm_accuracy=.75)))


def test_mean_is_mean_of_gains_not_gain_of_mean_ece():
    summary = reports.calibration_summary(cells())
    assert summary['mean_ece_before'] == pytest.approx(.25)
    assert summary['mean_ece_after'] == pytest.approx(.06)
    assert summary['mean_relative_improvement'] == pytest.approx(.55)
    assert summary['mean_relative_improvement'] != pytest.approx(1 - .06 / .25)
    assert summary['minimum_relative_improvement'] == pytest.approx(.20)
    assert summary['passed_cells'] == 6
    assert summary['acceptance_uses_each_cell_not_mean'] is True


@pytest.mark.parametrize('gain', [None, float('nan'), float('inf')])
def test_undefined_cell_is_not_dropped_from_average(gain):
    rows = cells()
    rows[0]['relative_improvement'] = gain
    summary = reports.calibration_summary(rows)
    assert summary['mean_relative_improvement'] is None
    assert summary['minimum_relative_improvement'] is None
    assert summary['valid_gain_cells'] == 5
    assert summary['passed_cells'] == 5
    assert reports.factual_rows(rows)[0][-1] == '未达标'
    assert reports.factual_rows(rows)[-1][6] == '未定义'


def test_incomplete_grid_does_not_claim_six_cell_average():
    assert reports.calibration_summary(cells()[:5])['mean_relative_improvement'] is None


def test_high_average_does_not_hide_failed_cell():
    rows = cells()
    rows[0]['relative_improvement'] = .01
    summary = reports.calibration_summary(rows)
    assert summary['mean_relative_improvement'] > .10
    assert summary['passed_cells'] == 5
    assert reports.factual_rows(rows)[0][-1] == '未达标'
    assert reports.factual_rows(rows)[-1][-1] == '仅展示'


def test_report_does_not_round_individual_gains_before_averaging():
    rows = cells()
    for cell in rows:
        cell['relative_improvement'] = .123456789
    assert reports.calibration_summary(rows)['mean_relative_improvement'] == pytest.approx(.123456789)
    assert reports.factual_rows(rows)[-1][6] == '12.35%'


def test_markdown_has_averages_confusion_counts_and_explicit_evidence_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(reports.common, 'ROOT', tmp_path)
    root = tmp_path / 'runs/new/multilayer'
    (root / 'joint').mkdir(parents=True)
    before = deepcopy(cells())
    reports.write_markdown(root, dict(cells=before), safety(), dict(mode='fitted', verdict='PASS'))
    text = (root / 'joint/report.md').read_text(encoding='utf-8')
    assert '| 六组平均 | 等权 | - | - | 0.2500 | 0.0600 | 55.00% | - | 仅展示 |' in text
    assert '| 全部测试样本 | 100 | 70 | 20 | 5 | 5 | 90.00% |' in text
    assert '高15.00个百分点' in text
    assert 'runs/new/factual/signal_usage' in text
    assert '(../../factual/report.json)' in text
    assert '(../experiments/multilayer_001/records/test)' in text
    assert before == cells()


def test_markdown_preserves_current_safety_source_counts():
    report = safety()
    report['by_source'] = dict(temporal_hold={'fixture': dict(report['primary']['temporal_hold'])})
    text = reports.result_tables_markdown(cells(), report)
    assert '| fixture | 100 | 70 | 20 | 5 | 5 | 90.00% |' in text
    assert '不同分支准确率的平均' in text


def test_failing_safety_result_says_below_not_negative_above():
    assert reports.percentage_point_comparison(.7, .75) == '低5.00个百分点'
    assert reports.percentage_point_comparison(.75, .75) == '高0.00个百分点'


def test_individual_markdown_reports_use_actual_metrics_without_mutating_inputs():
    factual, warning = cells(), safety()
    before = deepcopy((factual, warning))
    rendered = reports.indicator_markdown(factual, warning, context='历史实测汇总，不代表新一轮验收。')
    assert set(rendered) == {'signals', 'calibration', 'warning'}
    for key, number in reports.INDICATORS.items():
        assert rendered[key].startswith('# 指标' + number)
        assert '历史实测汇总' in rendered[key]
        assert '**数值判定：达标**' in rendered[key]
    assert '| LLaVA-1.5-13B | POPE | 10 | 6 | ≥3种 | 达标 |' in rendered['signals']
    assert '六组平均提升率：55.00%；最低提升率：20.00%' in rendered['calibration']
    assert '| 全部测试样本 | 100 | 70 | 20 | 5 | 5 | 90.00% |' in rendered['warning']
    assert '高15.00个百分点' in rendered['warning']
    assert '不是去掉PlugGuard的独立版' in rendered['warning']
    assert '未包含分来源统计' in rendered['warning']
    assert (factual, warning) == before


def test_individual_markdown_does_not_hide_failed_cell_or_warning():
    factual, warning = cells(), safety()
    factual[0]['minimum_actual_signals_per_test_sample'] = 2
    factual[0]['relative_improvement'] = .099999
    warning['primary']['temporal_hold']['accuracy'] = .749999
    rendered = reports.indicator_markdown(factual, warning)
    assert all('**数值判定：未达标**' in text for text in rendered.values())
    assert '达到要求：5/6组' in rendered['calibration']


def test_individual_markdown_incomplete_grid_cannot_claim_pass():
    rendered = reports.indicator_markdown(cells()[:5], safety())
    assert '**数值判定：未完成**' in rendered['signals']
    assert '**数值判定：未完成**' in rendered['calibration']


def test_individual_markdown_preserves_source_confusion_counts():
    warning = safety()
    warning['by_source'] = dict(temporal_hold={'fixture': dict(warning['primary']['temporal_hold'])})
    text = reports.indicator_markdown(cells(), warning)['warning']
    assert '| fixture | 100 | 70 | 20 | 5 | 5 | 90.00% |' in text
    assert '未包含分来源统计' not in text
