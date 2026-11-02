"""Keep numerical evidence and fresh functionality conclusions separate."""
import argparse
from pathlib import Path

from fast_review.common import load, output, relative_path, require, save, sha


def finish(root, approved):
    root = output(root)
    review = load(root / 'review/summary.json')
    require(review['format'] == 'msrc-fast-review-result-v1', 'unknown review result')
    require((root / 'demo/summary.json').is_file(), 'fresh sample demonstration has not completed')
    demo = load(root / 'demo/summary.json')
    require(demo['format'] == 'msrc-fresh-sample-demo-v1'
            and demo['status'] == 'FRESH_SAMPLE_DEMO_COMPLETE'
            and demo['evidence_manifest_sha256'] == review['evidence_manifest_sha256']
            and demo['cached_answers_used'] is False and demo['no_fitting'] is True
            and demo['factual_model_records'] == 18 and demo['safety_requests'] == 8,
            'incomplete or mismatched fresh demonstration')
    for name, digest in demo['files'].items():
        require(sha(relative_path(root / 'demo', name)) == digest, 'fresh demonstration artifact changed')
    factual = load(root / 'demo/factual/demo_results.json')
    require(len(factual) == 18 and min(r['n_actual_signals'] for r in factual) >= 3,
            'fresh signal demonstration does not meet functional requirement')
    require(len(list((root / 'demo/safety').glob('sample_*.json'))) == 8,
            'fresh safety sample records missing')
    status = ('FAST_REVIEW_PASS' if review['numeric_status'] == 'NUMERIC_PASS' else 'FAST_REVIEW_FAIL') \
        if approved else 'REVIEW_COMPLETE_ACCEPTANCE_METHOD_APPROVAL_PENDING'
    result = dict(status=status, full_fresh_inference=False, fresh_sample_demo=True,
        numeric_status=review['numeric_status'], acceptance_method_confirmed=approved,
        review_summary_sha256=sha(root / 'review/summary.json'),
        demo_summary_sha256=sha(root / 'demo/summary.json'),
        scope='full historical evidence review plus fresh fixed-sample demonstration; not full fresh inference')
    save(root / 'summary.json', result)
    text = '# 快速测评总报告\n\n状态：%s\n\n' % status \
        + '全量证据复核：%s。现场抽测：已完成。\n\n' % review['numeric_status'] \
        + '详细数值：review/report.html 与 review/report.md。现场记录：demo/demo_report.md。\n\n' \
        + '本次不是全量重新推理；本结论仅适用于测试方确认的快速测评方式。\n'
    (root / 'report.md').write_text(text, encoding='utf-8', newline='\n')
    import html
    (root / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>MSRC快速测评</title>'
        + '<style>body{font:16px Arial;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}</style>'
        + '<pre>' + html.escape(text) + '</pre><a href="review/report.html">三个指标详细报告</a>', encoding='utf-8')
    print(status)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/fast_001'))
    parser.add_argument('--method-approved', action='store_true',
                        help='set only after the acceptance party confirms the changed method')
    args = parser.parse_args()
    finish(args.run, args.method_approved)


if __name__ == '__main__':
    main()
