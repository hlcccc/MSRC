"""Keep the user's four acceptance variants separate and attest parameter scope."""
from pathlib import Path

from pipeline.common import ROOT, sha


VARIANTS = {
    'factual_no_fit': ('factual', 'fresh_fit'),
    'factual_fitted': ('factual', 'provided'),
    'safety_no_fit': ('safety', 'fresh_fit'),
    'safety_fitted': ('safety', 'provided'),
}


def audit(run, variant):
    from pipeline.acceptance_check import load, require
    run = Path(run)
    domain, mode = VARIANTS[variant]
    factual = run / 'factual'
    protocol = load(factual / 'protocol.json')
    require(protocol.get('parameter_mode') == mode, 'factual parameter mode differs from requested variant')
    if mode == 'provided':
        from pipeline.factual_parameters import load_package
        registry = load(ROOT / 'resources/reproduction/fitted.json')
        require(protocol['parameters_sha256'] == registry['factual_parameters_sha256'],
                'provided factual state is not the published fixed parameter release')
        package = load_package(factual / 'provided_parameters.json', factual, protocol)
        require(sha(factual / 'provided_parameters.json') == protocol['parameters_sha256'],
                'provided parameter file changed')
        for cell, state in package['states'].items():
            require(load(factual / 'calibrators' / (cell + '.json')) == state,
                    'provided calibrator/threshold was changed or refitted')
    if domain == 'safety':
        safety = run / 'safety'
        safety_protocol = load(safety / 'protocol.json')
        threshold = load(safety / 'threshold.json')
        expected = 'development_refit' if mode == 'fresh_fit' else 'loaded_development_frozen_head'
        require(safety_protocol['training_mode'] == expected
                and safety_protocol['development_generated_this_run'] == (mode == 'fresh_fit'),
                'safety training mode differs from requested variant')
        require(bool(threshold.get('checkpoint_reused')) == (mode == 'provided'),
                'safety head/threshold import differs from requested variant')
        require(sha(safety / 'head.pt') == threshold['head_sha256'], 'safety head changed')
        if mode == 'provided':
            require(threshold['head_sha256'] == registry['safety_head_sha256']
                    and threshold['threshold'] == registry['safety_threshold']
                    and threshold.get('origin_threshold_sha256') == registry['safety_threshold_sha256'],
                    'provided safety state is not the published fixed parameter release')
        if mode == 'fresh_fit':
            from pipeline.fresh_training import audit as fresh_audit
            fresh_audit(run)
    return dict(variant=variant, warning_domain=domain, parameter_mode=mode,
                definition='no supplied fitted parameters; fit onsite' if mode == 'fresh_fit'
                           else 'supplied fitted parameters; fresh inference without refitting',
                factual_protocol_sha256=sha(factual / 'protocol.json'),
                factual_report_sha256=sha(factual / 'report.json'),
                warning_report_sha256=sha(run / ('safety' if domain == 'safety' else 'factual_warning') / 'report.json'),
                scope='parameter provenance; final completeness/metrics are independently checked')
