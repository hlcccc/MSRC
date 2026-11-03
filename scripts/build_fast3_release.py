"""Package a committed fast3 candidate, with CPU evidence and exact archive QA."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

from build_fast3_docs import validate, ROOT, TITLE


BRANCH = 'codex/frozen-fast-local3x5090-20261103'


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd).decode('utf-8').strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8', newline='\n')


def junit(path):
    root = ET.parse(path).getroot()
    suites = list(root.iter('testsuite'))
    result = {key: sum(int(s.get(key, '0')) for s in suites)
        for key in ('tests', 'failures', 'errors', 'skipped')}
    result['passed'] = result['tests'] - sum(result[k] for k in ('failures', 'errors', 'skipped'))
    result['seconds'] = sum(float(s.get('time', '0')) for s in suites)
    result['skip_reasons'] = sorted({v.get('message', '') for v in root.iter('skipped')})
    if result['failures'] or result['errors'] or not result['passed']:
        raise ValueError('CPU regression evidence is not passing: ' + str(path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--contract-tests', type=Path, required=True)
    parser.add_argument('--fresh-clone', type=Path, required=True)
    parser.add_argument('--environment-evidence', type=Path, required=True)
    parser.add_argument('--dependency-evidence', type=Path, required=True)
    parser.add_argument('--dependency-report', type=Path, required=True)
    parser.add_argument('--dependency-script', type=Path, required=True)
    parser.add_argument('--document-evidence', type=Path, required=True)
    args = parser.parse_args()
    commit = git('rev-parse', 'HEAD')
    if git('branch', '--show-current') != BRANCH or git('status', '--porcelain'):
        raise ValueError('package only the clean committed release branch')
    if git('rev-parse', 'HEAD', cwd=args.fresh_clone) != commit:
        raise ValueError('fresh remote clone does not match the release commit')
    if git('ls-remote', 'origin', 'refs/heads/' + BRANCH).split()[0] != commit:
        raise ValueError('GitHub remote branch is not the release commit')
    full, contract = junit(args.tests), junit(args.contract_tests)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'env_patch/SHA256SUMS.txt').is_file() or not list((out / 'env_patch/wheels').glob('*.whl')):
        raise ValueError('required environment supplement has not been delivered')
    shutil.copyfile(args.environment_evidence, out / 'environment_patch_evidence.json')
    dependency = json.loads(args.dependency_evidence.read_text(encoding='utf-8'))
    if not (dependency['complete'] and dependency['no_packages_installed'] and
            dependency['only_inventory_plus_supplement_used'] and
            dependency['environment']['sys_platform'] == 'linux' and
            dependency['environment']['python_version'] == '3.10' and
            dependency['report_sha256'] == sha(args.dependency_report) and
            dependency['audit_script_sha256'] == sha(args.dependency_script)):
        raise ValueError('offline dependency evidence or its bound sources differ')
    for source, name in ((args.dependency_evidence, 'offline_dependency_audit.json'),
            (args.dependency_report, 'offline_pip_report.json'),
            (args.dependency_script, 'offline_dependency_audit_script.py')):
        shutil.copyfile(source, out / name)
    shutil.copyfile(ROOT / 'docs/FAST3_FLOW.docx', out / (TITLE + '.docx'))
    qa = validate(ROOT / 'docs/FAST3_FLOW.md', out / (TITLE + '.docx'))
    document_evidence = json.loads(args.document_evidence.read_text(encoding='utf-8'))
    if not (document_evidence['word_sha256'] == sha(ROOT / 'docs/FAST3_FLOW.docx') and
            document_evidence['markdown_sha256'] == sha(ROOT / 'docs/FAST3_FLOW.md') and
            document_evidence['template_paragraph_formatting_preserved'] and
            document_evidence['template_unchanged']):
        raise ValueError('Word template evidence does not match the committed flow')
    qa.update(document_evidence)
    for source, name in (
        (ROOT / 'docs/FAST3_FLOW.md', TITLE + '.md'),
        (ROOT / 'docs/FAST3_AUDIT.md', '逐步审计.md'),
        (ROOT / 'docs/FAST3_DELIVERY.md', '交付说明.md'),
        (ROOT / 'resources/FAST3_ASSET_SOURCES.json', 'FAST3_ASSET_SOURCES.json'),
        (ROOT / 'configs/local-models.fast3.json', 'local-models.fast3.json'),
        (args.tests, 'cpu_full_regression.xml'),
        (args.contract_tests, 'cpu_fast3_contract_tests.xml'),
    ):
        shutil.copyfile(source, out / name)
    save(out / 'document_qa.json', qa)
    archive = out / 'MSRC-fast-local3-code-20261103.zip'
    subprocess.run(['git', 'archive', '--format=zip', '--prefix=MSRC/',
        '--output=' + str(archive), commit], cwd=ROOT, check=True)
    expected = set(git('ls-tree', '-r', '--name-only', commit).splitlines())
    found = set()
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('archive CRC failure')
        for item in bundle.infolist():
            if item.is_dir():
                continue
            if not item.filename.startswith('MSRC/'):
                raise ValueError('archive root is not MSRC/')
            relative = item.filename[len('MSRC/'):]
            if relative in found or '..' in Path(relative).parts:
                raise ValueError('unsafe archive member')
            found.add(relative)
            if bundle.read(item) != (args.fresh_clone / relative).read_bytes():
                raise ValueError('ZIP differs from fresh GitHub checkout: ' + relative)
    if found != expected:
        raise ValueError('archive file inventory differs from commit')
    save(out / 'source_archive_verification.json', dict(
        complete=True, commit=commit, remote_readback_commit=commit,
        zip_sha256=sha(archive), files_checked=len(found), all_member_bytes_equal_fresh_remote_clone=True,
        archive_crc_ok=True, archive_root='MSRC/', no_untracked_files_included=True,
        not_a_gpu_execution_result=True))
    base = '53ce328'
    unchanged = ['pipeline/fast3_contract.py', 'pipeline/fast3_factual.py', 'pipeline/fast3_safety.py',
        'pipeline/frozen_fast_runtime_5090.py', 'pipeline/frozen_request_runtime_5090.py',
        'pipeline/frozen_safety_hotpath_5090.py', 'pipeline/frozen_judge_bucket_5090.py',
        'extensions/multilayer/runner.py', 'extensions/multilayer/variants.py']
    for name in unchanged:
        original = subprocess.check_output(['git', 'show', base + ':' + name], cwd=ROOT)
        if original != (ROOT / name).read_bytes():
            raise ValueError('unexpected change to frozen inference/evaluation logic: ' + name)
    save(out / 'local3_code_audit.json', dict(complete=True, base_three_host_commit=git('rev-parse', base),
        local_commit=commit, unchanged_algorithm_files={name: sha(ROOT / name) for name in unchanged},
        changed_paths=git('diff', '--name-only', base, commit).splitlines(),
        scope='dispatcher, explicit CUDA UUID binding, local receipt finalization and delivery only',
        no_gpu_run_claim=True))
    versions = {name: importlib.metadata.version(name) for name in ('pytest', 'torch', 'numpy', 'python-docx')}
    save(out / 'RELEASE.json', dict(
        format='msrc-fast3-delivery-v1', status='CANDIDATE_RELEASE_NOT_THREE_GPU_ACCEPTANCE',
        branch=BRANCH, commit=commit, github='https://github.com/hlcccc/MSRC/tree/' + commit,
        code_zip=archive.name, code_zip_sha256=sha(archive),
        scope=dict(factual_models=['internvl'], pope_test=4500, textvqa_test=2496,
            qwen_role='fresh auxiliary donor only', safety_requests=3360, safety_arms=5,
            safety_logical_reviews=16800, original_six_cell_acceptance=False),
        onsite_training=False, fitted_state='supplied, unchanged; pretrained offline, not newly fitted onsite',
        test_selection=False, physical_three_gpu_validation_completed=False,
        final_metric_pass_not_yet_measured=True, no_five_hour_guarantee=True,
        topology='one_host_three_gpus', launcher='scripts/run_fast3_local.sh',
        launch_command='bash scripts/run_fast3_local.sh --run-id local3_fitted_001 --gpus 0,1,2',
        shared_environment_and_assets=True, uses_ssh=False, uses_result_transport_archives=False,
        runtime_estimate_hours=None, estimate_only=False,
        runtime_note='one-host CPU, RAM and SSD contention has not been timed; old three-host estimate not inherited',
        cpu_evidence=dict(platform=platform.platform(), python=sys.version, versions=versions,
            full_regression=full, fast3_contract_and_synthetic_transport=contract,
            synthetic_transport_is_not_gpu_model_quality_evidence=True),
        document_qa=qa,
        pending=['physical three-5090 full run and final thresholds', 'fresh Linux offline install',
            'native Word pagination render', 'DSH upload/readback until separate verification receipt'],
        large_assets='separate reused-materials share; not contained in code ZIP',
        environment_supplement='env_patch/wheels; evidence in environment_patch_evidence.json; fresh GPU install pending',
        offline_dependency_resolution=dict(complete=True, packages=len(dependency['selected_packages']),
            linux_cp310_pep508_markers=True, cuda_runtime='cu128', not_a_fresh_install_or_cuda_test=True),
        old_single_gpu_results_are_planning_reference_not_this_release_results=True))
    save(out / 'READY.json', dict(ready_for_upload=True, commit=commit,
        delivery_only=True, scientific_acceptance_completed=False,
        destination='/MSRC_fast_local3x5090_20261103'))
    checksums = ''.join(sha(p) + '  ' + p.relative_to(out).as_posix() + '\n' for p in sorted(out.rglob('*'))
        if p.is_file() and p != out / 'SHA256SUMS.txt')
    (out / 'SHA256SUMS.txt').write_text(checksums, encoding='utf-8', newline='\n')
    print(json.dumps(dict(ready=True, commit=commit, files=len(checksums.splitlines()) + 1,
        checksum_manifest_sha256=sha(out / 'SHA256SUMS.txt'), output=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
