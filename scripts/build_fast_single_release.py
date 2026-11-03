"""Package the committed single-5090 branch, checking CPU, Word and source evidence."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

from build_fast3_docs import ROOT, validate
from build_fast3_release import git, junit, save, sha
from build_fast_single_docs import TITLE

BRANCH = 'codex/frozen-fast-single5090-20261103'
BASE = 'e0eea8f61df3df492b6bf3f8e5509d6717173c98'
NAME = 'MSRC_fast_single5090_20261103'
CORE = ['pipeline/common.py', 'pipeline/fast3_contract.py', 'pipeline/fast3_factual.py', 'pipeline/fast3_safety.py',
    'pipeline/frozen_fast_runtime_5090.py', 'pipeline/frozen_request_runtime_5090.py',
    'pipeline/frozen_safety_hotpath_5090.py', 'pipeline/frozen_judge_bucket_5090.py',
    'extensions/multilayer/runner.py', 'extensions/multilayer/variants.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--contract-tests', type=Path, required=True)
    parser.add_argument('--fresh-clone', type=Path, required=True)
    parser.add_argument('--materials-evidence', type=Path, required=True)
    parser.add_argument('--document-evidence', type=Path, required=True)
    args = parser.parse_args()
    commit = git('rev-parse', 'HEAD')
    if git('branch', '--show-current') != BRANCH or git('status', '--porcelain'):
        raise ValueError('package only the clean committed single-card branch')
    if git('ls-remote', 'origin', 'refs/heads/' + BRANCH).split()[0] != commit:
        raise ValueError('remote branch differs')
    if git('rev-parse', 'HEAD', cwd=args.fresh_clone) != commit:
        raise ValueError('fresh remote clone differs')
    full, focused = junit(args.tests), junit(args.contract_tests)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('preserve an existing delivery; use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    materials = args.materials_evidence
    manifest = materials / 'SHA256SUMS.txt'
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, relative = line.split('  ', 1)
        path = (materials / relative).resolve()
        if not path.is_relative_to(materials.resolve()) or sha(path) != digest:
            raise ValueError('reused material evidence changed: ' + relative)
    shutil.copytree(materials / 'env_patch', out / 'env_patch')
    for name in ['environment_patch_evidence.json', 'offline_dependency_audit.json',
            'offline_pip_report.json', 'offline_dependency_audit_script.py']:
        shutil.copyfile(materials / name, out / name)
    dependency = json.loads((out / 'offline_dependency_audit.json').read_text(encoding='utf-8'))
    if not (dependency['complete'] and dependency['no_packages_installed'] and
            dependency['only_inventory_plus_supplement_used'] and
            dependency['environment']['sys_platform'] == 'linux' and
            dependency['environment']['python_version'] == '3.10' and
            dependency['report_sha256'] == sha(out / 'offline_pip_report.json') and
            dependency['audit_script_sha256'] == sha(out / 'offline_dependency_audit_script.py')):
        raise ValueError('invalid reused dependency resolution evidence')
    word, markdown = ROOT / 'docs/FAST_SINGLE_FLOW.docx', ROOT / 'docs/FAST_SINGLE_FLOW.md'
    qa = validate(markdown, word, 'single_fitted_001', 'one_host_one_gpu')
    supplied_qa = json.loads(args.document_evidence.read_text(encoding='utf-8'))
    if not (supplied_qa['word_sha256'] == sha(word) and supplied_qa['markdown_sha256'] == sha(markdown)
            and supplied_qa['template_unchanged'] and supplied_qa['template_paragraph_formatting_preserved']):
        raise ValueError('Word evidence differs')
    qa.update(supplied_qa)
    for source, name in [(word, TITLE + '.docx'), (markdown, TITLE + '.md'),
            (ROOT / 'docs/FAST_SINGLE_SLURM.md', '单卡Slurm提交.md'),
            (ROOT / 'docs/FAST_SINGLE_AUDIT.md', '逐步审计.md'),
            (ROOT / 'docs/FAST_SINGLE_DELIVERY.md', '交付说明.md'),
            (ROOT / 'resources/FAST3_ASSET_SOURCES.json', 'FAST3_ASSET_SOURCES.json'),
            (ROOT / 'configs/local-models.fast3.json', 'local-models.fast3.json'),
            (args.tests, 'cpu_full_regression.xml'), (args.contract_tests, 'cpu_single_contract_tests.xml')]:
        shutil.copyfile(source, out / name)
    save(out / 'document_qa.json', qa)
    archive = out / 'MSRC-fast-single5090-code-20261103.zip'
    subprocess.run(['git', 'archive', '--format=zip', '--prefix=MSRC/', '--output=' + str(archive), commit],
        cwd=ROOT, check=True)
    expected = set(git('ls-tree', '-r', '--name-only', commit).splitlines())
    found = set()
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('archive CRC failure')
        for item in bundle.infolist():
            if item.is_dir():
                continue
            if not item.filename.startswith('MSRC/'):
                raise ValueError('archive must use MSRC/ root')
            relative = item.filename[len('MSRC/'):]
            if relative in found or '..' in Path(relative).parts:
                raise ValueError('unsafe archive member')
            found.add(relative)
            if bundle.read(item) != (args.fresh_clone / relative).read_bytes():
                raise ValueError('ZIP differs from fresh GitHub clone: ' + relative)
    if found != expected:
        raise ValueError('archive inventory differs')
    save(out / 'source_archive_verification.json', dict(complete=True, commit=commit,
        remote_readback_commit=commit, zip_sha256=sha(archive), files_checked=len(found),
        all_member_bytes_equal_fresh_remote_clone=True, archive_crc_ok=True, archive_root='MSRC/'))
    for name in CORE:
        if subprocess.check_output(['git', 'show', BASE + ':' + name], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('frozen core changed: ' + name)
    save(out / 'single_code_audit.json', dict(complete=True, base_three_card_commit=BASE, single_commit=commit,
        unchanged_algorithm_files={name: sha(ROOT / name) for name in CORE},
        changed_paths=git('diff', '--name-only', BASE, commit).splitlines(), no_gpu_run_claim=True))
    save(out / 'RELEASE.json', dict(format='msrc-single5090-delivery-v1',
        status='CANDIDATE_RELEASE_NOT_PHYSICAL_GPU_ACCEPTANCE', branch=BRANCH, commit=commit,
        github='https://github.com/hlcccc/MSRC/tree/' + commit, code_zip=archive.name, code_zip_sha256=sha(archive),
        topology='one_host_one_gpu', physical_gpu_count=1, logical_shards=3, workers_serial=True,
        launcher='scripts/run_fast_single.sh', slurm_launcher='scripts/run_fast_single_slurm.sh',
        launch_command='bash scripts/run_fast_single.sh --run-id single_fitted_001',
        scope=dict(factual_models=['internvl'], pope_test=4500, textvqa_test=2496,
            qwen_role='fresh auxiliary donor only', safety_requests=3360, safety_arms=5,
            safety_logical_reviews=16800, original_six_cell_acceptance=False),
        onsite_training=False, test_selection=False, uses_ssh=False, uses_result_transport_archives=False,
        final_metric_pass_not_yet_measured=True, physical_single_gpu_validation_completed=False,
        slurm_cluster_submission_tested=False, no_five_hour_guarantee=True, runtime_estimate_hours=None,
        runtime_note='three unchanged shards run serially; additional reload/development gate overhead; no end-to-end benchmark',
        timing_excludes=['initial downloads', 'environment installation', 'Slurm queue'],
        cpu_evidence=dict(platform=platform.platform(), python=sys.version,
            versions={name: importlib.metadata.version(name) for name in ('pytest', 'torch', 'numpy', 'python-docx')},
            full_regression=full, focused_single_and_fast3_tests=focused,
            synthetic_transport_is_not_gpu_model_quality_evidence=True), document_qa=qa,
        dependency_resolution_reused_from_three_card_release=True,
        large_assets='original separate materials share; unchanged requirements and model/data identities',
        pending=['physical single-5090 full run and final metrics', 'site-specific Slurm submission',
            'fresh Linux offline installation', 'native Word pagination render', 'netdisk upload/readback receipt']))
    save(out / 'READY.json', dict(ready_for_upload=True, commit=commit, delivery_only=True,
        scientific_acceptance_completed=False, destination='/' + NAME))
    checksums = ''.join(sha(p) + '  ' + p.relative_to(out).as_posix() + '\n' for p in sorted(out.rglob('*'))
        if p.is_file() and p != out / 'SHA256SUMS.txt')
    (out / 'SHA256SUMS.txt').write_text(checksums, encoding='utf-8', newline='\n')
    print(json.dumps(dict(ready=True, commit=commit, file_count=len(checksums.splitlines()) + 1,
        checksum_manifest_sha256=sha(out / 'SHA256SUMS.txt'), zip_sha256=sha(archive), output=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
