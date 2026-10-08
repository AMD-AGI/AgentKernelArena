#!/usr/bin/env python3
"""Validate the complete fixed-work FP4 comparison before admitting any gain."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.benchmark_quality import assess_comparison
from src.task_contract import fingerprint, require, strict_json, validate_report
from src.tools.gpu_binding import validate_preflight

READY = strict_json((HERE / 'READY.json').read_text())
TASK = 'tasks/' + READY['ready_tasks'][0]
SOURCE = 'source/kernel.py'
MANIFEST = strict_json((ROOT / TASK / 'cases.json').read_text())


def validate_controls(report):
    rows = report.get('submitted_source_controls', [])
    expected = {(kind, case, mode) for kind in ('no_op', 'wrong_output')
                for case in READY['case_ids'] for mode in ('eager', 'graph')}
    require(len(rows) == 56 and {(r['variant'], r['case_id'], r['mode']) for r in rows} == expected,
            'All 56 source controls are required, without duplicates')
    for row in rows:
        require(row['status'] == 'candidate_rejected' and row['reference_calibrated'] is True
                and row.get('candidate_compiled_and_engaged') is True,
                'A source control was accepted or failed before candidate engagement')
        require(row['error_type'] == 'AssertionError' and row['failure_phase'] == row['mode']
                and any(message in row['error'] for message in
                        ('independent packed-value oracle mismatch', 'unwritten/nonfinite output or reference'))
                and row['scoreable'] is False,
                'A source control did not fail at its numerical oracle')
        case = next(c for c in MANIFEST['cases'] if c['case_id'] == row['case_id'])
        require(fingerprint(row['case']) == fingerprint(case), 'Source-control case identity differs')
        require(row['source_proof']['source_sha256'] == READY['source_negative_sha256'][row['variant']]
                and row['source_proof']['writable_storage_tracking'] is True,
                'Source-control bytes or writable-storage ownership differ')
        compiled = row['candidate_compilation']
        require(compiled['native_invocations'] >= 1 and compiled['compiled_kernels']
                and all(item['kernel_name'] and re.fullmatch('[0-9a-f]{64}', item['kernel_hash'])
                        for item in compiled['compiled_kernels']), 'Missing actual candidate kernel evidence')
        require(row['seed'] == report['request']['challenge_seed'] and row['performance_samples'] == 0,
                'Source-control seed or diagnostic scope differs')
        require(row['mode'] != 'graph' or row['graph_captured'] and row['graph_replayed'],
                'Graph control was not captured and replayed')


def validate_and_assess(measurement, reports):
    require(measurement['status'] == 'measured' and measurement['full_case_coverage'], 'Incomplete measurement')
    require(measurement['trusted_commit'] == READY['qualified_commit'] and measurement['task_path'] == TASK,
            'Measurement commit/task differs from this starter')
    require(measurement['image'] == READY['runtime']['image']
            and measurement['manifest_sha256'] == fingerprint(MANIFEST), 'Runtime/case manifest differs')
    require(set(measurement['source_sha256']) == {'reference', 'candidate'}
            and all(set(value) == {SOURCE} for value in measurement['source_sha256'].values()), 'Executable source set differs')
    require(measurement['source_sha256']['reference'] == {SOURCE: READY['source_sha256']}, 'Reference source differs')
    requests = []; performance = {}; ids = READY['case_ids']
    require(set(reports) == {'reference', 'candidate'}, 'Both trusted source legs are required')
    for leg in ('reference', 'candidate'):
        require(set(reports[leg]) == set(measurement['reports'][leg]) == {'compile', 'correctness', 'performance'},
                'All six trusted phases are required')
        for phase, report in reports[leg].items():
            request = report['request']; requests.append(request)
            require(request['phase'] == phase and request['source_sha256'] == measurement['source_sha256'][leg]
                    and request['gpu'] == measurement['gpu'], 'Phase/source/GPU identity differs')
            measured = validate_report(report, MANIFEST, request)
            compiled = report['compiled_kernels']
            require(len(compiled) == 14 and {r['case_id'] for r in compiled} == set(ids), 'Compiled case coverage differs')
            require(len({r['private_module'] for r in compiled}) == 14, 'Private source bindings are reused')
            for row in compiled:
                require(row['source_sha256'] == measurement['source_sha256'][leg][SOURCE]
                        and row['wrapper_registration_removed'] is True and row['writable_storage_tracking'] is True,
                        'Private source binding or written-storage ownership differs')
                require(row['compiled_kernels'] and all(k['kernel_name'] and re.fullmatch('[0-9a-f]{64}', k['kernel_hash'])
                        for k in row['compiled_kernels']), 'Native specialization evidence missing')
            if phase == 'correctness':
                require(all(row['seeds'] == [0, 1, 2] and row['eager_and_graph_checked'] for row in report['cases']),
                        'Original eager/graph correctness seeds are incomplete')
                validate_controls(report)
            elif phase == 'performance':
                require({row['test_case_id'] for row in measured} == set(ids), 'Performance case coverage differs')
                performance[leg] = {row['case']['case_id']: row['samples_ms'] for row in report['cases']}
    require(len({r['request_id'] for r in requests}) == 6 and len({r['challenge_seed'] for r in requests}) == 1,
            'Fresh independent phase requests with one shared challenge are required')
    require(len(measurement['cases']) == 14 and {r['test_case_id'] for r in measurement['cases']} == set(ids),
            'Summary case coverage differs')
    quality = assess_comparison([{'case_id':case, 'work_kind':'fixed',
        'reference_samples_ms':performance['reference'][case], 'candidate_samples_ms':performance['candidate'][case]}
        for case in ids], reference_source=measurement['source_sha256']['reference'],
        candidate_source=measurement['source_sha256']['candidate'])
    require(math.isclose(measurement['arithmetic_mean_speedup'], quality['raw_arithmetic_mean_speedup'], rel_tol=1e-12),
            'Raw aggregate differs from retained samples')
    for row in measurement['cases']:
        observed = next(r for r in quality['cases'] if r['case_id'] == row['test_case_id'])
        case = next(c for c in MANIFEST['cases'] if c['case_id'] == row['test_case_id'])
        require(row['case_sha256'] == fingerprint(case), 'Summary case fingerprint differs')
        require(math.isclose(row['reference_ms'], observed['reference']['raw_mean_ms'], rel_tol=1e-12)
                and math.isclose(row['candidate_ms'], observed['candidate']['raw_mean_ms'], rel_tol=1e-12)
                and math.isclose(row['speedup'], observed['raw_speedup'], rel_tol=1e-12), 'Raw case mean/ratio differs')
    return quality


def read_and_assess(directory):
    directory = Path(directory)
    measurement = strict_json((directory / 'trusted_measurement.json').read_text()); reports = {}
    for leg in ('reference', 'candidate'):
        reports[leg] = {}
        for phase in ('compile', 'correctness', 'performance'):
            entry = measurement['reports'][leg][phase]
            require(entry['file'] == leg + '_' + phase + '.json', 'Unexpected phase filename')
            path = directory / entry['file']; require(path.is_file() and not path.is_symlink(), 'Missing regular report')
            raw = path.read_bytes(); require(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'Phase report hash differs')
            report = strict_json(raw.decode()); reports[leg][phase] = report
            diagnostics = directory / (leg + '_' + phase + '.diagnostics')
            hashes = strict_json((diagnostics / 'hashes.json').read_text())
            preflight = diagnostics / 'gpu_preflight.json'
            require(hashlib.sha256(preflight.read_bytes()).hexdigest() == hashes[preflight.name], 'GPU preflight hash differs')
            validate_preflight(strict_json(preflight.read_text()), report['request']['gpu'])
    fixture = directory / 'fixtures_receipt.json'
    require(measurement['fixtures']['file'] == fixture.name and hashlib.sha256(fixture.read_bytes()).hexdigest()
            == measurement['fixtures']['sha256'], 'Fixture receipt hash differs')
    receipt = strict_json(fixture.read_text()); expected = strict_json((ROOT / READY['fixtures']['manifest']).read_text())
    require(receipt['trusted_commit'] == READY['qualified_commit'] and receipt['case_manifest_fingerprint'] == fingerprint(MANIFEST)
            and receipt['assets'] == expected['assets'], 'Fixture receipt scope differs')
    return validate_and_assess(measurement, reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement-dir', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); quality = read_and_assess(args.measurement_dir)
    quality.update(qualified_task=TASK, qualified_commit=READY['qualified_commit'],
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        quality_module_sha256=hashlib.sha256((ROOT / 'src/benchmark_quality.py').read_bytes()).hexdigest(),
        measurement_sha256=hashlib.sha256((Path(args.measurement_dir) / 'trusted_measurement.json').read_bytes()).hexdigest())
    with Path(args.output).open('x') as output:json.dump(quality, output, indent=2); output.write('\n')
    print(json.dumps({k:quality[k] for k in ('status','comparison_status','gain_eligible','accepted_gain',
        'accepted_arithmetic_mean_speedup','raw_arithmetic_mean_speedup')}))
    return 0 if quality['status'] == 'pass' else 2

if __name__ == '__main__':raise SystemExit(main())
