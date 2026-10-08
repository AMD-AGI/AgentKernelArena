#!/usr/bin/env python3
"""Mandatory post-evaluation admission for this fixed two-case MLA task."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.benchmark_quality import assess_comparison
from src.task_contract import fingerprint, require, strict_json, validate_report

HERE = Path(__file__).resolve().parent
READY = strict_json((HERE / 'READY.json').read_text())
TASK = 'tasks/' + READY['ready_tasks'][0]
SOURCE = 'source/pa_sparse_prefill_opus.h'
MANIFEST = strict_json((ROOT / TASK / 'cases.json').read_text())


def validate_and_assess(measurement, reports):
    require(measurement['status'] == 'measured' and measurement['full_case_coverage'], 'Incomplete trusted measurement')
    require(measurement['trusted_commit'] == READY['qualified_commit'] and measurement['task_path'] == TASK,
            'Measurement is outside the qualified commit/task')
    require(measurement['image'] == READY['runtime']['image'], 'Runtime image differs')
    require(measurement['manifest_sha256'] == fingerprint(MANIFEST), 'Case manifest differs')
    require(set(measurement['source_sha256']) == {'reference', 'candidate'}
            and all(set(s) == {SOURCE} for s in measurement['source_sha256'].values()), 'Executable source set differs')
    require(measurement['source_sha256']['reference'] == {SOURCE: READY['source_sha256']}, 'Reference source differs')
    ids = READY['case_ids']; requests = []; performance = {}
    require(set(reports) == {'reference', 'candidate'}, 'Both trusted legs are required')
    for leg in ('reference', 'candidate'):
        require(set(reports[leg]) == {'compile', 'correctness', 'performance'}, 'All six phases are required')
        require(set(measurement['reports'][leg]) == set(reports[leg]), 'Measurement phase inventory differs')
        for phase, report in reports[leg].items():
            request = report['request']; requests.append(request)
            require(request['phase'] == phase and request['source_sha256'] == measurement['source_sha256'][leg]
                    and request['gpu'] == measurement['gpu'], 'Phase/source/GPU binding differs')
            measured = validate_report(report, MANIFEST, request)
            compiled = report['compiled_specializations']
            require(len(compiled) == 2 and {r['case_id'] for r in compiled} == set(ids), 'Compiled case coverage differs')
            for row in compiled:
                candidate, reference = row['candidate_binding'], row['reference_binding']
                require(row['invoked_and_synchronized'] and candidate['fresh_compilation'] and reference['fresh_compilation'],
                        'Fresh native invocation binding is missing')
                require(candidate['source_sha256'] == measurement['source_sha256'][leg]
                        and reference['source_sha256'] == measurement['source_sha256']['reference']
                        and candidate['private_torch_ops'] != reference['private_torch_ops'], 'Native source/operator identity differs')
            if phase == 'performance':
                performance[leg] = {row['case']['case_id']: row['samples_ms'] for row in report['cases']}
                require({row['test_case_id'] for row in measured} == set(ids), 'Performance coverage differs')
    require(len({r['request_id'] for r in requests}) == 6 and len({r['challenge_seed'] for r in requests}) == 1,
            'Independent shared-challenge requests are missing')
    require(len(measurement['cases']) == 2 and {r['test_case_id'] for r in measurement['cases']} == set(ids), 'Summary cases differ')
    comparison = [{'case_id': case, 'work_kind': 'fixed',
                   'reference_samples_ms': performance['reference'][case],
                   'candidate_samples_ms': performance['candidate'][case]} for case in ids]
    quality = assess_comparison(comparison, reference_source=measurement['source_sha256']['reference'],
                                candidate_source=measurement['source_sha256']['candidate'])
    require(math.isclose(measurement['arithmetic_mean_speedup'], quality['raw_arithmetic_mean_speedup'], rel_tol=1e-12),
            'Raw aggregate differs from the retained samples')
    for row in measurement['cases']:
        observed = next(q for q in quality['cases'] if q['case_id'] == row['test_case_id'])
        require(row['case_sha256'] == fingerprint(next(c for c in MANIFEST['cases'] if c['case_id'] == row['test_case_id'])),
                'Summary case fingerprint differs')
        require(all(math.isclose(row[key], observed[leg]['raw_mean_ms'], rel_tol=1e-12)
                    for key, leg in [('reference_ms', 'reference'), ('candidate_ms', 'candidate')])
                and math.isclose(row['speedup'], observed['raw_speedup'], rel_tol=1e-12), 'Summary mean/ratio differs')
    return quality


def read_and_assess(directory):
    directory = Path(directory)
    measurement = strict_json((directory / 'trusted_measurement.json').read_text())
    reports = {}
    for leg in ('reference', 'candidate'):
        reports[leg] = {}
        for phase in ('compile', 'correctness', 'performance'):
            item = measurement['reports'][leg][phase]
            require(item['file'] == leg + '_' + phase + '.json', 'Unexpected phase filename')
            path = directory / item['file']
            require(path.is_file() and not path.is_symlink(), 'Missing regular phase report')
            payload = path.read_bytes()
            require(hashlib.sha256(payload).hexdigest() == item['sha256'], 'Phase report SHA-256 differs')
            reports[leg][phase] = strict_json(payload.decode())
    fixture = measurement['fixtures']; path = directory / 'fixtures_receipt.json'
    require(fixture['file'] == path.name and hashlib.sha256(path.read_bytes()).hexdigest() == fixture['sha256'],
            'Fixture receipt hash differs')
    receipt = strict_json(path.read_text())
    expected = strict_json((ROOT / READY['fixtures']['manifest']).read_text())
    require(receipt['trusted_commit'] == READY['qualified_commit']
            and receipt['case_manifest_fingerprint'] == fingerprint(MANIFEST)
            and receipt['assets'] == expected['assets'], 'Fixture receipt identity/inventory differs')
    return validate_and_assess(measurement, reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement-dir', required=True)
    parser.add_argument('--output', required=True, help='new JSON file; original evidence is never rewritten')
    args = parser.parse_args()
    quality = read_and_assess(args.measurement_dir)
    quality['qualified_task'] = TASK
    quality['qualified_commit'] = READY['qualified_commit']
    quality['quality_module_commit'] = '47aaa88342ba07ef67f95ba7b3c348eb20f945e1'
    quality['quality_module_sha256'] = hashlib.sha256((ROOT / 'src/benchmark_quality.py').read_bytes()).hexdigest()
    quality['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    quality['measurement_sha256'] = hashlib.sha256((Path(args.measurement_dir) / 'trusted_measurement.json').read_bytes()).hexdigest()
    with Path(args.output).open('x') as output:
        json.dump(quality, output, indent=2, sort_keys=True); output.write('\n')
    print(json.dumps({k: quality[k] for k in ('status', 'comparison_status', 'gain_eligible', 'accepted_gain',
                                             'accepted_arithmetic_mean_speedup', 'raw_arithmetic_mean_speedup')}))
    return 0 if quality['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
