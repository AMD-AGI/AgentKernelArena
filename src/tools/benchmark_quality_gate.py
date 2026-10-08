"""CPU-only quality review of a retained native-quant six-phase bundle.

Writes one new result; original reports and raw timing samples are read-only.
Exit 1 means timing quality failed, 2 means control-only unchanged source.
Neither case contains an accepted speedup. No GPU calls or retries occur.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess

from ..benchmark_quality import gate_native_measurement
from . import trusted_native_eval as native


def review_bundle(*, repo, commit, task_path, evidence, candidate_source=None):
    repo, evidence = Path(repo).resolve(), Path(evidence).resolve()
    if not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', commit):
        raise ValueError('A full immutable commit is required')
    relative = Path(task_path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('A repository-relative task is required')
    task = repo / relative
    image, manifest = native.validate_contract(task)
    # Bind the current readable task to the explicit Git object without an
    # extraction, generated file, import from the candidate, or GPU execution.
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only',
                                      commit, '--', relative.as_posix()], text=True).splitlines()
    if not tracked:
        raise ValueError('Task is absent from the pinned commit')
    for name in tracked:
        expected = subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + name])
        if native.read_regular(repo / name) != expected:
            raise ValueError('Task differs from pinned commit: ' + name)
    reference_bytes = native.read_regular(task / native.SOURCE)
    candidate_bytes = native.read_regular(candidate_source) if candidate_source else reference_bytes
    guard_path = task / 'ut/source_guard.py'
    guard = {'__file__': str(guard_path), '__name__': '_quality_source_guard'}
    exec(compile(native.read_regular(guard_path), str(guard_path), 'exec'), guard)
    guard['validate_source'](candidate_bytes.decode(), reference_bytes.decode())
    identity = {'reference': native.identities(task),
                'candidate': native.identities(task, candidate_bytes=candidate_bytes)}
    path = evidence / 'trusted_measurement.json'
    measurement = json.loads(native.read_regular(path))
    if measurement.get('status') not in ('measured', 'rejected_timing_quality'):
        raise ValueError('An unsuccessful measurement cannot become scoreable')
    if measurement['status'] == 'rejected_timing_quality' and not measurement.get('benchmark_quality'):
        raise ValueError('Rejected measurement lacks its quality decision')
    if measurement.get('benchmark_quality') is not None:
        # A previously gated result remains reviewable from its retained raw
        # score fields; it cannot erase its original quality decision.
        measurement = deepcopy(measurement)
        measurement['status'] = 'measured'
        measurement['arithmetic_mean_speedup'] = measurement['raw_arithmetic_mean_speedup']
        for row in measurement['cases']:
            row['speedup'] = row['raw_speedup']
    if (measurement['trusted_commit'] != commit or measurement['task_path'] != task_path
            or measurement['image'] != image or measurement['reference_source_sha256'] != native.sha256(reference_bytes)
            or measurement['candidate_source_sha256'] != native.sha256(candidate_bytes)):
        raise ValueError('Comparison source, image, task or commit differs')
    if measurement['case_count'] != len(manifest['cases']) or measurement['full_case_coverage'] is not True:
        raise ValueError('Complete case coverage is required')
    if [(r['test_case_id'], r['shape']) for r in measurement['cases']] != [(c['case_id'], c['shape']) for c in manifest['cases']]:
        raise ValueError('Measurement case identity differs')
    reports = {}; run_ids = set()
    for leg in ['reference', 'candidate']:
        if set(measurement['reports'][leg]) != set(native.MODES):
            raise ValueError('All six phases must be retained')
        for mode in native.MODES:
            descriptor = measurement['reports'][leg][mode]
            if any(descriptor.get(k) != value for k, value in identity[leg].items()):
                raise ValueError('Report descriptor source/package identity differs')
            source = evidence / descriptor['file']
            if not source.resolve().is_relative_to(evidence):
                raise ValueError('Report escapes the evidence bundle')
            data = native.read_regular(source)
            if native.sha256(data) != descriptor['sha256']:
                raise ValueError('Retained report hash differs')
            report = json.loads(data)
            native.validate_report(report, mode, identity[leg], manifest['cases'])
            if report['run_id'] in run_ids:
                raise ValueError('Replayed phase run ID')
            run_ids.add(report['run_id'])
            if mode == 'performance':
                reports[leg] = report
    result = gate_native_measurement(measurement, reports['reference'], reports['candidate'])
    result['quality_review_input'] = {'measurement_sha256': native.sha256(native.read_regular(path)),
                                    'evidence_directory': str(evidence), 'commit': commit}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['repo', 'commit', 'task', 'evidence', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--candidate-source')
    args = parser.parse_args()
    result = review_bundle(repo=args.repo, commit=args.commit, task_path=args.task,
                           evidence=args.evidence, candidate_source=args.candidate_source)
    # Never replace an old acceptance/rejection or mutate the measurement.
    with Path(args.output).open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    quality = result['benchmark_quality']
    print(json.dumps({'status': quality['comparison_status'], 'accepted_gain': result['accepted_gain'],
                      'arithmetic_mean_speedup': result['arithmetic_mean_speedup'], 'output': args.output}))
    return 1 if quality['status'] == 'reject' else 0 if quality['gain_eligible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
