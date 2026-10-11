#!/usr/bin/env python3
"""Protected controller for the two captured native MoE routing cases."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

from _routing_contract import make_request, strict_json, validate_payload

TASK = Path(__file__).resolve().parents[1]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + secrets.token_hex(8))
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def evaluate(mode, timeout):
    build = TASK / 'build'
    report = build / (mode + '_report.json')
    write_json(report, {'status': 'fail', 'error': 'evaluation incomplete', 'test_cases': []})
    request = make_request(TASK, mode, secrets.token_hex(24))
    directory = build / 'reports' / request['run_id']
    directory.mkdir(parents=True)
    write_json(directory / 'request.json', request)
    if mode == 'compile':
        source = TASK / 'source/routing.py'
        compile(ast.parse(source.read_bytes(), filename=str(source)), str(source), 'exec')
        meta = strict_json((TASK / 'ut/meta.json').read_text())
        native = TASK / 'ut' / meta['source_provenance']['baseline_ref']
        compile(ast.parse(native.read_bytes(), filename=str(native)), str(native), 'exec')
        payload = {'status': 'ok', 'request': request,
                   'parsed_sources': ['source/routing.py'],
                   'symbols_found': {'select_routes': 'source/routing.py'},
                   'compile_scope': 'CPU policy syntax validation and protected native Python syntax; no GPU compilation'}
    else:
        env = dict(os.environ)
        for name in list(env):
            if name == 'PYTHONPATH' or name.startswith('GEAK_'):
                env.pop(name)
        command = [sys.executable, '-I', str(TASK / 'scripts/_bench.py'),
                   '--request', str(directory / 'request.json'),
                   '--out', str(directory / 'raw.json')]
        try:
            proc = subprocess.run(command, cwd=TASK, env=env, text=True,
                                  capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            for name, value in (('stdout', exc.stdout), ('stderr', exc.stderr)):
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='replace')
                (directory / (name + '.txt')).write_text(value or '')
            raise RuntimeError(f'native worker timed out after {timeout}s') from exc
        (directory / 'stdout.txt').write_text(proc.stdout)
        (directory / 'stderr.txt').write_text(proc.stderr)
        if proc.returncode:
            raise RuntimeError(f'native worker exited {proc.returncode}: ' + proc.stderr[-3000:])
        payload = strict_json((directory / 'raw.json').read_text())
        current = make_request(TASK, mode, request['run_id'])
        if current != request:
            raise ValueError('source or metadata changed during evaluation')
        rows = validate_payload(payload, strict_json((TASK / 'ut/meta.json').read_text()), request)
        if mode == 'performance':
            payload['test_cases'] = [dict(row, test_case_id=row['sig'], execution_time_ms=row['mean_ms'])
                                     for row in rows]
    write_json(directory / 'report.json', payload)
    write_json(report, payload)
    if mode == 'performance':
        print(json.dumps({'test_cases': payload['test_cases']}, allow_nan=False))
    print(mode.capitalize() + ': PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('compile', 'correctness', 'performance'))
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    try:
        evaluate(args.mode, args.timeout)
    except Exception as exc:
        write_json(TASK / 'build' / (args.mode + '_report.json'),
                   {'status': 'fail', 'error': str(exc), 'test_cases': []})
        print(f'{args.mode.capitalize()}: FAIL: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
