"""Task-local Arena protocol adapter. Original numerical/timing code is protected."""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'workloads.json'


def load_manifest():
    data = json.loads(MANIFEST.read_text())
    rows = data['cases']
    if not rows or len({r['test_case_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate case manifest')
    return data


def inspect_candidate(data, *, require_implemented=False):
    states = []
    for source, targets in data['candidate_symbols'].items():
        path = ROOT / source
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Candidate path escapes task workspace')
        tree = ast.parse(path.read_text(), filename=source)
        nodes = {n.name:n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for target in targets:
            node = nodes.get(target['name'])
            body = [] if node is None else [n for n in node.body if not (
                isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
            empty = not body or all(isinstance(n, ast.Pass) or (
                isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and n.value.value is Ellipsis)
                for n in body) or (len(body)==1 and isinstance(body[0],ast.Raise))
            states.append(not empty)
            if target['jit'] and node is not None and not any(
                    ast.unparse(d).endswith('.jit') or ast.unparse(d).startswith('triton.jit(')
                    for d in node.decorator_list):
                raise ValueError(f'{source}:{target["name"]} must remain a Triton JIT kernel')
    if not states or (any(states) and not all(states)):
        raise ValueError('Missing or partially implemented declared candidate')
    state = 'implemented' if all(states) else 'unimplemented'
    if require_implemented and state != 'implemented':
        raise ValueError('Final candidate is unimplemented; no baseline fallback is permitted')
    return state


def load_harness():
    path = ROOT / 'scripts/task_runner.py'
    spec = importlib.util.spec_from_file_location('_task_harness', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check_spec = importlib.util.spec_from_file_location('_bincount_checks', ROOT / '_arena_checks.py')
    checks = importlib.util.module_from_spec(check_spec)
    check_spec.loader.exec_module(checks)
    checks.install(module)
    return module


def evaluate(role, action):
    data = load_manifest()
    cases = [copy.deepcopy(row) for row in data['cases'] if action == 'validate-task' or action in row['checks']] if action != 'compile' else []
    result = {'protocol':'arena-eval-v1','role':role,'action':action,'status':'PASS','cases':cases}
    try:
        state = inspect_candidate(data, require_implemented=action != 'validate-task')
        harness = load_harness()
        # The manifest is protected data; verify it against the original input
        # table, independently of candidate outputs or measured performance rows.
        actual = json.loads(json.dumps(getattr(harness, data['case_table'])))
        if actual != data['input_table']:
            raise ValueError('Harness case table disagrees with protected workload manifest')
        if action == 'validate-task':
            for dependency in ('torch','triton'):
                if importlib.util.find_spec(dependency) is None:
                    raise RuntimeError(f'Required runtime dependency unavailable: {dependency}')
            result['metadata']={'candidate_state':state,'input_table_verified':True}
        elif action == 'compile':
            ok, error = harness.run_compile()
            if not ok:
                raise RuntimeError(error or 'Compilation/import check failed')
        elif action == 'correctness':
            for index, row in enumerate(cases):
                try:
                    ok, error = harness.run_correctness(case_index=row['params'].get('case_index', index))
                    if not ok:
                        raise RuntimeError(error or 'Original numerical/output contract rejected candidate')
                    row['metrics']={'original_case_checks_passed':True}
                except BaseException as exc:
                    row.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}', failure_kind='correctness_failure')
        elif action == 'performance':
            measured = harness.run_performance()
            if len(measured) != len(cases):
                raise RuntimeError('Performance omitted declared cases')
            rows_by_id = {r['test_case_id']:r for r in measured}
            if len(rows_by_id) != len(cases) or set(rows_by_id) != {r['test_case_id'] for r in cases}:
                raise RuntimeError('Performance returned missing, duplicate or extra cases')
            for row in cases:
                record = rows_by_id[row['test_case_id']]
                ms = record.get('execution_time_ms')
                method = record.get('benchmark_method')
                if type(ms) not in (int,float) or not math.isfinite(ms) or ms <= 0:
                    row.update(status='FAIL', reason='Invalid or failed device measurement', failure_kind='measurement_failure')
                elif method not in ('cuda_graph','cuda_event_fallback'):
                    row.update(status='FAIL', reason='Missing/unsupported device timing method', failure_kind='measurement_failure')
                else:
                    row.update(execution_time_ms=ms, benchmark_method=method,
                               metadata={'harness_measurement':record})
        else:
            raise ValueError(f'Unsupported action {action}')
        failures = [r for r in cases if r['status'] != 'PASS']
        if failures:
            result.update(status='FAIL', reason=f'{len(failures)} declared cases failed', failure_kind='case_failure')
    except BaseException as exc:
        result.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}', failure_kind='execution_failure')
        for row in cases:
            row.update(status='FAIL', reason=result['reason'], failure_kind='execution_failure')
    return result


def emit_result(result):
    """Even invalid/nonfinite diagnostic metadata must produce a failing envelope."""
    try:
        payload = json.dumps(result, allow_nan=False, separators=(',', ':'))
    except (TypeError, ValueError, OverflowError) as exc:
        result = {'protocol': 'arena-eval-v1', 'role': result['role'], 'action': result['action'],
                  'status': 'FAIL', 'cases': [], 'failure_kind': 'invalid_evidence',
                  'reason': f'Cannot serialize action evidence: {type(exc).__name__}: {exc}'}
        payload = json.dumps(result, allow_nan=False, separators=(',', ':'))
    print('ARENA_EVAL_RESULT=' + payload)
    return 0 if result['status'] == 'PASS' else 1


def main():
    os.chdir(ROOT)
    args = sys.argv[1:]
    role, action = ('task', 'validate-task') if args == ['validate-task'] else (
        tuple(args) if len(args)==2 else ('task','validate-task'))
    try:
        if args != ['validate-task'] and not (len(args)==2 and role in ('baseline','candidate') and
                action in ('compile','correctness','performance')):
            raise ValueError('Use validate-task or baseline|candidate compile|correctness|performance')
        result = evaluate(role, action)
    except BaseException as exc:
        result = {'protocol':'arena-eval-v1','role':role,'action':action,'status':'FAIL','cases':[],
                  'reason':f'{type(exc).__name__}: {exc}','failure_kind':'execution_failure'}
    return emit_result(result)


if __name__ == '__main__':
    raise SystemExit(main())
