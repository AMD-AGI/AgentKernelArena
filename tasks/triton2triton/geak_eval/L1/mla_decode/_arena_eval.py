"""Self-contained v2 action runner over this task's protected GEAK harness."""
from __future__ import annotations
import ast
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def inspect_candidate(data, *, require_implemented=False):
    path = ROOT / data['source']
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('Candidate escaped the task workspace')
    tree = ast.parse(path.read_text(), filename=data['source'])
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    implemented = []
    for target in data['candidate_symbols']:
        node = functions.get(target['name'])
        body = [] if node is None else [s for s in node.body if not (
            isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
        empty = not body or all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis) for s in body) or (len(body) == 1 and isinstance(body[0], ast.Raise))
        implemented.append(not empty)
        if node is not None and target['jit'] and not any(ast.unparse(d).endswith('.jit') for d in node.decorator_list):
            raise ValueError(f"{target['name']} must remain a Triton JIT kernel")
    if not implemented or (any(implemented) and not all(implemented)):
        raise ValueError('Missing or partially implemented candidate')
    state = 'implemented' if all(implemented) else 'unimplemented'
    if require_implemented and state != 'implemented':
        raise ValueError('Final candidate is unimplemented; no baseline fallback')
    return state


def load_actions():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('_task_actions', ROOT / '_arena_actions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_success(value, kind, count):
    """Interpret the actual harness return type; None is not a universal pass."""
    if kind == 'none':
        ok = value is None
    elif kind == 'zero':
        ok = type(value) is int and value == 0
    elif kind == 'bool':
        ok = value is True
    elif kind == 'dict':
        ok = (isinstance(value, dict) and value.get('correct') is True
              and value.get('num_correct') == count and value.get('num_failed') == 0
              and not value.get('skipped') and not value.get('failures'))
    else:
        raise ValueError('Unknown task-local correctness return contract')
    if not ok:
        raise RuntimeError(f'Correctness did not complete all {count} cases: {value!r}')


def capture_performance(actions, data):
    """Read current device measurements; never consume an old report or log."""
    measurements = []
    original = actions.h.benchmark_cuda_graph_or_events
    def measure(*args, **kwargs):
        ms, metadata = original(*args, **kwargs)
        if type(ms) not in (int, float) or not math.isfinite(ms) or ms <= 0:
            raise ValueError('Device latency must be positive and finite')
        if metadata.get('benchmark_method') not in ('cuda_graph', 'cuda_event_fallback'):
            raise ValueError('Missing graph/event device timing provenance')
        measurements.append((ms, metadata))
        return ms, metadata
    actions.h.benchmark_cuda_graph_or_events = measure
    try:
        value = actions.performance()
        if isinstance(value, dict) and value.get('skipped'):
            raise RuntimeError('The performance harness skipped this task')
    finally:
        actions.h.benchmark_cuda_graph_or_events = original
    rows = [r for r in data['cases'] if 'performance' in r['checks']]
    pattern = data['timing_calls_per_case']
    if len(measurements) != len(rows) * len(pattern):
        raise RuntimeError(f'Missing/extra measurements: got {len(measurements)}, expected {len(rows) * len(pattern)}')
    offset = pattern.index('candidate')
    selected = measurements[offset::len(pattern)]
    return selected


def evaluate(role, action):
    result = {'protocol': 'arena-eval-v1', 'role': role, 'action': action, 'status': 'FAIL', 'cases': []}
    log = io.StringIO()
    outcomes = None
    try:
        data = json.loads((ROOT / 'workloads.json').read_text())
        if action != 'compile':
            result['cases'] = [deepcopy(r) for r in data['cases'] if action == 'validate-task' or action in r['checks']]
            for row in result['cases']:
                row.update(status='FAIL', reason='Case has not completed')
        state = inspect_candidate(data, require_implemented=action != 'validate-task')
        if action == 'compile':
            # Original compile contract was AST/syntax checking. Correctness
            # must also run every required specialization before acceptance.
            compile((ROOT / data['source']).read_text(), data['source'], 'exec')
        else:
            with redirect_stdout(log), redirect_stderr(log):
                actions = load_actions()
                if serial(actions.inputs()) != data['input_tables']:
                    raise ValueError('Harness inputs differ from the independent protected manifest')
                actions.validate()
                if action == 'correctness':
                    outcomes = actions.correctness(require_success)
                elif action == 'performance':
                    measurements = capture_performance(actions, data)
                    for row, (ms, metadata) in zip(result['cases'], measurements):
                        row.update(execution_time_ms=ms, benchmark_method=metadata['benchmark_method'],
                                   metadata={'device_timing': metadata})
                elif action != 'validate-task':
                    raise ValueError('Unknown task action')
            if outcomes is not None:
                by_id = {row['test_case_id']: row for row in outcomes}
                if len(by_id) != len(outcomes) or set(by_id) != {r['test_case_id'] for r in result['cases']}:
                    raise ValueError('Incomplete or duplicate correctness outcomes')
                for row in result['cases']:
                    outcome = by_id[row['test_case_id']]
                    if outcome['status'] not in ('PASS', 'FAIL'):
                        raise ValueError('Unknown correctness outcome')
                    row.pop('reason', None)
                    row.update(outcome)
                failed = [row for row in result['cases'] if row['status'] == 'FAIL']
                if failed:
                    numerical = all(row.get('metadata', {}).get('failure_kind') == 'numerical_mismatch'
                                    for row in failed)
                    result.update(reason=f'{len(failed)} completed correctness cases failed',
                                  failure_kind='numerical_mismatch' if numerical else 'execution_failure',
                                  metadata={'harness_output_tail': log.getvalue()[-4000:]})
                    return result
            for row in result['cases']:
                row['status'] = 'PASS'
                row.pop('reason', None)
            if action == 'validate-task':
                result['metadata'] = {'candidate_state': state, 'manifest_verified': True}
        result['status'] = 'PASS'
    except BaseException as exc:
        # SystemExit, missing dependencies, incorrect shapes and skipped cases
        # are never mislabeled as a numerical-only baseline diagnostic.
        result.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}', failure_kind='execution_failure')
        # A setup/import/collection failure proves no per-case completion.
        # Report an action failure without inventing outcomes for unrun cases.
        result['cases'] = []
        if log.getvalue():
            result['metadata'] = {'harness_output_tail': log.getvalue()[-4000:]}
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
    role, action = ('task', 'validate-task') if args == ['validate-task'] else (tuple(args) if len(args) == 2 else ('task', 'validate-task'))
    if args != ['validate-task'] and not (len(args) == 2 and role in ('baseline', 'candidate') and action in ('compile', 'correctness', 'performance')):
        result = {'protocol': 'arena-eval-v1', 'role': role, 'action': action, 'status': 'FAIL', 'cases': [],
                  'reason': 'Use validate-task or baseline|candidate compile|correctness|performance', 'failure_kind': 'invalid_arguments'}
    else:
        result = evaluate(role, action)
    return emit_result(result)


if __name__ == '__main__':
    raise SystemExit(main())
