"""Explicit task actions. Public evaluation dispatch is independent of agents."""
import os
import test_kernel_harness as h


def inputs():
    return {'performance': list(h.ALL_CONFIGS), 'original_correctness': [h.ALL_CONFIGS[i] for i in h._pick(h.ALL_CONFIGS, 16)], 'controls': h.CONTROL_CASES}


def validate():
    return None


def correctness(require):
    original_indices = h._pick(h.ALL_CONFIGS, 16)
    outcomes = h.mode_correctness(original_indices, collect=True)
    additional = [i for i in range(len(h.ALL_CONFIGS)) if i not in original_indices]
    if additional:
        outcomes.extend(h.mode_correctness(additional, collect=True))

    control = {'test_case_id': h.CONTROL_CASES[0]['test_case_id'], 'status': 'PASS'}
    try:
        require(h.run_contract_controls(), 'none', len(h.CONTROL_CASES))
    except Exception as exc:
        control.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}',
                       failure_kind='execution_failure')
    outcomes.append(control)
    return outcomes


def performance():
    return h.mode_benchmark(list(range(len(h.ALL_CONFIGS))))
