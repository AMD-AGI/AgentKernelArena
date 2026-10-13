"""Event-only Triton rows bind the actual measured output and timing evidence."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.task_validator.report_v2 import _replay_validation_applicability


ROOT = Path(__file__).resolve().parents[1]
EVENT_TASKS = (
    'triton_expand', 'triton_fla_l2norm', 'triton_lightning_attn_diag',
    'triton_sample_recovered_tokens', 'triton_swiglustep_and_mul',
    'triton_w8a8_block_int8_matmul', 'triton_fused_moe_gptq_awq',
)


def _adapter(name):
    path = ROOT / 'tasks/triton2triton/vllm' / name / '_arena_eval.py'
    spec = importlib.util.spec_from_file_location(f'event_adapter_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _applicability(cases):
    spec = SimpleNamespace(candidate=SimpleNamespace(initial_state='unimplemented'),
                           baseline=SimpleNamespace(kind='provided'))
    return _replay_validation_applicability({
        'evidence_valid': True, 'accepted': True, 'spec': spec,
        'results': {('baseline', 'performance'): SimpleNamespace(passed=True, cases=cases)},
    })


@pytest.mark.parametrize('name', EVENT_TASKS)
@pytest.mark.parametrize('checked', [True, False])
def test_event_row_binds_actual_timing_and_checked_output(monkeypatch, name, checked):
    adapter = _adapter(name)
    data = adapter.load_manifest()
    raw = []
    for row in data['cases']:
        if 'performance' in row['checks']:
            raw.append({
                'test_case_id': row['test_case_id'], 'execution_time_ms': 0.25,
                'benchmark_method': 'cuda_event_fallback',
                'benchmark_fallback_reason': 'validate_each_public_invocation',
                'benchmark_warmup': 10, 'benchmark_samples': 100,
                'timed_output_checked': checked,
            })
    harness = SimpleNamespace(**{data['case_table']: data['input_table']})
    harness.CONTROL_CASES = tuple(row['params']['control'] for row in data['cases']
                                  if 'control' in row['params'])
    harness.run_performance = lambda: raw
    monkeypatch.setattr(adapter, 'inspect_candidate', lambda *a, **k: 'implemented')
    monkeypatch.setattr(adapter, 'load_harness', lambda: harness)
    result = adapter.evaluate('baseline', 'performance')
    assert result['status'] == 'PASS'
    rows = result['cases']
    assert len(rows) == len(raw)
    for row in rows:
        assert row['metadata']['device_timing']['benchmark_method'] == row['benchmark_method']
        assert row['metadata']['device_timing']['benchmark_samples'] == 100
        assert row['metadata']['timed_output_checked'] is checked
    assert _applicability(rows)['status'] == ('not_applicable' if checked else 'undetermined')
