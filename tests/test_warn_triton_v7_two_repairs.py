"""CPU controls for the expand bool domain and fused-MoE Event evidence."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.task_validator.report_v2 import _replay_validation_applicability


ROOT = Path(__file__).resolve().parents[1]
MOE = ROOT / 'tasks/triton2triton/vllm/triton_fused_moe'


def _adapter():
    spec = importlib.util.spec_from_file_location('moe_event_adapter', MOE / '_arena_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(row, *, sample_count=100, output_checked=True, readonly_checked=True):
    return {
        'test_case_id': row['test_case_id'], 'execution_time_ms': .125,
        'benchmark_method': 'cuda_event_fallback',
        'benchmark_target_ms': 1.0, 'benchmark_samples': 100,
        'benchmark_max_repeats': 1000, 'benchmark_warmup': 10,
        'benchmark_effective_repeats': 1,
        'benchmark_fallback_reason': 'fused_moe_host_routing_and_dynamic_allocations',
        'benchmark_timed_run_kind': 'eager_callable',
        'benchmark_original_output_checked': output_checked,
        'benchmark_replay_checked': True,
        'benchmark_readonly_inputs_checked': readonly_checked,
        'benchmark_measured_samples_checked': sample_count,
    }


def _evaluated(monkeypatch, *, bad_index=None, **changes):
    adapter = _adapter()
    scored = [row for row in adapter.load_manifest()['cases'] if 'performance' in row['checks']]
    assert len(scored) == 5
    records = [_record(row, **(changes if index == bad_index else {}))
               for index, row in enumerate(scored)]
    harness = SimpleNamespace(CONTROL_CASES=('optional_weights', 'unweighted',
                                              'invalid_experts', 'small_topk_one'),
                              TEST_SHAPES=adapter.load_manifest()['input_table'],
                              BENCHMARK_ITERATIONS=100,
                              run_performance=lambda: records)
    monkeypatch.setattr(adapter, 'load_harness', lambda: harness)
    result = adapter.evaluate('baseline', 'performance')
    return result


def test_fused_moe_event_metadata_is_source_bound_and_applicable(monkeypatch):
    result = _evaluated(monkeypatch)
    assert result['status'] == 'PASS'
    assert len(result['cases']) == 5
    for row in result['cases']:
        meta = row['metadata']
        assert meta['timed_output_checked'] is True
        assert meta['device_timing']['benchmark_method'] == row['benchmark_method'] == 'cuda_event_fallback'
        assert meta['device_timing']['benchmark_fallback_reason'] == 'fused_moe_host_routing_and_dynamic_allocations'
        assert meta['device_timing']['benchmark_measured_samples_checked'] == 100
        assert meta['harness_measurement']['benchmark_original_output_checked'] is True
    spec = SimpleNamespace(candidate=SimpleNamespace(initial_state='implemented'),
                           baseline=SimpleNamespace(kind='initial_candidate'))
    applicability = _replay_validation_applicability({
        'evidence_valid': True, 'accepted': True, 'spec': spec,
        'results': {('baseline', 'performance'): SimpleNamespace(passed=True, cases=result['cases'])},
    })
    assert applicability['status'] == 'not_applicable'
    assert applicability['roles'] == ['baseline'] and applicability['case_count'] == 5


@pytest.mark.parametrize('changes', [
    {'sample_count': 99}, {'output_checked': False}, {'readonly_checked': False},
])
def test_fused_moe_metadata_rejects_unchecked_measurements(monkeypatch, changes):
    result = _evaluated(monkeypatch, bad_index=2, **changes)
    assert result['status'] == 'FAIL'
    assert result['cases'][2]['status'] == 'FAIL'
    assert result['cases'][2]['failure_kind'] == 'measurement_failure'
    assert 'device_timing' not in result['cases'][2].get('metadata', {})
