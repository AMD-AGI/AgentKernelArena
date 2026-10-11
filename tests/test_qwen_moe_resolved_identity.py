"""Equivalent native MoE launch arguments cannot earn gains from route labels."""
import copy
import importlib.util
import math
from pathlib import Path
import shutil
import statistics
from unittest.mock import patch

import pytest

from src.benchmark_quality import attach_case_quality, case_quality, comparison_quality
from src.testcases import TestCaseResult as CaseResult

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/headkernel/qwen3.8-2.4t__fused_moe_2stage_mxfp4'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = load('_moe_identity_contract', TASK / 'scripts/_routing_contract.py')
with patch.dict('sys.modules', {'_routing_contract': CONTRACT}):
    RUNTIME = load('_moe_identity_runtime', TASK / 'scripts/_moe_runtime.py')
    with patch.dict('sys.modules', {'_moe_runtime': RUNTIME}):
        BENCH = load('_moe_identity_bench', TASK / 'scripts/_bench.py')
        with patch.dict('sys.modules', {'_bench': BENCH}):
            FIXTURE = load('_moe_identity_fixture', TASK / 'scripts/test_native_routing.py')


def validated_cases(payload, task):
    request = CONTRACT.validate_request(payload['request'], task, 'performance')
    rows = CONTRACT.validate_payload(payload, FIXTURE.META, request)
    identity = CONTRACT.effective_execution_identity(payload, FIXTURE.META, request)
    names = [row['sig'] for row in rows]
    cases = []
    for row in rows:
        proof = case_quality(row['sig'], case_ids=names, sources=identity,
                             request_id=request['run_id'], manifest_sha256=request['source']['meta_sha256'],
                             series={'candidate': row['samples_ms']}, kind='portable')
        case = CaseResult(row['sig'], execution_time_ms=row['mean_ms'], metadata={
            'benchmark_quality_required': True, 'benchmark_quality_case': proof, 'params': row['params']})
        cases.append(attach_case_quality(case, proof))
    return cases


def refresh_payload(payload, request):
    payload['request'] = request
    payload['request_sha256'] = CONTRACT.fingerprint(request)
    payload['effective_execution_sha256'] = CONTRACT.effective_execution_identity(payload, FIXTURE.META, request)
    return payload


@pytest.mark.parametrize('block', [32, 64, 128])
def test_observed_default_alias_with_favorable_samples_is_an_unchanged_control(tmp_path, block):
    task = tmp_path / 'task'
    shutil.copytree(TASK, task, symlinks=True,
                    ignore=shutil.ignore_patterns('build', '__pycache__', 'reference_io.pt'))
    with patch.object(FIXTURE, 'TASK', task):
        before, request = FIXTURE.ContractTests().payload()
    for row in before['cases']:
        row['resolved_execution'] = FIXTURE.example_execution(block)
        row['native_dispatch'].update(native_block_m=block, effective_block_size_M=block)
    refresh_payload(before, request)
    baseline = validated_cases(before, task)
    source = task / CONTRACT.POLICY_FILE
    source.write_text(source.read_text().replace('"captured"', '"block' + str(block) + '"', 1))
    request = CONTRACT.make_request(task, 'performance', 'b' * 48)
    after = copy.deepcopy(before)
    for row, spec in zip(after['cases'], CONTRACT.specs(FIXTURE.META)):
        route = request['source']['routes'][spec['sig']]
        row['selected_route'] = route
        row['native_dispatch']['declared_block_size_M'] = None if route == 'captured' else block
        samples = [value / 2 for value in row['samples_ms']]
        row.update(samples_ms=samples, mean_ms=math.fsum(samples) / 100,
                   median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples))
    refresh_payload(after, request)
    assert before['request']['source']['declared_policy_sha256'] != request['source']['declared_policy_sha256']
    assert before['effective_execution_sha256'] == after['effective_execution_sha256']
    result = comparison_quality(baseline, validated_cases(after, task))
    assert result['raw_arithmetic_mean_speedup'] == 2.0
    assert result['comparison_status'] == 'unchanged_source_control'
    assert result['accepted_gain'] is False
    assert result['accepted_arithmetic_mean_speedup'] is None


def test_missing_runtime_resolution_cannot_supply_an_effective_identity():
    payload, request = FIXTURE.ContractTests().payload()
    assert request['source']['effective_source_sha256'] is None
    payload['cases'][0].pop('resolved_execution')
    with pytest.raises(ValueError, match='resolved MoE execution'):
        CONTRACT.effective_execution_identity(payload, FIXTURE.META, request)


def test_effective_identity_binds_native_checksum_and_resolved_stage_configuration():
    payload, request = FIXTURE.ContractTests().payload()
    baseline = payload['effective_execution_sha256']
    changed = copy.deepcopy(request)
    changed['source']['native_source_sha256'] = '0' * 64
    assert CONTRACT.effective_execution_identity(payload, FIXTURE.META, changed) != baseline
    payload['cases'][0]['resolved_execution']['metadata'][0]['stage2']['callable'] = 'aiter.fused_moe:other_stage2'
    assert CONTRACT.effective_execution_identity(payload, FIXTURE.META, request) != baseline
