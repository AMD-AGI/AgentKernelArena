"""CPU controls for quantized MoE INT8 and SSD causal upper tiles."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MOE = ROOT / 'tasks/triton2triton/vllm/triton_fused_moe_gptq_awq'
SSD = ROOT / 'tasks/triton2triton/vllm/triton_ssd_bmm'


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _functions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


MOE_CHECKS = _module(MOE / '_contract_checks.py', '_moe_gap_checks')
MOE_ACCURACY = _module(MOE / '_numerical_contract.py', '_moe_gap_accuracy')
SSD_CHECKS = _module(SSD / 'scripts/contract_checks.py', '_ssd_gap_checks')


@pytest.mark.parametrize('name', ['int8_dense_explicit', 'int8_dense_default'])
def test_dense_int8_second_geometry_rejects_one_contribution_shortcut(name):
    path = MOE / 'scripts/task_runner.py'
    h = _functions(path, ['control_inputs', 'reference', 'check_output',
                          'check_control_output', 'numerical_contract'],
                   {'torch': torch, 'compare_output': MOE_CHECKS.compare_output,
                    'reference_and_bound': MOE_ACCURACY.reference_and_bound,
                    'assert_accuracy': MOE_ACCURACY.assert_accuracy,
                    'NumericalMismatch': MOE_CHECKS.NumericalMismatch})
    inputs, options = h['control_inputs'](name, 'cpu')
    assert inputs['A'].shape == (4, 64)
    assert inputs['qweight'].shape == (4, 64, 96)
    assert torch.count_nonzero(inputs['A'], dim=1).tolist() == [64] * 4
    expected = h['reference'](inputs, options)
    assert expected.shape == (12, 96) and torch.count_nonzero(expected) > 0
    for column in (0, 65, 95):
        expert = int(inputs['ids'][0, 0])
        scalar = sum(float(inputs['A'][0, k]) *
                     (int(inputs['qweight'][expert, k, column]) -
                      (int(inputs['zeros'][expert, k // 32, column])
                       if 'zeros' in inputs else 128)) *
                     float(inputs['scales'][expert, k // 32, column])
                     for k in range(64))
        if 'weights' in inputs:
            scalar *= float(inputs['weights'][0])
        assert float(expected[0, column]) == scalar
    oracle, check = h['numerical_contract'](options, exact_control=True)
    torch.testing.assert_close(oracle(inputs), expected, atol=0, rtol=0)
    check(expected.clone(), expected)
    sparse = dict(inputs)
    sparse['A'] = inputs['A'].clone()
    sparse['A'][:, 1:] = 0
    with pytest.raises(MOE_CHECKS.NumericalMismatch):
        check(h['reference'](sparse, options), expected)
    dispatch = _functions(path, ['run_correctness'],
                          {**h, 'load_module': lambda: object(),
                           'CONTROL_CASES': (name,),
                           'control_inputs': lambda control, device: (inputs, options),
                           'checked_call': MOE_CHECKS.checked_call,
                           'invoke': lambda mod, live, opts: h['reference'](sparse, opts)})
    ok, error = dispatch['run_correctness'](control=name)
    assert not ok and isinstance(error, MOE_CHECKS.NumericalMismatch)
    manifest = json.loads((MOE / 'workloads.json').read_text())
    assert name in [row['params']['control'] for row in manifest['cases']
                    if 'control' in row['params']]
    assert [row['params']['configuration'] for row in manifest['cases']
            if 'performance' in row['checks']] == manifest['input_table']


def test_causal_128_upper_tile_is_required_by_real_control_dispatch():
    path = SSD / 'scripts/semantic_controls.py'
    h = _functions(path, ['control_cases', 'run_controls'],
                   {'torch': torch, 'InputSnapshot': SSD_CHECKS.InputSnapshot,
                    'check_outputs': SSD_CHECKS.check_outputs,
                    'ContractFailure': SSD_CHECKS.ContractFailure})
    case = next(case for case in h['control_cases']('cpu')
                if case['name'] == 'causal_128_complete_upper_tiles')
    kwargs = dict(case['kwargs'])
    assert kwargs['causal'] and kwargs['chunk_size'] == 128
    assert case['expected'].shape == (1, 2, 128, 128)
    assert torch.count_nonzero(case['expected'][..., :64, 64:]) == 2 * 64 * 64
    runner = _functions(SSD / 'scripts/task_runner.py', ['reference_bmm'], {'torch': torch})
    kwargs.pop('output_dtype')
    torch.testing.assert_close(runner['reference_bmm'](**kwargs), case['expected'], atol=0, rtol=0)

    class Incomplete:
        @staticmethod
        def bmm_chunk_fwd(**call):
            actual = runner['reference_bmm'](**{k: v for k, v in call.items()
                                                if k != 'output_dtype'})
            if call['output_dtype'] is not None:
                actual = actual.to(call['output_dtype'])
            if call['causal'] and call['chunk_size'] == 128:
                actual[..., :64, 64:] = 0
            return actual

    with pytest.raises(SSD_CHECKS.ContractFailure):
        h['run_controls'](Incomplete(), 'cpu')
    source = (SSD / 'source/triton_ssd_bmm.py').read_text()
    assert 'if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:' not in source
    manifest = json.loads((SSD / 'workloads.json').read_text())
    assert [row['params']['configuration'] for row in manifest['cases']
            if 'performance' in row['checks']] == manifest['input_table']
