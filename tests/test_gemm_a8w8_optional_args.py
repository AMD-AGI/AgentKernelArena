from pathlib import Path
import ast
import importlib.util
import json
import sys
import types
import pytest
import torch

HERE = Path(__file__).resolve().parent
TASK = Path(__file__).resolve().parents[1] / 'tasks/triton2flydsl/aiter/gemm_a8w8'
AFTER = TASK
PACKAGE = 'gemm_api_control_cpu'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(AFTER / 'scripts'), str(TASK / 'scripts')]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + '.api_controls', AFTER / 'scripts/api_controls.py')
controls = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controls
spec.loader.exec_module(controls)
tree = ast.parse((TASK / 'test_kernel_harness.py').read_text())
names = {'_make_inputs', '_torch_ref', '_checked_scaled_gemm_output'}
selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
namespace = {'SEED': 0}
exec(compile(ast.fix_missing_locations(selected), 'protected_reference_functions', 'exec'), namespace)

def reference_call(x, w, xs, ws, bias=None, dtype=torch.bfloat16, y=None, config=None):
    expected = namespace['_torch_ref'](x, w, xs.reshape(x.shape[0], 1), ws.reshape(1, w.shape[0]), bias, dtype)
    if y is not None:
        y.copy_(expected)
        return y
    return expected

def check(fn):
    return controls.run_api_controls(types.SimpleNamespace(gemm_a8w8=fn), namespace['_make_inputs'],
        namespace['_torch_ref'], namespace['_checked_scaled_gemm_output'], torch.float8_e4m3fn,
        device='cpu', verbose=False)

def test_optional_arguments_match_independent_reference_and_preserve_output():
    seen = []
    def implementation(*args, **kwargs):
        seen.append((args[5], args[2].ndim, args[3].ndim, kwargs['y'], kwargs['config']))
        return reference_call(*args, **kwargs)
    assert check(implementation) == []
    assert len(seen) == 4
    assert {row[0] for row in seen} == {torch.float16, torch.bfloat16}
    assert {(row[1], row[2]) for row in seen} == {(1, 1), (2, 2)}
    assert sum(row[3] is not None for row in seen) == 3
    assert sum(row[4] is not None for row in seen) == 2
    assert any(row[3] is not None and not row[3].is_contiguous() for row in seen)

def test_ignoring_preallocated_output_rejected():
    def implementation(*args, **kwargs):
        return reference_call(*args, y=None, config=kwargs['config'])
    assert set(check(implementation)) == {case['name'] for case in controls.API_CASES if case['preallocated_output']}

def test_wrong_numeric_result_rejected_for_every_new_case():
    def implementation(*args, **kwargs):
        output = reference_call(*args, **kwargs)
        output.zero_()
        return output
    assert len(check(implementation)) == 4

def test_input_mutation_rejected():
    def implementation(*args, **kwargs):
        output = reference_call(*args, **kwargs)
        args[2].zero_()
        return output
    assert len(check(implementation)) == 4

def test_configuration_mutation_rejected():
    def implementation(*args, **kwargs):
        output = reference_call(*args, **kwargs)
        if kwargs['config'] is not None:
            kwargs['config']['GROUP_SIZE_M'] = 8
        return output
    assert set(check(implementation)) == {case['name'] for case in controls.API_CASES if case['config_override']}

def test_writes_between_strided_elements_rejected():
    def implementation(*args, **kwargs):
        output = reference_call(*args, **kwargs)
        if not output.is_contiguous():
            torch.as_strided(output, (output.shape[0], output.shape[1] * 2), (output.stride(0), 1))[:, 1::2].zero_()
        return output
    assert check(implementation) == ['api_bf16_strided_output']

def test_optional_cases_are_unscored_and_original_performance_cases_are_retained():
    rows = json.loads((TASK / 'cases.json').read_text())['cases']
    import hashlib
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    assert len(rows) == 22
    assert hashlib.sha256(canonical(rows[:18])).hexdigest() == '118cc00445e062845a378dd8be1573284a3901ca2c5c8efa92b8f2ecdd9533bc'
    assert len([row for row in rows if 'performance' in row['checks']]) == 9
    assert rows[18:] == [{'test_case_id': case['name'], 'params': dict(case, seed=0), 'checks': ['correctness']} for case in controls.API_CASES]


def test_original_reference_input_and_performance_functions_are_preserved():
    tree = ast.parse((TASK / 'test_kernel_harness.py').read_text())
    import hashlib
    retained = {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name != 'run_correctness'}
    assert hashlib.sha256(json.dumps(retained, sort_keys=True).encode()).hexdigest() == 'fd22ede2d4eab7c81218cec1686e17ca5cab5ee594114c723fb92e10e7196e14'
