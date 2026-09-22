"""FNUZ cache contract preservation and explicit gfx950 numerical conversion."""
import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'tasks/flydsl2flydsl/pa_decode_swa_kernel'
CONTRACT = {'cases': 'c732f6c58b5a150ee25d98ca366d5be0adac6c1374f53918b2549ca88fb9cab5', 'harness_prefix': '9b34b5b9bb1fd26e791333e183de4a7864a1fd6d072b78da656e2487609a87a8', 'operator_ast': {'kernel.py': 'f8c9414e24eb26dcbc6684c663ce6affc82bd8dc959391927fd4e987b7e2ace7', 'kernels/__init__.py': 'ffba730c73ed3fb7eda50ac01930a9251bca46e6e069d7fb267bc4660f06382b', 'kernels/dpp_utils.py': '7194b5e55c10dc1aaae7fc40e67b79478ecc62c76da2dc0fe0658d7552427fb6'}}


def test_original_cases_inputs_reference_and_timing_harness_preserved():
    assert hashlib.sha256((ROOT / 'cases.json').read_bytes()).hexdigest() == CONTRACT['cases']
    prefix = (ROOT / 'test_kernel_harness.py').read_text().split('def arena_benchmark(', 1)[0]
    prefix = prefix.replace('from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events\n'
                            'from scripts.replay_checks import verify_timed_run, compare_output',
                            'from _aka_benchmark import benchmark_cuda_graph_or_events')
    assert hashlib.sha256(prefix.encode()).hexdigest() == CONTRACT['harness_prefix']


class Scalar(int):
    def ir_value(self): return int(self)
    def __or__(self, other): return Scalar(int(self) | int(other))
    def __and__(self, other): return Scalar(int(self) & int(other))
    def __rshift__(self, other): return Scalar(int(self) >> int(other))
    def __lshift__(self, other): return Scalar(int(self) << int(other))
    def __sub__(self, other): return Scalar(int(self) - int(other))
    def __add__(self, other): return Scalar(int(self) + int(other))


@pytest.mark.parametrize('arch', ['gfx942', 'gfx950'])
def test_actual_conversion_function_all_256_encodings_and_legacy_identity(arch):
    import torch
    tree = ast.parse((ROOT / 'kernel.py').read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_kv_word_for_arch')
    scope = {'_KV_USES_OCP_INSTRUCTIONS': arch == 'gfx950', 'fx': SimpleNamespace(Int64=Scalar),
             'arith': SimpleNamespace(select=lambda condition, yes, no: yes if condition else no),
             'range_constexpr': range}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'actual_register_conversion', 'exec'), scope)
    raw = torch.arange(256, dtype=torch.uint8)
    expected = raw if arch == 'gfx942' else raw.view(torch.float8_e4m3fnuz).float().to(torch.float8_e4m3fn).view(torch.uint8)
    for start in range(0, 256, 8):
        word = int.from_bytes(bytes(range(start, start + 8)), 'little')
        actual = scope['_kv_word_for_arch'](word) & ((1 << 64) - 1)
        assert list(actual.to_bytes(8, 'little')) == expected[start:start+8].tolist()


@pytest.mark.parametrize('behavior', ['correct', 'wrong_measured', 'stale_replay', 'no_replay_write'])
def test_measured_decode_output_and_same_replay_checked(behavior):
    import torch
    from src.tools.perf.aka_benchmark import TimedRun
    spec = importlib.util.spec_from_file_location('pa_replay', ROOT / 'scripts/replay_checks.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    q = torch.tensor([[1., -2.]])
    original = q.clone()
    expected = q * 2
    out = expected.clone()
    def replay():
        if behavior == 'stale_replay': out.copy_(expected)
        elif behavior != 'no_replay_write': out.copy_(q * 2)
        return out
    timed = TimedRun();timed._bind(replay, out)
    if behavior == 'wrong_measured':out.fill_(float('nan'))
    args = dict(inputs=(q,), originals=(original,), expected=expected,
                reference=lambda:q * 2, perturb=lambda:q.neg_(),
                compare=lambda actual,ref:checks.compare_output(actual,ref,.03))
    if behavior == 'correct':assert checks.verify_timed_run(timed, **args)['replay_correctness'] == 'PASS'
    else:
        with pytest.raises(AssertionError):checks.verify_timed_run(timed, **args)
