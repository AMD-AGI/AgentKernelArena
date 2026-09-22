"""MXFP4 measured-payload and scale failure controls for the owned port."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'tasks/flydsl2flydsl/silu_and_mul_fq_kernel'
CONTRACT = {'cases': '5c48da0714d4090bba7febcb059171ff341d55506e11bd4530cbbd442345ef75', 'harness_prefix': 'f7b58a6335a78b71ff475ff3bb0f8f00677be71db6d9a92fd3f07b81e9d4f4a0', 'operator_ast': {'kernel.py': 'ca7480637eab945536dd3734ab9a156f2ee0172f0a988624a835e39926af3610'}}


def test_original_operator_cases_and_numerical_timing_harness_preserved():
    assert hashlib.sha256((ROOT / 'cases.json').read_bytes()).hexdigest() == CONTRACT['cases']
    prefix = (ROOT / 'test_kernel_harness.py').read_text().split('def arena_benchmark(', 1)[0]
    prefix = prefix.replace('from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events\n'
                            'from scripts.replay_checks import prepare_check, verify_timed_run',
                            'from _aka_benchmark import benchmark_cuda_graph_or_events')
    assert hashlib.sha256(prefix.encode()).hexdigest() == CONTRACT['harness_prefix']
    for name, digest in CONTRACT['operator_ast'].items():
        tree = ast.parse((ROOT / name).read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == digest


def reference_namespace():
    # Load the actual protected independent CPU codec without the GPU bootstrap.
    names = {'_torch_ref_silu_mul', '_e8m0_biased', '_decode_e2m1', '_scale_tiled_offsets',
             '_nearest_e2m1_code', 'reference_mxfp4', 'decode_kernel_fp4'}
    tree = ast.parse((ROOT / 'test_kernel_harness.py').read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
            or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_E2M1_MAG' for t in node.targets)]
    scope = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), 'protected_reference_codec', 'exec'), scope)
    return scope


@pytest.mark.parametrize('behavior', ['correct', 'wrong_payload', 'wrong_scale', 'stale_replay',
                                    'no_payload_write', 'no_scale_write', 'input_mutation'])
def test_measured_fp4_payload_and_scales_match_the_original_codec(behavior):
    import torch
    from src.tools.perf.aka_benchmark import TimedRun
    spec = importlib.util.spec_from_file_location('quant_replay', ROOT / 'scripts/replay_checks.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    ref = reference_namespace()
    torch.manual_seed(7)
    inputs = dict(x=torch.randn(2, 64).to(torch.bfloat16), rows=2, inter_dim=32, quant_mode='fp4')
    payload = torch.zeros(2, 16, dtype=torch.uint8)
    scales = torch.zeros(320, dtype=torch.uint8)
    def encode():
        real = ref['_torch_ref_silu_mul'](inputs['x'], 32)
        _, e8 = ref['reference_mxfp4'](real, 1)
        grid_scale = torch.pow(2., 127.-torch.tensor(e8, dtype=torch.float32))
        grid = (real.view(2, 1, 32) * grid_scale.unsqueeze(-1)).view(2, 32)
        code, _ = ref['_nearest_e2m1_code'](grid)
        code = code.to(torch.uint8) | ((grid < 0).to(torch.uint8) * 8)
        packed = code[:, ::2] | (code[:, 1::2] << 4)
        out_scale = torch.zeros_like(scales)
        offset = torch.tensor(ref['_scale_tiled_offsets'](2, 1), dtype=torch.long)
        out_scale[offset] = torch.tensor(e8, dtype=torch.uint8)
        return packed, out_scale
    initial = encode()
    payload.copy_(initial[0]); scales.copy_(initial[1])
    check = checks.prepare_check(inputs, payload, scales, ref['_torch_ref_silu_mul'],
                                 ref['reference_mxfp4'], ref['decode_kernel_fp4'])
    def replay():
        result = initial if behavior == 'stale_replay' else encode()
        if behavior != 'no_payload_write':
            payload.copy_(result[0])
        if behavior != 'no_scale_write':
            scales.copy_(result[1])
        return payload, scales
    timed = TimedRun()
    timed._bind(replay, (payload, scales))
    if behavior == 'wrong_payload':
        payload.zero_()
    elif behavior == 'wrong_scale':
        scales.zero_()
    elif behavior == 'input_mutation':
        inputs['x'].add_(1)
    if behavior == 'correct':
        assert checks.verify_timed_run(timed, **check)['replay_correctness'] == 'PASS'
    else:
        with pytest.raises(AssertionError):
            checks.verify_timed_run(timed, **check)
