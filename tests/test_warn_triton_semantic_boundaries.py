"""Extra unscored probes preserve scored workloads while rejecting weak answers."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _runner(name):
    path = ROOT / 'tasks/triton2triton/vllm' / name / 'scripts/task_runner.py'
    spec = importlib.util.spec_from_file_location('semantic_runner_' + name, path)
    module = importlib.util.module_from_spec(spec)
    previous = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    return module


@pytest.mark.parametrize('mode', ['correct', 'zero', 'ignore_gate'])
def test_kda_gla_each_scored_seed_has_significant_gate_state_probe(monkeypatch, mode):
    h = _runner('triton_kda_gla_fwd_o')
    original_gen = h.gen_inputs
    h.gen_inputs = lambda seed, device, **kwargs: original_gen(seed, 'cpu', **kwargs)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    tensor_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == 'cuda':
            return tensor
        return tensor_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)

    def candidate(q, v, g, A, state, scale, chunk_size=64):
        if mode == 'zero':
            return torch.zeros_like(v)
        if mode == 'ignore_gate':
            g = torch.zeros_like(g)
        return h.reference(q, v, g, A, state, scale, chunk_size)

    h.load_module = lambda: SimpleNamespace(kda_gla_fwd_o=candidate)
    for index in range(len(h.SEEDS)):
        ok, reason = h.run_correctness(case_index=index)
        assert ok is (mode == 'correct'), (index, reason)


def test_fla_chunk_scored_length_gate_control_uses_independent_oracle():
    h = _runner('triton_fla_chunk_fwd_o')
    path = ROOT / 'tasks/triton2triton/vllm/triton_fla_chunk_fwd_o/scripts/semantic_controls.py'
    spec = importlib.util.spec_from_file_location('fla_chunk_semantic_controls', path)
    semantic_controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantic_controls)
    cases = [case for case in semantic_controls.control_cases('cpu')
             if case['name'] == 'gated_scored_length']
    assert len(cases) == 1
    case = cases[0]
    assert case['kwargs']['q'].shape[1] == 64
    expected = case['expected']
    torch.testing.assert_close(h.reference(**case['kwargs']), expected,
                               atol=case['atol'], rtol=case['rtol'])
    wrong = dict(case['kwargs'])
    wrong['g'] = None
    assert not torch.allclose(h.reference(**wrong), expected,
                              atol=case['atol'], rtol=case['rtol'])


@pytest.mark.parametrize('wrong_scale', [False, True])
def test_flash_control_checks_strides_and_nondefault_scale(wrong_scale):
    path = ROOT / 'tasks/instruction2triton/rocmbench/test_flashattention_fwd/_arena_reference.py'
    spec = importlib.util.spec_from_file_location('flash_control_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    seen = []
    class FakeKernel:
        def __getitem__(self, grid):
            def launch(q, k, v, scale, L, m, output):
                effective_scale = 0.2 if wrong_scale else scale
                scores = (q @ k.transpose(-1, -2)) * effective_scale
                causal = torch.ones(scores.shape[-2:], dtype=torch.bool).triu_(1)
                scores.masked_fill_(causal, float('-inf'))
                output.copy_(torch.softmax(scores.float(), dim=-1).to(q.dtype) @ v)
                stats = (q.float() @ k.float().transpose(-1, -2)) * effective_scale
                stats.masked_fill_(causal, float('-inf'))
                row_max = stats.max(-1).values
                m.copy_(row_max.flatten(0, 1))
                L.copy_(torch.exp(stats - row_max[..., None]).sum(-1).flatten(0, 1))
            return launch

    def attention(q, k, v, scale):
        seen.append((q.stride(-1), k.stride(-1), v.stride(-1), scale))
        output = torch.empty_like(q)
        L = torch.empty(q.shape[:3]).reshape(-1, q.shape[-2])
        m = torch.empty_like(L)
        module.flash_fwd_kernel[(1,)](q, k, v, scale, L, m, output)
        return output
    module = SimpleNamespace(attention=attention, flash_fwd_kernel=FakeKernel())
    if wrong_scale:
        with pytest.raises(reference.NumericalMismatch):
            reference.check_scale_stride_control(module, 'cpu')
    else:
        reference.check_scale_stride_control(module, 'cpu')
        assert seen == [(2, 2, 2, 0.35)]


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('bad_stat', [False, True])
def test_layernorm_control_checks_mean_rstd_and_row_strides(dtype, bad_stat):
    path = ROOT / 'tasks/instruction2triton/rocmbench/layernorm/_arena_reference.py'
    spec = importlib.util.spec_from_file_location('layernorm_control_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(3)
    context = {'x': torch.randn(8, 257, dtype=dtype),
               'w': torch.rand(257, dtype=dtype),
               'b': torch.rand(257, dtype=dtype)}
    seen = []
    def launch(grid, x, y, w, b, mean, rstd, x_stride, y_stride,
               rows0, cols0, rows1, cols1, eps, block):
        seen.append((x.stride(0), y.stride(0), x_stride, y_stride, eps))
        assert grid == (7,) and (rows0, cols0, rows1, cols1) == (7, 129, 7, 129)
        xf = x.float()
        mu = xf.mean(-1)
        sigma = torch.rsqrt(((xf - mu[:, None]) ** 2).mean(-1) + eps)
        mean.copy_(mu)
        rstd.copy_(sigma)
        y.copy_(torch.nn.functional.layer_norm(x, (129,), w, b, eps))
        if bad_stat:
            mean.zero_()
    module = SimpleNamespace(layernorm_wrapper_fn=launch)
    if bad_stat:
        with pytest.raises(reference.NumericalMismatch):
            reference.check_stats_stride_control(context, module)
    else:
        reference.check_stats_stride_control(context, module)
        assert seen == [(132, 134, 132, 134, 1e-3)]


@pytest.mark.parametrize('bad_output', [False, True])
def test_rmsnorm_control_checks_row_strided_output(bad_output):
    path = ROOT / 'tasks/triton2triton/rocmbench/medium/rmsnorm_fwd/_arena_reference.py'
    spec = importlib.util.spec_from_file_location('rmsnorm_control_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(7)
    context = {'x': torch.randn(8, 257), 'g': torch.rand(257),
               'y_buffer': torch.empty(8, 257), 'ZERO_CENTERED_GAMMA': False}
    seen = []
    def rmsnorm(x,g,y,rsigma,dx,dg,dg_tmp,rows,cols,zero_centered,
                block,blocked,programs,eps):
        seen.append((x.stride(0),y.stride(0),rows,cols,eps))
        xf=x.float()
        r=torch.rsqrt((xf*xf).mean(-1)+eps)
        rsigma.copy_(r)
        y.copy_((xf*r[:,None]*g.float()).to(y.dtype))
        if bad_output:
            y.zero_()
    module = SimpleNamespace(rmsnorm=rmsnorm)
    if bad_output:
        with pytest.raises(reference.NumericalMismatch):
            reference.check_row_stride_control(context,module)
    else:
        reference.check_row_stride_control(context,module)
        assert seen == [(132,134,7,129,1e-3)]


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('broken', [None, 'a', 'b', 'c'])
def test_block_pointer_controls_check_each_operand_stride(broken,dtype):
    path = ROOT / 'tasks/triton2triton/rocmbench/hard/test_block_pointer_matmul/_arena_reference.py'
    spec = importlib.util.spec_from_file_location('block_pointer_control_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    seen = []
    def wrapper(a,b,c,warps):
        which = 'a' if a.stride(1) == 2 else 'b' if b.stride(1) == 2 else 'c'
        seen.append(which)
        c.copy_(a@b)
        if broken == which:
            c.zero_()
        return c
    module = SimpleNamespace(block_pointer_matmul_triton_wrapper=wrapper)
    if broken:
        with pytest.raises(reference.NumericalMismatch):
            reference.check_stride_controls({'a':torch.empty(1,1,dtype=dtype)},module)
    else:
        reference.check_stride_controls({'a':torch.empty(1,1,dtype=dtype)},module)
        assert seen == ['a','b','c']


@pytest.mark.parametrize('name,control,operand', [
    ('triton_moe_mmk','rectangular_tail','A'),
    ('triton_moe_mmk','n_tail','B'),
    ('triton_fused_moe','optional_weights','A'),
    ('triton_fused_moe','unweighted','B'),
])
def test_moe_controls_reject_contiguous_stride_assumptions(name,control,operand):
    h = _runner(name)
    if name == 'triton_moe_mmk':
        inputs = h.control_inputs(control,'cpu')
        reference = lambda values: h.reference(values)
    else:
        inputs,options = h.control_inputs(control,'cpu')
        reference = lambda values: h.reference(values,options)
    tensor = inputs[operand]
    assert tensor.stride(-1) == 2
    expected = reference(inputs)
    wrong = dict(inputs)
    wrong[operand] = torch.as_strided(tensor, tensor.shape,
                                     torch.empty(tensor.shape).stride())
    actual = reference(wrong)
    assert not torch.allclose(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)


def test_ssd_bmm_controls_cover_bfloat16_and_float32_storage():
    path = ROOT / 'tasks/triton2triton/vllm/triton_ssd_bmm/scripts/semantic_controls.py'
    spec = importlib.util.spec_from_file_location('ssd_semantic_controls', path)
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    cases = list(controls.control_cases('cpu'))
    assert {case['kwargs']['a'].dtype for case in cases} == {
        torch.float16, torch.bfloat16, torch.float32}
    for case in cases:
        expected = case['expected']
        output_dtype = case['kwargs']['output_dtype'] or case['kwargs']['a'].dtype
        assert expected.dtype == output_dtype
        bad = expected.to(torch.float16)
        if bad.dtype != expected.dtype:
            with pytest.raises(controls.ContractFailure):
                controls.check_outputs(bad, expected, atol=case['atol'], rtol=case['rtol'])
