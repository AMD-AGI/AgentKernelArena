"""Each measured Event output receives the full oracle with bounded retention."""

import ast
import gc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import weakref

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / "tasks/torch2flydsl"
GENERIC = (
    "batched_gemm_a8w8_kernel", "gemm_a16w8_blockscale_kernel",
    "gemm_a16wfp4_kernel", "gemm_a8w8_per_token_scale_kernel",
    "gemm_afp4wfp4_kernel", "gemm_afp8wfp8_kernel",
)


def _controls(task):
    path = ROOT / task / "scripts/sample_controls.py"
    spec = importlib.util.spec_from_file_location("stream_" + task, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compare(actual, expected):
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("task", GENERIC)
def test_all_generic_samples_checked_without_retaining_outputs(task):
    controls = _controls(task)
    generator = torch.Generator().manual_seed(21)
    batched = task.startswith("batched_")
    if batched:
        left = torch.randn((1, 4, 16), generator=generator, dtype=torch.bfloat16)
        right = torch.randn((1, 8, 16), generator=generator, dtype=torch.bfloat16)
        compute = lambda: (left.float() @ right.float().transpose(-1, -2)).to(torch.bfloat16)
        shape = (1, 4, 8)
    else:
        left = torch.randn((4, 16), generator=generator, dtype=torch.bfloat16)
        right = torch.randn((8, 16), generator=generator, dtype=torch.bfloat16)
        compute = lambda: (left.float() @ right.float().T).to(torch.bfloat16)
        shape = (4, 8)
    oracle_calls = [0]
    def reference():
        oracle_calls[0] += 1
        return compute()
    stream = controls.MeasuredInputStream(
        left, right, seed=21, case_index=0, samples=100,
        output_shape=shape, output_dtype=torch.bfloat16,
    )
    stream.bind(reference, _compare)
    seen = []
    for index in range(100):
        stream.prepare()
        assert oracle_calls[0] == index + 1
        output = compute()
        seen.append(weakref.ref(output))
        stream.observe(output)
        assert oracle_calls[0] == index + 1  # Observer issues no oracle GPU work.
    del output
    stream.validate(reference, _compare)
    gc.collect()
    assert stream.checked == stream.prepared == 100
    assert not hasattr(stream, "outputs")
    assert all(ref() is None for ref in seen)


@pytest.mark.parametrize("task", GENERIC)
def test_wrong_middle_sample_rejected_before_next_preparation(task):
    controls = _controls(task)
    left = torch.ones((4, 16), dtype=torch.bfloat16)
    right = torch.ones((8, 16), dtype=torch.bfloat16)
    if task.startswith("batched_"):
        left, right = left[None], right[None]
    reference = lambda: (left.float() @ right.float().transpose(-1, -2)).to(torch.bfloat16)
    stream = controls.MeasuredInputStream(
        left, right, seed=21, case_index=0, samples=100,
        output_shape=reference().shape, output_dtype=torch.bfloat16,
    )
    stream.bind(reference, _compare)
    for index in range(100):
        stream.prepare()
        output = torch.zeros_like(reference()) if index == 50 else reference()
        if index == 50:
            with pytest.raises(AssertionError):
                stream.observe(output)
            break
        stream.observe(output)
    assert stream.checked == 50 and stream.prepared == 51


@pytest.mark.parametrize("task", ("gemm_a4w4_kernel", "hgemm_kernel"))
def test_single_last_oracle_for_special_streams(task):
    controls = _controls(task)
    a = torch.randn((4, 16), dtype=torch.bfloat16)
    b = torch.randn((8, 16), dtype=torch.bfloat16)
    compute = lambda: (a.float() @ b.float().T).to(torch.bfloat16)
    oracle_calls = [0]
    def reference():
        oracle_calls[0] += 1
        return compute()
    stream = controls.MeasuredInputStream(a, b, seed=21, case_index=0, samples=100)
    stream.bind(reference, _compare)
    for index in range(100):
        stream.prepare()
        assert oracle_calls[0] == index + 1
        stream.observe(compute())
        assert oracle_calls[0] == index + 1
    assert torch.equal(stream.validate(reference, _compare), compute())
    assert stream.checked == 100 and not hasattr(stream, "outputs")
    assert stream.last_expected.numel() == 32


def test_bpreshuffle_stream_checks_all_samples_with_bounded_memory():
    controls = _controls("gemm_a8w8_bpreshuffle_kernel")
    x = torch.randn((4, 16), dtype=torch.bfloat16)
    w = torch.randn((8, 16), dtype=torch.bfloat16)
    quantize = lambda value: (value.round().to(torch.int8),
                              torch.ones((value.shape[0], 1)))
    xq, xs = quantize(x)
    wq, ws = quantize(w)
    inputs = (x, w, xq, wq, wq.clone(), xs, ws)
    def make_inputs(m, n, k, *, device, seed):
        gen = torch.Generator().manual_seed(seed)
        return (torch.randn((m, k), generator=gen, dtype=torch.bfloat16),
                torch.randn((n, k), generator=gen, dtype=torch.bfloat16))
    def unchanged(current, saved):
        assert all(torch.equal(a, b) for a, b in zip(current, saved))
    stream = controls.MeasuredQuantizedStream(
        inputs, seed=21, case_index=0, samples=100,
        make_inputs=make_inputs, quantize=quantize,
        checked_preshuffle=lambda value: value.clone(), check_unchanged=unchanged,
    )
    compute = lambda: (xq.float() @ wq.float().T).to(torch.bfloat16)
    oracle_calls = [0]
    def reference():
        oracle_calls[0] += 1
        return compute()
    stream.bind(reference, _compare)
    for index in range(100):
        stream.prepare()
        assert oracle_calls[0] == index + 1
        stream.observe(compute())
        assert oracle_calls[0] == index + 1
    assert torch.equal(stream.validate(reference, _compare), compute())
    assert stream.checked == 100 and not hasattr(stream, "outputs")
    assert stream.last_expected.numel() == 32


def test_hgemm_local_package_wins_with_foreign_kernels_cached(monkeypatch):
    task = ROOT / "hgemm_kernel"
    monkeypatch.syspath_prepend(str(task))
    foreign = ModuleType("kernels")
    foreign.__path__ = ["/foreign/kernels"]
    monkeypatch.setitem(sys.modules, "kernels", foreign)
    sys.modules.pop("hgemm_kernel_ops", None)
    try:
        import hgemm_kernel_ops
        assert Path(hgemm_kernel_ops.__file__).resolve() == task / "hgemm_kernel_ops/__init__.py"
        assert sys.modules["kernels"] is foreign
        source = (task / "kernel.py").read_text()
        assert "from hgemm_kernel_ops import buffer_ops, vector" in source
        assert "from kernels import" not in source
    finally:
        sys.modules.pop("hgemm_kernel_ops", None)


def test_a4w4_diagnostic_receives_the_same_scored_input_sequence():
    controls = _controls("gemm_a4w4_kernel")
    a = torch.randn((4, 16), dtype=torch.bfloat16)
    w = torch.randn((8, 16), dtype=torch.bfloat16)
    stream = controls.MeasuredInputStream(a, w, seed=21, case_index=0, samples=4)
    reference = lambda: (a.float() @ w.float().T).to(torch.bfloat16)
    stream.bind(reference, _compare)
    scored = []
    for _ in range(4):
        stream.prepare()
        scored.append((a.clone(), w.clone()))
        stream.observe(reference())
    stream.validate(reference, _compare)
    diagnostic = []
    for _ in range(4):
        stream.prepare_reference()
        diagnostic.append((a.clone(), w.clone()))
    assert all(torch.equal(left, right)
               for pair in zip(scored, diagnostic)
               for left, right in zip(*pair))
    source = ast.parse((ROOT / "gemm_a4w4_kernel/test_kernel_harness.py").read_text())
    for name in ("run_benchmark", "arena_benchmark"):
        fn = next(node for node in source.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
        calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == "benchmark_cuda_graph_or_events"]
        assert len(calls) == 2
        assert any(keyword.arg == "prepare_fn"
                   and ast.unparse(keyword.value) == "sample_stream.prepare_reference"
                   for keyword in calls[1].keywords)
