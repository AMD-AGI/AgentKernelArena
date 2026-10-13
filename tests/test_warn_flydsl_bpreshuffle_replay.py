"""CPU negative control for changed quantized operands in bpreshuffle replay."""
from pathlib import Path
import subprocess
import sys


TASK = Path(__file__).resolve().parents[1] / "tasks/torch2flydsl/gemm_a8w8_bpreshuffle_kernel"


def test_replay_changes_both_quantized_operands_and_rejects_cached_output():
    code = """
import importlib.util
import sys
import types
import torch

stub = types.ModuleType('_aka_benchmark')
stub.TimedRun = object
stub.benchmark_cuda_graph_or_events = lambda *a, **k: None
sys.modules['_aka_benchmark'] = stub
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('bpreshuffle_replay_harness', 'test_kernel_harness.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

def quantize(value):
    return value.float().round().clamp(-3, 3).to(torch.int8), torch.ones(
        (value.shape[0], 1), dtype=torch.float32)
model = types.SimpleNamespace(pertoken_quant=quantize)
def pack(weight):
    n, k = weight.shape
    return weight.view(n // 16, 16, k // 32, 2, 16).permute(
        0, 2, 3, 1, 4).contiguous().view(n, k)
h._checked_preshuffle = lambda _kernel, weight: pack(weight)

x, weight = h._make_inputs(8, 16, 32, device='cpu')
xq, x_scale = quantize(x)
wq, w_scale = quantize(weight)
wq_shuf = pack(wq)
inputs = (x, weight, xq, wq, wq_shuf, x_scale, w_scale)
originals = tuple(t.clone() for t in inputs)
expected = h._quantized_dense_reference(xq, wq, x_scale, w_scale)
perturb = lambda: h._install_alternate_quantized_inputs(None, model, inputs, case_index=0)
perturb()
assert not torch.equal(xq, originals[2]) and not torch.equal(wq, originals[3])
assert torch.equal(wq_shuf, pack(wq))
for current, original in zip(inputs, originals):
    current.copy_(original)

class Timed:
    bound = True
    def __init__(self, rerun):
        self.outputs = expected.clone()
        self._rerun = rerun
    def rerun(self):
        return self._rerun()

cached = expected.clone()
try:
    h.verify_timed_run(Timed(lambda: cached.clone()), inputs=inputs,
        originals=originals, expected=expected, perturb=perturb,
        reference=lambda: h._quantized_dense_reference(xq, wq, x_scale, w_scale),
        compare=h._compare_preshuffle_output, minimum_replay_change=2 * h.NORM_TOL)
except AssertionError:
    pass
else:
    raise AssertionError('cached original output passed changed xq/wq replay')
assert all(torch.equal(a, b) for a, b in zip(inputs, originals))

passed = h.verify_timed_run(Timed(lambda: h._quantized_dense_reference(
    xq, wq, x_scale, w_scale)), inputs=inputs, originals=originals,
    expected=expected, perturb=perturb,
    reference=lambda: h._quantized_dense_reference(xq, wq, x_scale, w_scale),
    compare=h._compare_preshuffle_output, minimum_replay_change=2 * h.NORM_TOL)
assert passed['timed_output_checked'] is True
assert all(torch.equal(a, b) for a, b in zip(inputs, originals))

"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=TASK, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_all_measured_quantized_outputs_are_checked_not_only_final_replay():
    code = """
import importlib.util
import sys
import types
import torch

stub = types.ModuleType('_aka_benchmark')
stub.TimedRun = object
stub.benchmark_cuda_graph_or_events = lambda *a, **k: None
sys.modules['_aka_benchmark'] = stub
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('bpreshuffle_samples_harness', 'test_kernel_harness.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
torch.cuda.synchronize = lambda: None

def quantize(value):
    return value.float().round().clamp(-3, 3).to(torch.int8), torch.ones(
        (value.shape[0], 1), dtype=torch.float32)
model = types.SimpleNamespace(pertoken_quant=quantize)
def pack(weight):
    n, k = weight.shape
    return weight.view(n // 16, 16, k // 32, 2, 16).permute(
        0, 2, 3, 1, 4).contiguous().view(n, k)
h._checked_preshuffle = lambda _kernel, weight: pack(weight)

class TimedRun:
    def __init__(self):
        self.bound = False
        self.after_sample = None
        self.outputs = None
    def rerun(self):
        return self._fn()

samples = []
def benchmark(fn, *, repetition, prepare_fn=None, timed_run=None, **kwargs):
    for _ in range(repetition):
        if prepare_fn is not None:
            prepare_fn()
        result = fn()
        if timed_run is not None:
            samples.append((xq.clone(), wq.clone()))
            timed_run.after_sample(result)
    if timed_run is not None:
        timed_run.outputs = result
        timed_run._fn = fn
        timed_run.bound = True
    return 1.0, {'benchmark_method':'cuda_event_fallback',
                 'benchmark_samples':repetition}
h.TimedRun = TimedRun
h.benchmark_cuda_graph_or_events = benchmark

x, weight = h._make_inputs(8, 16, 32, device='cpu')
xq, x_scale = quantize(x)
wq, w_scale = quantize(weight)
wq_shuf = pack(wq)
inputs = (x, weight, xq, wq, wq_shuf, x_scale, w_scale)
originals = tuple(v.clone() for v in inputs)
def correct():
    return h._quantized_dense_reference(xq, wq, x_scale, w_scale)

calls = 0
def bad_middle():
    global calls
    calls += 1
    output = correct()
    return torch.zeros_like(output) if calls == 4 else output
try:
    h._timed_bpreshuffle_case(None, model, inputs, bad_middle,
                              case_index=0, warmup=1, iters=4)
except AssertionError:
    pass
else:
    raise AssertionError('wrong middle measured quantized output passed')
assert all(torch.equal(a, b) for a, b in zip(inputs, originals))
assert len(samples) == 2 and all(not torch.equal(samples[0][i], samples[1][i]) for i in (0, 1))

samples.clear()
_, metadata, _, ref_metadata = h._timed_bpreshuffle_case(
    None, model, inputs, correct, case_index=0, warmup=1, iters=4)
assert metadata['validated_sample_count'] == metadata['benchmark_samples'] == 4
assert metadata['timed_output_checked'] is True
assert metadata['replay_quantized_operands_changed'] is True
assert ref_metadata['benchmark_samples'] == 4
assert all(torch.equal(a, b) for a, b in zip(inputs, originals))

cached = correct().clone()
try:
    h._timed_bpreshuffle_case(None, model, inputs, lambda: cached.clone(),
                              case_index=0, warmup=1, iters=4)
except AssertionError:
    pass
else:
    raise AssertionError('cached original GEMM passed distinct measured quantized operands')
assert all(torch.equal(a, b) for a, b in zip(inputs, originals))

calls = 0
def mutate_middle_quantized_input():
    global calls
    calls += 1
    output = correct()
    if calls == 4:
        xq[0, 0] += 1
    return output
try:
    h._timed_bpreshuffle_case(None, model, inputs, mutate_middle_quantized_input,
                              case_index=0, warmup=1, iters=4)
except AssertionError:
    pass
else:
    raise AssertionError('middle measured call changed a read-only quantized operand')
assert all(torch.equal(a, b) for a, b in zip(inputs, originals))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=TASK, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
