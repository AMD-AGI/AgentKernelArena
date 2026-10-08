"""CPU controls for the hgemm Event samples that feed the reported latency."""
from pathlib import Path
import subprocess
import sys


TASK = Path(__file__).resolve().parents[1] / "tasks/torch2flydsl/hgemm_kernel"


def test_actual_middle_measured_output_is_checked_and_inputs_restored():
    code = """
import importlib.util
import sys
import types
import torch

class TimedRun:
    def __init__(self):
        self.bound = False
        self.outputs = None
        self.after_sample = None
    def rerun(self):
        return self._fn()

samples = []
def benchmark(fn, *, repetition, prepare_fn=None, timed_run=None, **kwargs):
    for _ in range(repetition):
        if prepare_fn is not None:
            prepare_fn()
        output = fn()
        if timed_run is not None:
            samples.append((a.clone(), b.clone()))
            if timed_run.after_sample is not None:
                timed_run.after_sample(output)
    if timed_run is not None:
        timed_run.outputs = output
        timed_run._fn = fn
        timed_run.bound = True
    return 1.0, {'benchmark_method': 'cuda_event_fallback',
                 'benchmark_samples': repetition}

stub = types.ModuleType('_aka_benchmark')
stub.TimedRun = TimedRun
stub.benchmark_cuda_graph_or_events = benchmark
sys.modules['_aka_benchmark'] = stub
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('hgemm_sample_harness', 'test_kernel_harness.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
torch.cuda.synchronize = lambda: None
a, b = h._make_inputs(4, 5, 16, device='cpu')
originals = a.clone(), b.clone()
calls = 0
def bad_middle():
    global calls
    calls += 1
    result = h._gemm_reference(a, b)
    return torch.zeros_like(result) if calls == 4 else result

try:
    h._timed_gemm_case(bad_middle, a, b, case_index=0, warmup=1, iters=4)
except AssertionError:
    pass
else:
    raise AssertionError('wrong middle measured output passed because final output was correct')
assert torch.equal(a, originals[0]) and torch.equal(b, originals[1])
assert len(samples) == 4 and torch.equal(samples[0][0], originals[0])
assert len({pair[0].view(torch.uint8).numpy().tobytes() +
            pair[1].view(torch.uint8).numpy().tobytes() for pair in samples}) == 4

samples.clear()
def good():
    return h._gemm_reference(a, b)
_, metadata, _, reference_metadata = h._timed_gemm_case(
    good, a, b, case_index=0, warmup=1, iters=4)
assert metadata['validated_sample_count'] == metadata['benchmark_samples'] == 4
assert metadata['timed_output_checked'] is True
assert reference_metadata['benchmark_samples'] == 4
assert torch.equal(a, originals[0]) and torch.equal(b, originals[1])

cached = h._gemm_reference(a, b).clone()
try:
    h._timed_gemm_case(lambda: cached.clone(), a, b,
                       case_index=0, warmup=1, iters=4)
except AssertionError:
    pass
else:
    raise AssertionError('candidate cached the original GEMM across distinct measured operands')
assert torch.equal(a, originals[0]) and torch.equal(b, originals[1])

calls = 0
def mutate_middle_input():
    global calls
    calls += 1
    result = h._gemm_reference(a, b)
    if calls == 4:
        a.mul_(0.5)
    return result
try:
    h._timed_gemm_case(mutate_middle_input, a, b,
                       case_index=0, warmup=1, iters=4)
except AssertionError:
    pass
else:
    raise AssertionError('middle measured call changed a read-only operand')
assert torch.equal(a, originals[0]) and torch.equal(b, originals[1])
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=TASK, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
