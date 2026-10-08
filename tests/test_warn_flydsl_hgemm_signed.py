"""CPU control for the hgemm candidate's signed-input correctness case."""
from pathlib import Path
import subprocess
import sys


TASK = Path(__file__).resolve().parents[1] / "tasks/torch2flydsl/hgemm_kernel"


def test_signed_control_calls_candidate_and_rejects_positive_only_implementation():
    # A subprocess avoids cross-task imports of the many task-local scripts packages.
    code = """
import importlib.util
import sys
import types
import torch

benchmark = types.ModuleType("_aka_benchmark")
benchmark.TimedRun = object
benchmark.benchmark_cuda_graph_or_events = lambda *a, **k: None
sys.modules["_aka_benchmark"] = benchmark
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("hgemm_harness_control", "test_kernel_harness.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

def reference(a, b):
    return (a.float() @ b.float().T).to(a.dtype)

def correct(a, b, **kwargs):
    return reference(a, b)

def positive_only(a, b, **kwargs):
    return reference(a.clamp_min(0), b.clamp_min(0))

shape = h.SHAPES[0]
a, b = h._make_inputs(shape["m"], shape["n"], shape["k"], device="cpu")
h._compare_gemm_output(positive_only(a, b), reference(a, b))
h._signed_candidate_control(correct, reference, shape, device="cpu")
h._mixed_sign_candidate_control(correct, reference, shape, device="cpu")
try:
    h._signed_candidate_control(positive_only, reference, shape, device="cpu")
except AssertionError:
    pass
else:
    raise AssertionError("positive-only candidate passed signed control")

def constant_signed_only(a, b, **kwargs):
    if bool((a == a[:, :1]).all() and (b == b[:, :1]).all()):
        return reference(a, b)
    return reference(a.clamp_min(0), b.clamp_min(0))

h._signed_candidate_control(constant_signed_only, reference, shape, device="cpu")
try:
    h._mixed_sign_candidate_control(constant_signed_only, reference, shape, device="cpu")
except AssertionError:
    pass
else:
    raise AssertionError("constant signed-only candidate passed mixed-sign control")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=TASK, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shape_dependent_signed_shortcut_is_rejected_at_later_shapes():
    code = r"""
import importlib.util, sys, types
import torch
benchmark = types.ModuleType("_aka_benchmark")
benchmark.TimedRun = object
benchmark.benchmark_cuda_graph_or_events = lambda *a, **k: None
sys.modules["_aka_benchmark"] = benchmark
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("hgemm_all_signed", "test_kernel_harness.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
# Small CPU geometries preserve the five separate shape/tiling dispatches.
h.SHAPES = [{**shape, "m": index + 2, "n": 16, "k": 32}
            for index, shape in enumerate(h.SHAPES)]
class Reference:
    def to(self, *_): return self
    def eval(self): return self
    def __call__(self, a, b): return (a.float() @ b.float().T).to(a.dtype)
reference = Reference()
def shortcut(a, b, **_):
    if a.shape[0] != h.SHAPES[0]["m"]:
        a, b = a.clamp_min(0), b.clamp_min(0)
    return reference(a, b)
kernel = types.SimpleNamespace(flydsl_hgemm=shortcut)
model = types.SimpleNamespace(Model=Reference, get_init_inputs=lambda: [])
h._load_module = lambda _directory, filename, _alias: kernel if filename == h.KERNEL_FILE else model
make = h._make_inputs
h._make_inputs = lambda m, n, k: make(m, n, k, device="cpu")
for name in ("_signed_candidate_control", "_mixed_sign_candidate_control"):
    original = getattr(h, name)
    setattr(h, name, lambda candidate, model, shape, original=original:
            original(candidate, model, shape, device="cpu"))
torch.cuda.synchronize = lambda: None
try:
    h.run_correctness(verbose=False)
except AssertionError as error:
    for shape in h.SHAPES[1:]:
        assert "signed_candidate_control/" + shape["name"] in str(error)
        assert "mixed_sign_candidate_control/" + shape["name"] in str(error)
else:
    raise AssertionError("Shape-dependent signed shortcut passed full correctness")
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=TASK,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
