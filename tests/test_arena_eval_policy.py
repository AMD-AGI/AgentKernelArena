"""CPU checks of the v3 evaluator's comparison rules and candidate policy."""
from __future__ import annotations

import pytest
import torch

from src.tools.arena_eval.audit import PolicyError, audit_hip_sources, audit_python_sources, preparation_only
from src.tools.arena_eval.compare import compare_tensor

TRITON_OK = '''
import torch
import triton
import triton.language as tl
from .helpers import BLOCK


@triton.jit
def _k(x, y, n, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.store(y + i, tl.load(x + i, mask=i < n), mask=i < n)


def run(*, x, out):
    _k[(1,)](x, out, x.numel(), B=BLOCK)
'''


def _sources(text, extra=None):
    sources = {"kernel.py": text, "helpers.py": "BLOCK = 128\n"}
    sources.update(extra or {})
    return sources


def test_a_plain_triton_candidate_passes():
    audit_python_sources(_sources(TRITON_OK), "triton")


@pytest.mark.parametrize("snippet, message", [
    ("import os", "not allowed"),
    ("import aiter", "not allowed"),
    ("from triton.experimental import gluon", "not allowed"),
    ("import torch.nn.functional as F", "not allowed"),
    ("y = torch.matmul", "not a candidate implementation dependency"),
    ("z = eval('1')", "eval"),
    ("print('x')", "print"),
    ("torch.cuda.synchronize()", "synchronize"),
    ("v = (lambda f: f.__globals__)(run)", "__globals__"),
    ("torch.manual_seed(0)", "candidate implementation dependency"),
    ("triton.language.sum = None", "mutate an imported module"),
    ("import _aka_benchmark", "not allowed"),
    ("from .. import reference", "relative imports"),
])
def test_rejected_candidate_code(snippet, message):
    with pytest.raises(PolicyError, match=message):
        audit_python_sources(_sources(TRITON_OK + "\n" + snippet + "\n"), "triton")


@pytest.mark.parametrize("snippet, message", [
    # Process, FFI and interpreter modules that torch and triton re-export as attributes.
    ("L = torch.ctypes.CDLL('libhipblaslt.so')", "ctypes reaches a process, FFI or interpreter module"),
    ("m = torch.sys.modules", "sys reaches"),
    ("torch.builtins.setattr(torch, 'x', 1)", "builtins reaches"),
    ("e = torch.os.environ", "os reaches"),
    ("triton.runtime.build.subprocess.run(['true'])", "subprocess reaches"),
    # getattr follows the attribute rules, and a computed name works only on self.
    ("L = getattr(torch, 'ctypes')", "ctypes reaches"),
    ("L = getattr(torch, '_C')", "torch._C is not a candidate implementation dependency"),
    ("L = getattr(torch, 'ct' + 'ypes')", "computed name is allowed only on self"),
    ("import functools\nL = functools.reduce(getattr, ['ctypes'], torch)", "only be called directly"),
    ("import operator\nL = operator.attrgetter('ctypes')(torch)", "attrgetter"),
    ("from operator import methodcaller", "methodcaller"),
    ("L = x._aka_benchmark", "evaluator names"),
])
def test_reflective_and_reexported_module_routes_are_rejected(snippet, message):
    with pytest.raises(PolicyError, match=message):
        audit_python_sources(_sources(TRITON_OK + "\n" + snippet + "\n"), "triton")


def test_literal_getattr_and_self_attribute_proxies_stay_allowed():
    allowed = ("\nDTYPE = getattr(tl, 'float16')\nHALF = torch.float16\nEXP = tl.math.exp\n\n"
               "class Proxy:\n    def __getattr__(self, name):\n        return getattr(self._parent, name)\n")
    audit_python_sources(_sources(TRITON_OK + allowed), "triton")


def test_every_python_baseline_passes_the_candidate_audit():
    # A baseline is also the starting candidate when the run targets its language.
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1] / "tasks"
    audited = 0
    for task in sorted(root.glob("*/*/*/*/task.yaml")):
        language = (yaml.safe_load(task.read_text(encoding="utf-8")).get("baseline") or {}).get("language")
        if language not in ("triton", "gluon", "flydsl"):
            continue
        baseline = task.parent / "baseline"
        sources = {p.relative_to(baseline).as_posix(): p.read_text(encoding="utf-8")
                   for p in baseline.rglob("*.py") if "__pycache__" not in p.parts}
        audit_python_sources(sources, language)
        audited += 1
    assert audited


def test_a_candidate_must_import_its_language():
    with pytest.raises(PolicyError, match="imports no"):
        audit_python_sources({"kernel.py": "import torch\n\ndef run(*, x, out):\n    pass\n"}, "triton")


def test_gluon_and_flydsl_imports_are_language_specific():
    gluon = TRITON_OK.replace("import triton.language as tl", "import triton.language as tl\n"
                              "from triton.experimental import gluon")
    audit_python_sources(_sources(gluon), "gluon")
    with pytest.raises(PolicyError):
        audit_python_sources(_sources(TRITON_OK.replace("import triton\n", "import flydsl\n")), "triton")


HIP_OK = '''// copy
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include "util.h"
__global__ void k(const float* x, float* y, int n) { int i = threadIdx.x; if (i < n) y[i] = x[i]; }
void run(const at::Tensor& x, at::Tensor& out) {
  hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, at::hip::getCurrentHIPStream(),
                     x.data_ptr<float>(), out.data_ptr<float>(), (int)x.numel());
}
#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) { m.def("run(Tensor x, Tensor(a!) out) -> ()", &run); }
'''


def test_hip_candidate_rules():
    sources = {"kernel.hip": HIP_OK, "util.h": "#pragma once\n"}
    audit_hip_sources(sources, "kernel.hip")
    for bad, message in [("#include <rocblas/rocblas.h>", "allowed header"),
                         ("void f() { hipDeviceSynchronize(); }", "not allowed"),
                         ("void g(at::Tensor a) { at::matmul(a, a); }", "not allowed"),
                         ("__device__ int counter;", "persists between calls")]:
        with pytest.raises(PolicyError, match=message):
            audit_hip_sources({"kernel.hip": HIP_OK + bad + "\n", "util.h": ""}, "kernel.hip")
    with pytest.raises(PolicyError, match="single"):
        audit_hip_sources({"kernel.hip": HIP_OK.replace("TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME", "X("),
                           "util.h": ""}, "kernel.hip")


@pytest.mark.parametrize("bad, message", [
    # The reported exploit: a CUDA-backend operator header and its direct entry point.
    ("#include <ATen/ops/mm_cuda_dispatch.h>", "declares library operator kernels"),
    ("#include <ATen/CUDAFunctions.h>", "declares library operator kernels"),
    ("#include <ATen/native/Resize.h>", "declares library operator kernels"),
    ("#include <ATen/cuda/CUDABlas.h>", "declares library operator kernels"),
    ("void f(at::Tensor a, at::Tensor o) { at::cuda::mm_out(o, a, a); }", "outside the dispatcher"),
    ("void f(at::Tensor a) { at::native::mm(a, a); }", "outside the dispatcher"),
    ("void f(at::Tensor a) { at::_ops::mm::redispatch(c10::DispatchKeySet(), a, a); }", "outside the dispatcher"),
    ("void f() { auto& d = c10::Dispatcher::singleton(); }", "outside the dispatcher"),
    ("void f(at::Tensor a) { torch::matmul(a, a); }", "not allowed"),
])
def test_hip_candidates_cannot_call_library_kernels_outside_the_dispatcher(bad, message):
    with pytest.raises(PolicyError, match=message):
        audit_hip_sources({"kernel.hip": HIP_OK + bad + "\n", "util.h": ""}, "kernel.hip")


def test_hip_device_and_stream_helpers_stay_allowed():
    allowed = ("#include <ATen/hip/HIPContext.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <ATen/Dispatch.h>\n"
               "int sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }\n"
               "hipStream_t s() { return at::hip::getCurrentHIPStream(); }\n")
    audit_hip_sources({"kernel.hip": HIP_OK + allowed, "util.h": ""}, "kernel.hip")


def test_hipified_stream_guard_and_device_utility_spellings_stay_allowed():
    allowed = ("#include <ATen/hip/Exceptions.h>\n#include <ATen/hip/Atomic.cuh>\n#include <ATen/cuda/DeviceUtils.cuh>\n"
               "#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>\n"
               "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n"
               "hipStream_t s() { return at::hip::getCurrentHIPStreamMasqueradingAsCUDA(); }\n"
               "void g(const at::Tensor& x) { const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device()); }\n")
    audit_hip_sources({"kernel.hip": HIP_OK + allowed, "util.h": ""}, "kernel.hip")


@pytest.mark.parametrize("bad, message", [
    # Work on another stream escapes same-stream timing.
    ("void f(hipStream_t s) { at::cuda::CUDAStreamGuard g(s); }", "outside the dispatcher"),
    ("void f(hipStream_t s) { at::hip::HIPStreamGuardMasqueradingAsCUDA g(s); }", "outside the dispatcher"),
    ("void f() { auto s = at::cuda::getStreamFromPool(); }", "outside the dispatcher"),
    # Exceptions.h brings in more library headers; their calls stay rejected.
    ("void f() { hipsparseCreate(nullptr); }", "not allowed"),
    ("void f() { hipsolverDnCreate(nullptr); }", "not allowed"),
    ("#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>", "library operator kernels"),
])
def test_stream_switches_and_other_library_calls_stay_rejected(bad, message):
    with pytest.raises(PolicyError, match=message):
        audit_hip_sources({"kernel.hip": HIP_OK + bad + "\n", "util.h": ""}, "kernel.hip")


def test_every_hip_baseline_passes_the_candidate_audit():
    # A baseline is also the starting candidate when the run targets its language.
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "tasks"
    baselines = sorted(p.parent for p in root.glob("*/*/*/*/baseline/kernel.hip"))
    assert baselines
    for baseline in baselines:
        sources = {p.relative_to(baseline).as_posix(): p.read_text(encoding="utf-8")
                   for p in baseline.rglob("*") if p.is_file() and p.suffix in (".hip", ".h", ".hpp", ".cuh")}
        audit_hip_sources(sources, "kernel.hip")


def test_runtime_audit_allows_preparation_and_rejects_compute():
    x = torch.randn(8)
    with preparation_only("candidate"):
        y = torch.empty_like(x)
        y.copy_(x)
        x.view(2, 4).permute(1, 0)
    with pytest.raises(RuntimeError, match="aten::add"):
        with preparation_only("candidate"):
            x + 1
    with pytest.raises(RuntimeError, match="_to_copy"):
        with preparation_only("candidate"):
            x.to(torch.bfloat16)


def test_baseline_relative_bound_tracks_the_baseline_error():
    torch.manual_seed(0)
    reference = torch.randn(64, 256, dtype=torch.float64)
    baseline = reference.to(torch.bfloat16)
    declaration = {"bound": "baseline_relative", "k": 2.0}
    same = compare_tensor(declaration, baseline.clone(), reference, dtype_name="bfloat16", baseline=baseline)
    assert same.passed
    shifted = (reference * (1 + 16 * 2.0 ** -8)).to(torch.bfloat16)
    assert not compare_tensor(declaration, shifted, reference, dtype_name="bfloat16", baseline=baseline).passed
    unwritten = baseline.clone()
    unwritten[0, 0] = float("nan")
    result = compare_tensor(declaration, unwritten, reference, dtype_name="bfloat16", baseline=baseline)
    assert not result.passed and result.nonfinite_mismatches == 1


def test_integer_outputs_compare_exactly():
    reference = torch.arange(10, dtype=torch.int64)
    actual = reference.to(torch.int32)
    assert compare_tensor({"bound": "exact"}, actual, reference, dtype_name="int32").passed
    actual[3] += 1
    assert not compare_tensor({"bound": "exact"}, actual, reference, dtype_name="int32").passed


def test_fp8_control_counts_as_applied_only_when_it_changes_an_output():
    from types import SimpleNamespace

    from src.tools.arena_eval.controls import _mutants

    contract = SimpleNamespace(inputs={"x": {"kind": "tensor"}}, mutable=(), checked_tensors=("out",))

    def baseline(*, x, out):
        out.copy_(x)

    integer = _mutants(contract, baseline)
    integer["fp8_rounded"](x=torch.arange(8, dtype=torch.int32), out=torch.empty(8, dtype=torch.int32))
    integer["fp8_rounded"](x=torch.ones(8), out=torch.empty(8))  # 1.0 is exact in fp8
    assert "fp8_rounded" not in integer["_effective"]
    floating = _mutants(contract, baseline)
    floating["fp8_rounded"](x=torch.full((8,), 1.0 + 2 ** -10), out=torch.empty(8))
    assert "fp8_rounded" in floating["_effective"]


def test_hip_loader_ignores_license_and_documentation_files():
    from src.tools.arena_eval.loader import _is_documentation

    assert all(_is_documentation(p) for p in ("LICENSE", "LICENSE.MIT", "NOTICE", "README.md", "sub/COPYING"))
    assert not any(_is_documentation(p) for p in ("kernel.hip", "license_utils.h", "notes.hpp"))


def test_float32_probe_serves_float64_requests_in_float32_and_restores_torch():
    from src.tools.arena_eval.controls import _float32_torch

    x = torch.ones(3, dtype=torch.bfloat16)
    with _float32_torch():
        assert torch.zeros(2, dtype=torch.float64).dtype == torch.float32
        assert x.double().dtype == torch.float32 and x.to(torch.double).dtype == torch.float32
    assert torch.float64 is not torch.float32 and x.double().dtype == torch.float64
    assert torch.zeros(1, dtype=torch.float64).dtype == torch.float64


def test_perturbing_controls_that_change_nothing_are_not_counted_as_applied():
    from types import SimpleNamespace

    from src.tools.arena_eval.controls import _mutants

    contract = SimpleNamespace(inputs={"x": {"kind": "tensor"}}, mutable=(), checked_tensors=("out",))

    def baseline(*, x, out):
        out.copy_(x)

    scalar = _mutants(contract, baseline)
    scalar["half_written"](x=torch.tensor(3.0), out=torch.empty(()))
    scalar["zeros"](x=torch.zeros(4), out=torch.empty(4))
    scalar["shifted_16_ulp"](x=torch.zeros(4), out=torch.empty(4))
    assert not scalar["_effective"] & {"half_written", "zeros", "shifted_16_ulp"}
    vector = _mutants(contract, baseline)
    vector["half_written"](x=torch.arange(1.0, 5.0), out=torch.empty(4))
    vector["zeros"](x=torch.arange(1.0, 5.0), out=torch.empty(4))
    vector["shifted_16_ulp"](x=torch.arange(1.0, 5.0), out=torch.empty(4))
    assert {"half_written", "zeros", "shifted_16_ulp"} <= vector["_effective"]


def test_short_rows_use_the_whole_outputs_baseline_rms():
    torch.manual_seed(0)
    reference = torch.randn(4096, 1, dtype=torch.float64)
    baseline = (reference + 1e-6 * torch.randn_like(reference)).float()
    baseline[0, 0] = reference[0, 0].float()  # the baseline happens to be exact on one row
    candidate = (reference + 1e-6 * torch.randn_like(reference)).float()
    result = compare_tensor({"bound": "baseline_relative", "k": 8}, candidate, reference, dtype_name="float32",
                            baseline=baseline)
    assert result.passed
    shifted = (reference * (1 + 64 * 2.0 ** -24) + 1e-4).float()
    assert not compare_tensor({"bound": "baseline_relative", "k": 8}, shifted, reference, dtype_name="float32",
                              baseline=baseline).passed


def test_bias_fallback_allows_the_baseline_bias_plus_sampling_noise():
    import math

    torch.manual_seed(1)
    reference = torch.randn(4096, 64, dtype=torch.float64)
    n, sigma = reference.numel(), 1e-10

    def draw(mean):
        noise = torch.randn_like(reference)
        return reference + sigma * (noise - noise.mean()) + mean * sigma / math.sqrt(n)

    baseline, candidate = draw(0.5), draw(7.0)  # z test fails (7 > 5), within 8 * 0.5 + 5 sampling sigmas
    declaration = {"bound": "baseline_relative", "k": 8}
    assert compare_tensor(declaration, candidate, reference, dtype_name="float64", baseline=baseline).passed
    assert not compare_tensor(declaration, draw(12.0), reference, dtype_name="float64", baseline=baseline).passed


def test_violation_fraction_admits_outliers_but_not_arbitrary_errors():
    torch.manual_seed(2)
    reference = torch.randn(256, 1024, dtype=torch.float64)
    baseline = reference.to(torch.bfloat16)
    declaration = {"bound": "baseline_relative", "max_violation_fraction": 0.01}
    outliers = reference.clone()
    picked = outliers.view(-1)[::200]
    signs = torch.ones_like(picked)
    signs[1::2] = -1  # outliers in both directions, as accumulation-order differences are
    picked += signs * (3 * (baseline.double() - reference).abs().view(-1)[::200] + 2 ** -8)
    outliers = outliers.to(torch.bfloat16)
    assert compare_tensor(declaration, outliers, reference, dtype_name="bfloat16", baseline=baseline).passed
    skipped = baseline.clone()
    skipped[:, -8:] = 0  # an unwritten tail of under 1% of the elements
    result = compare_tensor(declaration, skipped, reference, dtype_name="bfloat16", baseline=baseline)
    assert result.violation_fraction < 0.01 and not result.passed


def test_structural_zeros_do_not_count_toward_a_rows_error_sample():
    torch.manual_seed(2)
    reference = torch.randn(256, 64, 64, dtype=torch.float64).tril().reshape(-1, 64)  # triangular chunks
    nonzero = reference != 0

    def draw(scale):
        return (reference + scale * torch.randn_like(reference) * nonzero).float()

    baseline, candidate = draw(1e-6), draw(1e-6)
    baseline[:7] = reference[:7].float()  # rows of 1-7 nonzero elements where the baseline happened to be exact
    declaration = {"bound": "baseline_relative", "k": 8}
    assert compare_tensor(declaration, candidate, reference, dtype_name="float32", baseline=baseline).passed
    shifted = (reference * (1 + 64 * 2.0 ** -24) + 1e-4 * nonzero).float()
    assert not compare_tensor(declaration, shifted, reference, dtype_name="float32", baseline=baseline).passed


def test_calibration_probe_covers_every_non_exact_bound(monkeypatch):
    from types import SimpleNamespace

    from src.tools.arena_eval import controls

    def reference(x):
        y = x.to(torch.float64)
        return {"out": (y.exp() * 1.1).cumsum(-1), "idx": x.argmax(-1)}

    monkeypatch.setattr(controls.values, "reference_function", lambda contract: reference)
    torch.manual_seed(3)
    x = torch.randn(64, 512)
    expected = reference(x)
    data = SimpleNamespace(inputs=[SimpleNamespace(logical={"x": x})], references=[expected],
                           baselines=[{"out": expected["out"].float(), "idx": expected["idx"]}])
    case = SimpleNamespace(case_id="c0")

    def probe(out_bound):
        contract = SimpleNamespace(checked_tensors=("out", "idx"), element_order="fixed",
                                   compare={"out": out_bound, "idx": {"bound": "exact"}})
        return controls._calibration_probe(contract, case, data)

    strict = probe({"bound": "sqnr", "min_db": 200})
    assert [(item["output"], item["bound"], item["passed"]) for item in strict] == [("out", "sqnr", False)]
    assert strict[0]["sqnr_db"] < 200
    assert probe({"bound": "sqnr", "min_db": 100})[0]["passed"]
    assert probe({"bound": "tolerance", "rtol": 1e-3, "atol": 0})[0]["passed"]
