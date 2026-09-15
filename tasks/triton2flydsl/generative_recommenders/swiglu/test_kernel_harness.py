#!/usr/bin/env python3
"""Task runner for triton2flydsl/generative_recommenders/swiglu.

Self-contained harness mirroring the triton2flydsl template:
  - compile      : ast-parse + import the standalone source, assert entry/kernel symbols
  - correctness  : run the Triton kernel on TEST_SHAPES, assert finite output AND
                   closeness to a trivial inline torch reference (bf16/fp16 gate)
  - performance  : graph-first GPU timing, write build/performance_report.json

Fused SwiGLU forward: out = silu(x @ w_gate^T) * (x @ w_up^T). Public entry:
`triton_swiglu_fwd(x, w_gate, w_up)`; @triton.jit kernel: `_swiglu_fwd_kernel`.
The Triton kernel IS the reference target; the inline torch closeness check is the
correctness gate (no separate torch reference file is shipped).

GPU may be shared; kernel launches retry with backoff on transient CUDA/HIP OOM.
"""
import sys
import os
import json
import time
import argparse
import importlib.util
from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events
from scripts.replay_checks import require_tensor_contract, require_unchanged, verify_timed_run

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/generative_recommenders/swiglu"
from task_runtime import candidate_relative_path
SOURCE_FILE = candidate_relative_path()
ENTRY = 'triton_swiglu_fwd'

# Test configurations: (M, N, K)
#   M = rows (batch_size * seq_len)
#   N = output / hidden dim
#   K = input / reduction dim
TEST_SHAPES = [
    (256, 512, 256),
    (512, 1024, 512),
    (1024, 2048, 1024),
    (200, 384, 320),    # unaligned M/N/K
    (1, 512, 256),      # single row
    (2048, 512, 768),   # tall
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

MAX_OOM_RETRIES = 5
# bf16 GEMM tolerance (upstream norm/GEMM closeness floor).
ATOL = 1e-1
RTOL = 1e-2
PASS_FRACTION = 0.999


def load_module():
    spec = importlib.util.spec_from_file_location("swiglu_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return ("out of memory" in msg) or ("hip error: out of memory" in msg) or (
        "cuda error: out of memory" in msg
    )


def _retry_oom(fn):
    import torch
    delay = 1.0
    for attempt in range(MAX_OOM_RETRIES):
        try:
            return fn()
        except RuntimeError as e:
            if _is_oom(e) and attempt < MAX_OOM_RETRIES - 1:
                torch.cuda.empty_cache()
                time.sleep(delay)
                delay *= 2.0
                continue
            raise


def make_test_data(M, N, K, device="cuda", dtype=None):
    """Build (x [M,K], w_gate [N,K], w_up [N,K]) for triton_swiglu_fwd."""
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    scale = 1.0 / (K ** 0.5)
    x = torch.randn(M, K, device=device, dtype=dtype)
    w_gate = (torch.randn(N, K, device=device, dtype=dtype) * scale)
    w_up = (torch.randn(N, K, device=device, dtype=dtype) * scale)
    return x, w_gate, w_up


def _torch_ref(x, w_gate, w_up):
    """Trivial fp32 SwiGLU reference: silu(x@Wg^T) * (x@Wu^T)."""
    import torch
    xf = x.float()
    gate = xf @ w_gate.float().t()
    up = xf @ w_up.float().t()
    silu = gate * torch.sigmoid(gate)
    return (silu * up).to(x.dtype)


def _close(ref, out):
    import torch
    ref = ref.float()
    out = out.float()
    close = torch.isclose(out, ref, atol=ATOL, rtol=RTOL)
    frac = close.float().mean().item()
    denom = ref.abs().max().item()
    norm = (out - ref).abs().max().item() / denom if denom > 0 else (out - ref).abs().max().item()
    return (frac >= PASS_FRACTION) or (norm <= 1e-2), frac, norm


def _call_kernel(mod, x, w_gate, w_up):
    return _retry_oom(lambda: mod.triton_swiglu_fwd(x, w_gate, w_up))


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "triton_swiglu_fwd"), "Missing triton_swiglu_fwd entry"
        assert hasattr(mod, "_swiglu_fwd_kernel"), "Missing _swiglu_fwd_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def _checked_gr_output(out, x, columns):
    import torch
    if (not isinstance(out, torch.Tensor) or out.shape != (x.shape[0], columns)
            or out.dtype != x.dtype or out.device != x.device):
        raise AssertionError("GR output shape/dtype/device contract mismatch")


def _compare_gr_output(actual, expected):
    import torch
    require_tensor_contract(actual, expected, dtype=torch.bfloat16)
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("Non-finite GR/reference output")
    close, fraction, normalized = _close(expected, actual)
    if not close:
        raise AssertionError(f"Numerical mismatch: fraction={fraction}, normalized_max_error={normalized}")


def _gr_replay_validator(x, w_gate, w_up):
    inputs = tuple(v for v in (x, w_gate, w_up,) if v is not None)
    originals = tuple(v.clone() for v in inputs)
    expected = _torch_ref(x, w_gate, w_up)
    def perturb():
        w_up.neg_()
    def reference():
        return _torch_ref(x, w_gate, w_up)
    def validate(timed):
        return verify_timed_run(timed, inputs=inputs, originals=originals,
                                expected=expected, perturb=perturb,
                                reference=reference, compare=_compare_gr_output)
    return validate


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    device = "cuda"
    dtype = torch.bfloat16
    details = []

    for i, (M, N, K) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x, w_gate, w_up = make_test_data(M, N, K, device, dtype)
            protected_inputs = tuple(v for v in (x, w_gate, w_up,) if v is not None)
            originals = tuple(v.clone() for v in protected_inputs)
            result = _call_kernel(mod, x, w_gate, w_up)
            torch.cuda.synchronize()
            require_unchanged(protected_inputs, originals)
            _checked_gr_output(result, x, w_gate.shape[0])

            finite = bool(torch.isfinite(result.float()).all().item())
            shape_ok = list(result.shape) == [M, N]
            ref = _torch_ref(x, w_gate, w_up)
            close, frac, norm = _close(ref, result)
            passed = finite and shape_ok and close
            details.append({
                "shape_id": i + 1,
                "shape": [M, N, K],
                "out_shape": list(result.shape),
                "finite": finite,
                "close_frac": round(frac, 5),
                "norm_err": round(norm, 5),
                "passed": bool(passed),
            })
            if not passed:
                if not finite:
                    reason = "non-finite output"
                elif not shape_ok:
                    reason = f"bad out shape {list(result.shape)}"
                else:
                    reason = f"closeness fail frac={frac:.4f} norm={norm:.4f}"
                return False, f"Shape {i+1} {TEST_SHAPES[i]}: {reason}", details
        except Exception as e:
            details.append({
                "shape_id": i + 1, "shape": [M, N, K], "error": str(e),
            })
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}", details

    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.bfloat16
    test_cases = []

    for test_idx, (M, N, K) in enumerate(TEST_SHAPES):
        params = {"M": M, "N": N, "K": K}
        try:
            torch.manual_seed(42 + test_idx)
            x, w_gate, w_up = make_test_data(M, N, K, device, dtype)
            replay_validate = _gr_replay_validator(x, w_gate, w_up)

            def launch():
                return mod.triton_swiglu_fwd(x, w_gate, w_up)

            _retry_oom(launch)

            for _ in range(WARMUP_ITERATIONS):
                launch()
            torch.cuda.synchronize()

            timed = TimedRun()
            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                launch,
                warmup=0,
                repetition=BENCHMARK_ITERATIONS, timed_run=timed,
            )

            bench_meta.update(replay_validate(timed))
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **bench_meta,
                "params": params,
            })
        except Exception as error:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "benchmark_method": "benchmark_failed",
                "benchmark_fallback_reason": "performance case failed: " + str(error),
                "params": params,
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("--compile", dest="mode", action="store_const", const="compile")
    parser.add_argument("--correctness", dest="mode", action="store_const", const="correctness")
    parser.add_argument("--full-benchmark", dest="mode", action="store_const", const="performance")
    parser.add_argument("--benchmark", dest="mode", action="store_const", const="performance")
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump({"status": "ok" if ok else "fail", "error": err}, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err, details = run_correctness()
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump({"status": "ok" if ok else "fail", "error": err,
                       "num_shapes": len(TEST_SHAPES), "details": details}, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for d in details:
            if "finite" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: out={d['out_shape']} "
                      f"finite={d['finite']} close_frac={d['close_frac']} "
                      f"norm_err={d['norm_err']} -> {'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: ERROR {d['error']}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total = sum(c["execution_time_ms"] for c in test_cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
