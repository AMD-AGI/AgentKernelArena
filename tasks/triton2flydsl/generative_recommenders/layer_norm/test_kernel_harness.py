#!/usr/bin/env python3
"""Task runner for triton2flydsl/generative_recommenders/layer_norm.

Self-contained harness mirroring the triton2flydsl template:
  - compile      : ast-parse + import the standalone source, assert entry/kernel symbols
  - correctness  : run the Triton kernel on TEST_SHAPES, assert finite output AND
                   closeness to a trivial inline torch reference (fp32 layernorm)
  - performance  : graph-first GPU timing, write build/performance_report.json

Multi-row LayerNorm forward over the feature dim: weighted affine when (weight,
bias) are given, else unweighted (mean-subtract + rstd-scale). Public entry:
`triton_layer_norm(x, weight, bias, eps)`; @triton.jit kernels: `_layer_norm_fwd`
and `_weighted_layer_norm_fwd`. The Triton kernel is the reference target; the
inline torch closeness check is the correctness gate (no torch reference file).

GPU may be shared; kernel launches retry with backoff on transient CUDA/HIP OOM.
"""
import sys
import os
import json
import time
import argparse
import importlib.util
from _aka_benchmark import benchmark_cuda_graph_or_events

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/generative_recommenders/layer_norm"
SOURCE_FILE = os.path.join(TASK_DIR, "layer_norm.py")

# Test configurations: (N, D, learnable)
#   N         = number of rows
#   D         = feature dim normalized over
#   learnable = affine (weight + bias) vs unweighted centered layernorm
TEST_SHAPES = [
    (256, 256, True),
    (1024, 512, True),
    (2048, 384, False),     # unweighted (centered) path
    (300, 200, True),       # unaligned D
    (4, 512, False),        # tiny N, unweighted
    (4096, 128, True),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
EPS = 1e-6

MAX_OOM_RETRIES = 5
ATOL = 1e-2
RTOL = 1e-2
PASS_FRACTION = 0.999


def load_module():
    spec = importlib.util.spec_from_file_location("layer_norm_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
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


def make_test_data(N, D, learnable, device="cuda", dtype=None):
    """Build (x [N,D], weight [D] or None, bias [D] or None)."""
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    x = torch.randn(N, D, device=device, dtype=dtype)
    if learnable:
        weight = torch.empty(D, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        bias = torch.empty(D, device=device, dtype=dtype).uniform_(-1.0, 1.0)
    else:
        weight = None
        bias = None
    return x, weight, bias


def _torch_ref(x, weight, bias, eps):
    """Trivial fp32 LayerNorm reference over the last dim (no affine if weight None)."""
    import torch
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    y = (xf - mean) / torch.sqrt(var + eps)
    if weight is not None:
        y = y * weight.float().unsqueeze(0) + bias.float().unsqueeze(0)
    return y.to(x.dtype)


def _close(ref, out):
    import torch
    ref = ref.float()
    out = out.float()
    close = torch.isclose(out, ref, atol=ATOL, rtol=RTOL)
    frac = close.float().mean().item()
    denom = ref.abs().max().item()
    norm = (out - ref).abs().max().item() / denom if denom > 0 else 0.0
    return (frac >= PASS_FRACTION) or (norm <= 1e-2), frac, norm


def _call_kernel(mod, x, weight, bias):
    return _retry_oom(lambda: mod.triton_layer_norm(x, weight, bias, EPS))


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "triton_layer_norm"), "Missing triton_layer_norm entry"
        assert hasattr(mod, "_layer_norm_fwd"), "Missing _layer_norm_fwd"
        assert hasattr(mod, "_weighted_layer_norm_fwd"), "Missing _weighted_layer_norm_fwd"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    device = "cuda"
    dtype = torch.bfloat16
    details = []

    for i, (N, D, learnable) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x, weight, bias = make_test_data(N, D, learnable, device, dtype)
            result = _call_kernel(mod, x, weight, bias)
            torch.cuda.synchronize()

            finite = bool(torch.isfinite(result.float()).all().item())
            shape_ok = list(result.shape) == [N, D]
            ref = _torch_ref(x, weight, bias, EPS)
            close, frac, norm = _close(ref, result)
            passed = finite and shape_ok and close
            details.append({
                "shape_id": i + 1,
                "shape": [N, D, learnable],
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
                "shape_id": i + 1, "shape": [N, D, learnable], "error": str(e),
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

    for test_idx, (N, D, learnable) in enumerate(TEST_SHAPES):
        params = {"N": N, "D": D, "learnable": learnable}
        try:
            torch.manual_seed(42 + test_idx)
            x, weight, bias = make_test_data(N, D, learnable, device, dtype)

            def launch():
                return mod.triton_layer_norm(x, weight, bias, EPS)

            _retry_oom(launch)

            for _ in range(WARMUP_ITERATIONS):
                launch()
            torch.cuda.synchronize()

            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                launch,
                warmup=0,
                repetition=BENCHMARK_ITERATIONS,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **bench_meta,
                "params": params,
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "benchmark_method": "benchmark_failed",
                "benchmark_fallback_reason": "performance case failed before timing completed",
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
