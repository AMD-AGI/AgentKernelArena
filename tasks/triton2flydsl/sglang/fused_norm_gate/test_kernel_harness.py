#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/fused_norm_gate.

Standalone harness for the fused (RMS/Layer)Norm + output-gate Triton kernels
(layer_norm_gated_fwd_kernel for D<=512, layer_norm_gated_fwd_kernel1 for D>512).

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton layer_norm_gated_fwd vs torch fp32 reference.
  --full-benchmark : graph-first GPU timing, write build/performance_report.json

Reference per row (no residual):
  x_hat = x*rstd (RMS) | (x-mean)*rstd (Layer);  y = x_hat*w (+b)
  swish/silu => y *= g*sigmoid(g);  sigmoid => y *= sigmoid(g)
  rstd = 1/sqrt(mean(var)+eps), fp32.
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

TASK_NAME = "triton2flydsl/sglang/fused_norm_gate"
SOURCE_FILE = os.path.join(TASK_DIR, "fused_norm_gate.py")
EPS = 1e-5

# Test configs: (T, D, is_rms_norm, activation, has_bias). GDN gated RMSNorm is
# applied per value head: D = head_v_dim (128/256); T = tokens * heads.
# Qwen3-Next / Kimi-Linear. D>512 exercises the kernel1 row path.
TEST_SHAPES = [
    (2048, 128, True, "swish", False),
    (4096, 128, True, "swish", False),
    (8192, 256, True, "swish", False),
    (2048, 512, True, "swish", False),
    (1024, 1024, True, "swish", False),   # kernel1 (D>512)
    (2048, 128, True, "sigmoid", False),
    (2048, 128, False, "swish", True),    # layernorm + bias
    (2048, 768, False, "swish", True),    # kernel1 layernorm + bias
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5
DTYPE_NAME = os.environ.get("GDN_DTYPE", "bfloat16")


def load_module():
    spec = importlib.util.spec_from_file_location("fused_norm_gate_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_oom(err):
    return "out of memory" in str(err).lower()


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


def make_test_data(T, D, has_bias, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = getattr(torch, DTYPE_NAME)
    x = torch.randn(T, D, device=device, dtype=dtype)
    g = torch.randn(T, D, device=device, dtype=dtype)
    weight = (torch.ones(D, device=device, dtype=dtype)
              + 0.1 * torch.randn(D, device=device, dtype=dtype))
    bias = 0.1 * torch.randn(D, device=device, dtype=dtype) if has_bias else None
    return {"x": x, "g": g, "weight": weight, "bias": bias}


def reference_norm_gate(inp, is_rms_norm, activation, eps=EPS):
    import torch
    x, g = inp["x"], inp["g"]
    weight, bias = inp["weight"], inp["bias"]
    xf = x.float()
    if is_rms_norm:
        var = (xf * xf).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_hat = xf * rstd
    else:
        mean = xf.mean(dim=-1, keepdim=True)
        xbar = xf - mean
        var = (xbar * xbar).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_hat = xbar * rstd
    y = x_hat
    if weight is not None:
        y = y * weight.float()[None, :]
    if bias is not None:
        y = y + bias.float()[None, :]
    gf = g.float()
    if activation in ("swish", "silu"):
        y = y * gf * torch.sigmoid(gf)
    elif activation == "sigmoid":
        y = y * torch.sigmoid(gf)
    return y.to(x.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "layer_norm_gated_fwd"), "Missing entry layer_norm_gated_fwd"
        assert hasattr(mod, "layer_norm_gated_fwd_kernel"), \
            "Missing @triton.jit layer_norm_gated_fwd_kernel"
        assert hasattr(mod, "layer_norm_gated_fwd_kernel1"), \
            "Missing @triton.jit layer_norm_gated_fwd_kernel1"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    dtype = getattr(torch, DTYPE_NAME)
    atol = 1e-4 if dtype == torch.float32 else 2e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2
    details = []
    for i, (T, D, is_rms, act, has_bias) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            inp = make_test_data(T, D, has_bias, "cuda", dtype)
            y_t = _retry_oom(lambda: mod.layer_norm_gated_fwd(
                x=inp["x"], g=inp["g"], weight=inp["weight"], bias=inp["bias"],
                activation=act, eps=EPS, residual=None, out_dtype=inp["x"].dtype,
                is_rms_norm=is_rms)[0])
            torch.cuda.synchronize()
            y_r = reference_norm_gate(inp, is_rms, act)
            diff = (y_t.float() - y_r.float()).abs().max().item()
            isclose = torch.isclose(y_t.float(), y_r.float(), atol=atol, rtol=rtol)
            err_ratio = (~isclose).float().mean().item()
            passed = err_ratio <= 0.02
            details.append({"shape_id": i + 1, "shape": [T, D],
                            "is_rms_norm": is_rms, "activation": act, "has_bias": has_bias,
                            "max_diff": diff, "err_ratio": err_ratio, "passed": bool(passed)})
            if not passed:
                return False, (f"Shape {i+1} {TEST_SHAPES[i]}: max_diff={diff:.4e} "
                               f"err_ratio={err_ratio:.4f}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": [T, D], "error": str(e)})
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}", details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    dtype = getattr(torch, DTYPE_NAME)
    test_cases = []
    for ti, (T, D, is_rms, act, has_bias) in enumerate(TEST_SHAPES):
        params = {"T": T, "D": D, "is_rms_norm": is_rms, "activation": act,
                  "has_bias": has_bias}
        try:
            torch.manual_seed(42 + ti)
            inp = make_test_data(T, D, has_bias, "cuda", dtype)

            def fn():
                mod.layer_norm_gated_fwd(
                    x=inp["x"], g=inp["g"], weight=inp["weight"], bias=inp["bias"],
                    activation=act, eps=EPS, residual=None, out_dtype=inp["x"].dtype,
                    is_rms_norm=is_rms)[0]

            _retry_oom(fn)
            for _ in range(WARMUP_ITERATIONS):
                fn()
            torch.cuda.synchronize()
            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                fn, warmup=0, repetition=BENCHMARK_ITERATIONS
            )
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": elapsed_ms,
                               **bench_meta,
                               "params": params})
        except Exception:
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": -1.0,
                               "benchmark_method": "benchmark_failed",
                               "benchmark_fallback_reason": "performance case failed before timing completed",
                               "params": params})
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("--compile", dest="mode", action="store_const", const="compile")
    parser.add_argument("--correctness", dest="mode", action="store_const", const="correctness")
    parser.add_argument("--full-benchmark", dest="mode", action="store_const", const="performance")
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(build_dir, "compile_report.json"), "w"), indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err, details = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err,
                   "num_shapes": len(TEST_SHAPES), "details": details},
                  open(os.path.join(build_dir, "correctness_report.json"), "w"), indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for d in details:
            if "passed" in d:
                print(f"  shape {d['shape_id']} {d['shape']} rms={d['is_rms_norm']} "
                      f"act={d['activation']} bias={d['has_bias']}: max_diff={d['max_diff']:.4e} "
                      f"err_ratio={d['err_ratio']:.4f} -> {'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: ERROR {d['error']}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        json.dump(test_cases, open(os.path.join(build_dir, "performance_report.json"), "w"), indent=2)
        if test_cases:
            total = sum(c["execution_time_ms"] for c in test_cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} case(s), total {total:.4f} ms")
            for c in test_cases:
                print(f"  {c['test_case_id']} {c['params']}: {c['execution_time_ms']:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
