#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/gdn_l2norm_fwd.

Standalone harness for the GDN L2-norm forward Triton kernel.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton l2norm_fwd vs torch fp32 reference, assert close.
  --full-benchmark : graph-first GPU timing, write build/performance_report.json

Reference: y = x / sqrt(sum(x*x, dim=-1) + eps)  (pure L2 norm, no mean sub).
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

TASK_NAME = "triton2flydsl/sglang/gdn_l2norm_fwd"
SOURCE_FILE = os.path.join(TASK_DIR, "gdn_l2norm_fwd.py")

# Test configs: (rows, D). rows = flattened (B*T*H); D = head dim.
#   GDN q/k L2-norm => D=128, rows = T*H (real prefill: T=15360, H=16 => 245760).
TEST_SHAPES = [
    (2048, 128),
    (16384, 128),
    (131072, 128),
    (245760, 128),    # real Qwen3.5-35B prefill (T=15360 * H=16)
    (4096, 64),
    (4096, 256),
    (1024, 512),
    (1024, 1024),     # exercises the D>512 (kernel1) path
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5
EPS = 1e-6
DTYPE_NAME = os.environ.get("GDN_DTYPE", "bfloat16")


def load_module():
    spec = importlib.util.spec_from_file_location("gdn_l2norm_src", SOURCE_FILE)
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


def make_x(rows, D, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = getattr(torch, DTYPE_NAME)
    return torch.randn(rows, D, device=device, dtype=dtype)


def reference_l2norm(x, eps=EPS):
    import torch
    xf = x.float()
    rstd = torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)
    return (xf * rstd).to(x.dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "l2norm_fwd"), "Missing entry l2norm_fwd"
        assert hasattr(mod, "l2norm_fwd_kernel"), "Missing @triton.jit l2norm_fwd_kernel"
        assert hasattr(mod, "l2norm_fwd_kernel1"), "Missing @triton.jit l2norm_fwd_kernel1"
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
    for i, (rows, D) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            x = make_x(rows, D, "cuda", dtype)
            y = _retry_oom(lambda: mod.l2norm_fwd(x, EPS))
            torch.cuda.synchronize()
            ref = reference_l2norm(x, EPS)
            diff = (y.float() - ref.float()).abs().max().item()
            passed = bool(torch.allclose(y.float(), ref.float(), atol=atol, rtol=rtol))
            details.append({"shape_id": i + 1, "shape": [rows, D],
                            "max_diff": diff, "passed": passed})
            if not passed:
                return False, f"Shape {i+1} {TEST_SHAPES[i]}: max_diff={diff:.4e}", details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": [rows, D], "error": str(e)})
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
    for ti, (rows, D) in enumerate(TEST_SHAPES):
        params = {"rows": rows, "D": D}
        try:
            torch.manual_seed(42 + ti)
            x = make_x(rows, D, "cuda", dtype)

            def fn():
                mod.l2norm_fwd(x, EPS)

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
                print(f"  shape {d['shape_id']} {d['shape']}: max_diff={d['max_diff']:.4e} "
                      f"-> {'PASS' if d['passed'] else 'FAIL'}")
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
