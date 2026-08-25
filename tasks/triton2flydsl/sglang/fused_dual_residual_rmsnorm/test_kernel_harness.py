#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/fused_dual_residual_rmsnorm.

Standalone harness for sglang's fused dual-residual RMSNorm Triton kernel
(`fused_dual_residual_rmsnorm` -> `fused_dual_residual_rmsnorm_kernel`):
  mid    = residual + RMSNorm1(x) * weight1
  output = RMSNorm2(mid) * weight2
returning (output, mid). RMS is computed in fp32; the first norm is cast to the
residual dtype before the add.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton vs torch fp32 reference, assert close (output AND mid).
  --full-benchmark : graph-first GPU timing, write build/performance_report.json
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

TASK_NAME = "triton2flydsl/sglang/fused_dual_residual_rmsnorm"
SOURCE_FILE = os.path.join(TASK_DIR, "fused_dual_residual_rmsnorm.py")
EPS = 1e-6

# [batch_size, hidden_dim] real transformer norm shapes (Llama/Qwen hidden dims).
TEST_SHAPES = [
    {"bs": 16, "hidden": 4096, "dtype": "bf16"},
    {"bs": 128, "hidden": 4096, "dtype": "bf16"},
    {"bs": 1, "hidden": 8192, "dtype": "bf16"},
    {"bs": 64, "hidden": 5120, "dtype": "bf16"},
    {"bs": 256, "hidden": 2048, "dtype": "bf16"},
    {"bs": 32, "hidden": 3072, "dtype": "bf16"},   # non-pow2 hidden
    {"bs": 8, "hidden": 4096, "dtype": "fp16"},
    {"bs": 4, "hidden": 1024, "dtype": "fp32"},
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fused_dual_residual_rmsnorm_src", SOURCE_FILE)
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


def make_inputs(cfg, device="cuda"):
    import torch
    dt = getattr(torch, _DTYPES[cfg["dtype"]])
    bs, h = cfg["bs"], cfg["hidden"]
    x = torch.randn(bs, h, device=device, dtype=dt)
    residual = torch.randn(bs, h, device=device, dtype=dt)
    w1 = torch.randn(h, device=device, dtype=dt)
    w2 = torch.randn(h, device=device, dtype=dt)
    return x, residual, w1, w2


def reference(x, residual, w1, w2, eps=EPS):
    import torch
    dt = x.dtype
    a = x.float()
    rms1 = torch.sqrt((a * a).mean(dim=-1, keepdim=True) + eps)
    inner = (a / rms1 * w1.float()).to(dt)
    mid = residual + inner  # in input dtype, this becomes the new residual
    a2 = mid.float()
    rms2 = torch.sqrt((a2 * a2).mean(dim=-1, keepdim=True) + eps)
    out = (a2 / rms2 * w2.float()).to(dt)
    return out, mid


def _shape_of(cfg):
    return [cfg["bs"], cfg["hidden"]]


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "fused_dual_residual_rmsnorm"), \
            "Missing entry fused_dual_residual_rmsnorm"
        assert hasattr(mod, "fused_dual_residual_rmsnorm_kernel"), \
            "Missing @triton.jit fused_dual_residual_rmsnorm_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        shape = _shape_of(cfg)
        atol = 1e-2 if cfg["dtype"] != "fp32" else 1e-4
        rtol = 1e-2 if cfg["dtype"] != "fp32" else 1e-4
        try:
            torch.manual_seed(42 + i)
            x, residual, w1, w2 = make_inputs(cfg, "cuda")
            o_t, mid_t = _retry_oom(lambda: mod.fused_dual_residual_rmsnorm(
                x, residual, w1, w2, EPS))
            torch.cuda.synchronize()
            o_r, mid_r = reference(x, residual, w1, w2)
            finite = bool(torch.isfinite(o_t).all().item())
            diff = (o_t.float() - o_r.float()).abs().max().item()
            mid_diff = (mid_t.float() - mid_r.float()).abs().max().item()
            close = bool(torch.allclose(o_t.float(), o_r.float(), atol=atol, rtol=rtol))
            mid_close = bool(torch.allclose(
                mid_t.float(), mid_r.float(), atol=atol, rtol=rtol))
            passed = finite and close and mid_close
            details.append({"shape_id": i + 1, "shape": shape, "dtype": cfg["dtype"],
                            "max_diff": diff, "mid_diff": mid_diff, "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} {shape} ({cfg['dtype']}): "
                               f"max_diff={diff:.4e} mid_diff={mid_diff:.4e} "
                               f"finite={finite}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": shape, "error": str(e)})
            return False, f"Shape {i+1} {shape}: exception: {e}", details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    test_cases = []
    for ti, cfg in enumerate(TEST_SHAPES):
        params = {"shape": _shape_of(cfg), "dtype": cfg["dtype"]}
        try:
            torch.manual_seed(42 + ti)
            x, residual, w1, w2 = make_inputs(cfg, "cuda")

            def fn():
                mod.fused_dual_residual_rmsnorm(x, residual, w1, w2, EPS)

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
                print(f"  shape {d['shape_id']} {d['shape']} {d['dtype']}: "
                      f"max_diff={d['max_diff']:.4e} mid_diff={d['mid_diff']:.4e} "
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
