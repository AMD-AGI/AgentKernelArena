#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/experts_combine.

Standalone harness for sglang's MoE/MLP experts-combine Triton kernel
(`experts_combine_triton` -> `experts_combine_kernel`):
  out = (sum_k moe_hidden_states[:, k] + mlp_hidden_states) / sqrt(2)
moe_hidden_states is [num_tokens, combine_k, hidden_dim] (combine_k expert
outputs) or [num_tokens, hidden_dim] (combine_k = 1); mlp_hidden_states is
[num_tokens, hidden_dim].

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton vs torch fp32 reference, assert close.
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

TASK_NAME = "triton2flydsl/sglang/experts_combine"
SOURCE_FILE = os.path.join(TASK_DIR, "experts_combine.py")
SQRT2 = 1.4142135623730951

# [num_tokens, combine_k, hidden_dim]; combine_k=1 => 2D pre-combined path.
TEST_SHAPES = [
    {"tokens": 128, "combine_k": 1, "hidden": 4096, "dtype": "bf16"},
    {"tokens": 128, "combine_k": 2, "hidden": 4096, "dtype": "bf16"},
    {"tokens": 64, "combine_k": 4, "hidden": 2048, "dtype": "bf16"},
    {"tokens": 256, "combine_k": 8, "hidden": 1024, "dtype": "bf16"},
    {"tokens": 1, "combine_k": 2, "hidden": 7168, "dtype": "bf16"},  # DeepSeek hidden
    {"tokens": 32, "combine_k": 2, "hidden": 3072, "dtype": "bf16"},  # non-pow2 hidden
    {"tokens": 16, "combine_k": 2, "hidden": 4096, "dtype": "fp16"},
    {"tokens": 8, "combine_k": 2, "hidden": 2048, "dtype": "fp32"},
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_module():
    spec = importlib.util.spec_from_file_location("experts_combine_src", SOURCE_FILE)
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
    t, k, h = cfg["tokens"], cfg["combine_k"], cfg["hidden"]
    if k == 1:
        moe = torch.randn(t, h, device=device, dtype=dt)
    else:
        moe = torch.randn(t, k, h, device=device, dtype=dt)
    mlp = torch.randn(t, h, device=device, dtype=dt)
    return moe, mlp


def reference(moe, mlp):
    dt = mlp.dtype
    if moe.dim() == 3:
        moe_sum = moe.float().sum(dim=1)
    else:
        moe_sum = moe.float()
    out = (moe_sum + mlp.float()) / SQRT2
    return out.to(dt)


def _shape_of(cfg):
    if cfg["combine_k"] == 1:
        return [cfg["tokens"], cfg["hidden"]]
    return [cfg["tokens"], cfg["combine_k"], cfg["hidden"]]


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "experts_combine_triton"), \
            "Missing entry experts_combine_triton"
        assert hasattr(mod, "experts_combine_kernel"), \
            "Missing @triton.jit experts_combine_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    # Normalized worst-element gate (the convention's bf16 elementwise gate):
    # the kernel reduces the top-k expert outputs in bf16, so a per-element ULP at
    # the summed magnitude can exceed a raw atol near zero-crossings; compare
    # max|ref-out| / max|ref| <= REL instead. fp32 path uses a tight raw band.
    REL = 1e-2
    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        shape = _shape_of(cfg)
        try:
            torch.manual_seed(42 + i)
            moe, mlp = make_inputs(cfg, "cuda")
            o_t = _retry_oom(lambda: mod.experts_combine_triton(moe, mlp))
            torch.cuda.synchronize()
            o_r = reference(moe, mlp)
            finite = bool(torch.isfinite(o_t).all().item())
            diff = (o_t.float() - o_r.float()).abs().max().item()
            denom = o_r.float().abs().max().item()
            if cfg["dtype"] == "fp32":
                close = bool(torch.allclose(
                    o_t.float(), o_r.float(), atol=1e-4, rtol=1e-4))
                rel = diff / denom if denom > 0 else diff
            else:
                rel = diff / denom if denom > 0 else diff
                close = rel <= REL
            passed = finite and close
            details.append({"shape_id": i + 1, "shape": shape, "dtype": cfg["dtype"],
                            "max_diff": diff, "rel": rel, "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} {shape} ({cfg['dtype']}): "
                               f"max_diff={diff:.4e} rel={rel:.4e} "
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
            moe, mlp = make_inputs(cfg, "cuda")

            def fn():
                mod.experts_combine_triton(moe, mlp)

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
                      f"max_diff={d['max_diff']:.4e} rel={d['rel']:.4e} "
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
