#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/merge_state.

Standalone harness for sglang's attention-state merge Triton kernel
(`merge_state_triton` -> `merge_state_kernel`): the numerically stable
softmax-recombination of two partial flash-attention results (prefix v_a/s_a and
suffix v_b/s_b) into a single output v_merged plus merged LSE s_merged.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton merge_state vs torch fp32 reference, assert close.
  --full-benchmark : graph-first GPU timing, write build/performance_report.json

Reference (per token, head):
  m       = max(s_a, s_b)
  denom   = exp(s_a - m) + exp(s_b - m)
  v_merged = v_a * exp(s_a - m)/denom + v_b * exp(s_b - m)/denom
  s_merged = log(denom) + m
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

TASK_NAME = "triton2flydsl/sglang/merge_state"
SOURCE_FILE = os.path.join(TASK_DIR, "merge_state.py")

# Real flash-decoding combine shapes: [num_tokens, num_heads, head_size].
# head_size includes a non-power-of-2 case (192, DeepSeek-style) to exercise the
# PADDED_HEAD_SIZE masking path.
TEST_SHAPES = [
    {"N": 128, "H": 32, "D": 128, "dtype": "bf16"},
    {"N": 1, "H": 32, "D": 128, "dtype": "bf16"},
    {"N": 256, "H": 8, "D": 64, "dtype": "bf16"},
    {"N": 64, "H": 16, "D": 256, "dtype": "bf16"},
    {"N": 100, "H": 40, "D": 192, "dtype": "bf16"},  # non-pow2 head_size
    {"N": 512, "H": 64, "D": 128, "dtype": "fp16"},
    {"N": 33, "H": 12, "D": 128, "dtype": "fp32"},
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_module():
    spec = importlib.util.spec_from_file_location("merge_state_src", SOURCE_FILE)
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
    N, H, D = cfg["N"], cfg["H"], cfg["D"]
    p_out = torch.randn(N, H, D, device=device, dtype=dt)
    s_out = torch.randn(N, H, D, device=device, dtype=dt)
    # LSE values in a realistic range; fp32 as in flash-decoding metadata.
    p_lse = torch.randn(N, H, device=device, dtype=torch.float32) * 2.0
    s_lse = torch.randn(N, H, device=device, dtype=torch.float32) * 2.0
    return p_out, p_lse, s_out, s_lse


def reference_merge(p_out, p_lse, s_out, s_lse):
    import torch
    out_dtype = p_out.dtype
    p = p_lse.float()
    s = s_lse.float()
    neg_inf = float("-inf")
    p = torch.where(torch.isinf(p) & (p > 0), torch.full_like(p, neg_inf), p)
    s = torch.where(torch.isinf(s) & (s > 0), torch.full_like(s, neg_inf), s)
    m = torch.maximum(p, s)
    pe = torch.exp(p - m)
    se = torch.exp(s - m)
    denom = pe + se
    out_lse = torch.log(denom) + m
    p_scale = (pe / denom).unsqueeze(-1)
    s_scale = (se / denom).unsqueeze(-1)
    out = p_out.float() * p_scale + s_out.float() * s_scale
    return out.to(out_dtype), out_lse


def _shape_of(cfg):
    return [cfg["N"], cfg["H"], cfg["D"]]


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "merge_state_triton"), "Missing entry merge_state_triton"
        assert hasattr(mod, "merge_state_kernel"), \
            "Missing @triton.jit merge_state_kernel"
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
        # bf16/fp16 -> 1e-2; fp32 -> tight.
        atol = 1e-2 if cfg["dtype"] != "fp32" else 1e-4
        rtol = 1e-2 if cfg["dtype"] != "fp32" else 1e-4
        try:
            torch.manual_seed(42 + i)
            p_out, p_lse, s_out, s_lse = make_inputs(cfg, "cuda")
            o_t, lse_t = _retry_oom(lambda: mod.merge_state_triton(
                p_out, p_lse, s_out, s_lse))
            torch.cuda.synchronize()
            o_r, lse_r = reference_merge(p_out, p_lse, s_out, s_lse)
            finite = bool(torch.isfinite(o_t).all().item())
            diff = (o_t.float() - o_r.float()).abs().max().item()
            lse_diff = (lse_t.float() - lse_r.float()).abs().max().item()
            close = bool(torch.allclose(o_t.float(), o_r.float(), atol=atol, rtol=rtol))
            lse_close = bool(torch.allclose(
                lse_t.float(), lse_r.float(), atol=1e-3, rtol=1e-3))
            passed = finite and close and lse_close
            details.append({"shape_id": i + 1, "shape": shape, "dtype": cfg["dtype"],
                            "max_diff": diff, "lse_diff": lse_diff, "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} {shape} ({cfg['dtype']}): "
                               f"max_diff={diff:.4e} lse_diff={lse_diff:.4e} "
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
            p_out, p_lse, s_out, s_lse = make_inputs(cfg, "cuda")

            def fn():
                mod.merge_state_triton(p_out, p_lse, s_out, s_lse)

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
                      f"max_diff={d['max_diff']:.4e} lse_diff={d['lse_diff']:.4e} "
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
