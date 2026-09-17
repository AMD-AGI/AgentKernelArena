#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/fused_gdn_gating.

Standalone harness for the fused GDN input-gating Triton kernel
(fused_gdn_gating_kernel). Exercises the decode path (seq_len == 1).

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton fused_gdn_gating vs torch fp32 reference, assert close.
  --full-benchmark : graph-first GPU timing, write build/performance_report.json

Reference:
  g           = -exp(A_log) * softplus_beta(a + dt_bias)
  beta_output = sigmoid(b)
where softplus_beta(x) = (1/beta)*log(1+exp(beta*x)) if beta*x<=threshold else x.
g is fp32; beta_output is rounded to b's dtype (mirrors the kernel store).
"""
import sys
import os
import json
import time
import argparse
import importlib.util
from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events
from scripts.replay_checks import require_tensor_contract, require_unchanged, verify_timed_pair

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/sglang/fused_gdn_gating"
from task_runtime import candidate_relative_path
SOURCE_FILE = candidate_relative_path()
ENTRY = 'fused_gdn_gating'

BETA = 1.0
THRESHOLD = 20.0

# Test configs: (batch, num_heads). batch = decode tokens; num_heads = GDN linear
# attention value heads. Qwen3-Next: 32 (TP=1) / 16 (TP=2); Kimi-Linear similar.
TEST_SHAPES = [
    (1, 32),
    (8, 32),
    (32, 32),
    (128, 16),
    (256, 16),
    (512, 32),
    (64, 8),
    (16, 4),
    (128, 128),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5
DTYPE_NAME = 'bfloat16'  # protected suite dtype


def load_module():
    spec = importlib.util.spec_from_file_location("fused_gdn_gating_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
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


def make_test_data(batch, num_heads, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = getattr(torch, DTYPE_NAME)
    # a, b are projection outputs (bf16); A_log, dt_bias are fp32 parameters.
    a = torch.randn(batch, num_heads, device=device, dtype=dtype)
    b = torch.randn(batch, num_heads, device=device, dtype=dtype)
    A_log = torch.randn(num_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, device=device, dtype=torch.float32)
    return {"a": a, "b": b, "A_log": A_log, "dt_bias": dt_bias}


def reference_gating(inp, beta=BETA, threshold=THRESHOLD):
    import torch
    a, b = inp["a"], inp["b"]
    A_log, dt_bias = inp["A_log"], inp["dt_bias"]
    x = a.float() + dt_bias.float()[None, :]
    bx = beta * x
    softplus_x = torch.where(
        bx <= threshold, (1.0 / beta) * torch.log(1.0 + torch.exp(bx)), x
    )
    g = -torch.exp(A_log.float())[None, :] * softplus_x
    # mirror kernel store: g in fp32, beta_output rounded to b's dtype.
    beta_output = torch.sigmoid(b.float()).to(b.dtype).float()
    batch, num_heads = a.shape
    return g.view(1, batch, num_heads), beta_output.view(1, batch, num_heads)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "fused_gdn_gating"), "Missing entry fused_gdn_gating"
        assert hasattr(mod, "fused_gdn_gating_kernel"), \
            "Missing @triton.jit fused_gdn_gating_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def _checked_router_output(outputs, x, shape):
    import torch
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise AssertionError("Operator must return both result tensors")
    for output, dtype in zip(outputs, (torch.float32, torch.float32)):
        if (not isinstance(output, torch.Tensor) or output.shape != shape
                or output.dtype != dtype or output.device != x.device):
            raise AssertionError("Output shape/dtype/device contract mismatch")


def _compare_router_output(actual, expected, dtype):
    import torch
    _checked_router_output(actual, expected[0], tuple(expected[0].shape))
    if not all(bool(torch.isfinite(v).all()) for pair in (actual, expected) for v in pair):
        raise AssertionError("Non-finite operator/reference output")
    bo_atol = 1e-4 if dtype == torch.float32 else 2e-2
    bo_rtol = 1e-4 if dtype == torch.float32 else 1e-2
    g_ok = torch.allclose(actual[0].float(), expected[0].float(), atol=1e-3, rtol=1e-3)
    bo_ok = torch.allclose(actual[1].float(), expected[1].float(), atol=bo_atol, rtol=bo_rtol)
    if not (g_ok and bo_ok):
        raise AssertionError("Numerical mismatch: GDN gate or sigmoid output")


def _router_replay_validator(inp):
    inputs = tuple(inp.values())
    originals = tuple(v.clone() for v in inputs)
    expected = reference_gating(inp)
    def perturb():
        inp["a"].add_(1.)
        inp["b"].neg_()
    def replay_reference():
        return reference_gating(inp)
    def compare(actual, expected):
        _compare_router_output(actual, expected, inp["a"].dtype)
    def validate(timed):
        return verify_timed_pair(timed, inputs=inputs, originals=originals,
                                 expected=expected, perturb=perturb,
                                 reference=replay_reference, compare=compare)
    return validate


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    dtype = getattr(torch, DTYPE_NAME)
    # g follows the fp32 compute path; beta_output is rounded to bf16/fp16.
    g_atol = 1e-3 if dtype == torch.float32 else 1e-3
    g_rtol = 1e-3
    bo_atol = 1e-4 if dtype == torch.float32 else 2e-2
    bo_rtol = 1e-4 if dtype == torch.float32 else 1e-2
    details = []
    for i, (batch, num_heads) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            inp = make_test_data(batch, num_heads, "cuda", dtype)
            protected_inputs = tuple(inp.values())
            originals = tuple(v.clone() for v in protected_inputs)
            g_t, bo_t = _retry_oom(lambda: mod.fused_gdn_gating(
                inp["A_log"], inp["a"], inp["b"], inp["dt_bias"], BETA, THRESHOLD))
            torch.cuda.synchronize()
            require_unchanged(protected_inputs, originals)
            _checked_router_output((g_t, bo_t), inp["a"], (1, batch, num_heads))
            g_r, bo_r = reference_gating(inp)
            g_diff = (g_t.float() - g_r.float()).abs().max().item()
            bo_diff = (bo_t.float() - bo_r.float()).abs().max().item()
            g_ok = bool(torch.allclose(g_t.float(), g_r.float(), atol=g_atol, rtol=g_rtol))
            bo_ok = bool(torch.allclose(bo_t.float(), bo_r.float(), atol=bo_atol, rtol=bo_rtol))
            passed = g_ok and bo_ok
            details.append({"shape_id": i + 1, "shape": [batch, num_heads],
                            "g_max_diff": g_diff, "beta_max_diff": bo_diff,
                            "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} {TEST_SHAPES[i]}: g_diff={g_diff:.4e} "
                               f"beta_diff={bo_diff:.4e}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": [batch, num_heads], "error": str(e)})
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
    for ti, (batch, num_heads) in enumerate(TEST_SHAPES):
        params = {"batch": batch, "num_heads": num_heads}
        try:
            torch.manual_seed(42 + ti)
            inp = make_test_data(batch, num_heads, "cuda", dtype)
            replay_validate = _router_replay_validator(inp)

            def fn():
                return mod.fused_gdn_gating(
                    inp["A_log"], inp["a"], inp["b"], inp["dt_bias"], BETA, THRESHOLD)

            _retry_oom(fn)
            for _ in range(WARMUP_ITERATIONS):
                fn()
            torch.cuda.synchronize()
            timed = TimedRun()
            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                fn, warmup=0, repetition=BENCHMARK_ITERATIONS, timed_run=timed
            )
            bench_meta.update(replay_validate(timed))
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": elapsed_ms,
                               **bench_meta,
                               "params": params})
        except Exception as error:
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": -1.0,
                               "benchmark_method": "benchmark_failed",
                               "benchmark_fallback_reason": "performance case failed: " + str(error),
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
                print(f"  shape {d['shape_id']} {d['shape']}: g_diff={d['g_max_diff']:.4e} "
                      f"beta_diff={d['beta_max_diff']:.4e} -> {'PASS' if d['passed'] else 'FAIL'}")
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
