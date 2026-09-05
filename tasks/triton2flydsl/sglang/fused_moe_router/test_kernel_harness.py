#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/fused_moe_router.

Standalone harness for sglang's fused MoE router Triton kernels
(`fused_moe_router_shim` -> cudacore / tensorcore kernels): logits = x @ W^T
(fp32), optional tanh logit-softcap, optional per-expert correction bias, then a
top-k selection whose weights are the (un-renormalized) softmax probabilities of
the selected experts. Returns (topk_weights [bs, topk] fp32, topk_ids int32).

The shim dispatches: cudacore (tl.sum) for small bs and <=8 experts, tensorcore
(tl.dot) for large bs / many experts (topk<=2). Both paths are exercised.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton router vs torch fp32 reference; topk_ids EXACT,
                     topk_weights close.
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

TASK_NAME = "triton2flydsl/sglang/fused_moe_router"
SOURCE_FILE = os.path.join(TASK_DIR, "fused_moe_router.py")

# num_experts must be a power of two (cudacore uses tl.arange(0, num_experts);
# tensorcore BLOCK_SIZE_N = max(num_experts, 16)). hidden % 256 == 0 for the
# tensorcore K-block. cap = moe_softcapping (Gemma uses 30.0; 0 disables softcap).
# Shapes chosen so both the cudacore and tensorcore branches of the shim are hit.
TEST_SHAPES = [
    {"bs": 128, "E": 8, "hidden": 2048, "topk": 2, "cap": 30.0, "bias": False},   # cudacore
    {"bs": 64, "E": 8, "hidden": 2048, "topk": 4, "cap": 30.0, "bias": False},    # cudacore topk>2
    {"bs": 512, "E": 8, "hidden": 2048, "topk": 2, "cap": 30.0, "bias": False},   # tensorcore (bs)
    {"bs": 256, "E": 16, "hidden": 4096, "topk": 2, "cap": 30.0, "bias": False},  # tensorcore (E>8)
    {"bs": 128, "E": 32, "hidden": 2048, "topk": 2, "cap": 30.0, "bias": True},   # tensorcore + bias
    {"bs": 64, "E": 8, "hidden": 2048, "topk": 2, "cap": 0.0, "bias": False},     # cudacore no softcap
    {"bs": 200, "E": 64, "hidden": 1024, "topk": 1, "cap": 30.0, "bias": False},  # tensorcore topk=1
    {"bs": 128, "E": 8, "hidden": 2048, "topk": 1, "cap": 30.0, "bias": True},    # cudacore topk=1 + bias
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5


def load_module():
    spec = importlib.util.spec_from_file_location("fused_moe_router_src", SOURCE_FILE)
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
    bs, E, h = cfg["bs"], cfg["E"], cfg["hidden"]
    # Scale down so raw logits are O(1) (avoid argmax ties being decided by noise).
    x = (torch.randn(bs, h, device=device, dtype=torch.bfloat16) * 0.1)
    w = (torch.randn(E, h, device=device, dtype=torch.bfloat16) * 0.1)
    bias = None
    if cfg["bias"]:
        bias = torch.randn(E, device=device, dtype=torch.float32) * 0.1
    return x, w, bias


def reference(x, w, cfg, bias):
    import torch
    logits = x.float() @ w.float().T  # [bs, E]
    cap = cfg["cap"]
    if cap != 0:
        logits = torch.tanh(logits / cap) * cap
    if bias is not None:
        logits = logits + bias.float()
    probs = torch.softmax(logits, dim=-1)
    vals, ids = torch.topk(logits, cfg["topk"], dim=-1)
    weights = torch.gather(probs, 1, ids)
    return weights, ids.to(torch.int32)


def _shape_of(cfg):
    return [cfg["bs"], cfg["E"], cfg["hidden"], cfg["topk"]]


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "fused_moe_router_shim"), \
            "Missing entry fused_moe_router_shim"
        assert hasattr(mod, "fused_moe_router_cudacore_kernel"), \
            "Missing @triton.jit fused_moe_router_cudacore_kernel"
        assert hasattr(mod, "fused_moe_router_tensorcore_kernel"), \
            "Missing @triton.jit fused_moe_router_tensorcore_kernel"
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
        try:
            torch.manual_seed(42 + i)
            x, w, bias = make_inputs(cfg, "cuda")
            tw, tid = _retry_oom(lambda: mod.fused_moe_router_shim(
                cfg["cap"], x, w, cfg["topk"], False, correction_bias=bias))
            torch.cuda.synchronize()
            rw, rid = reference(x, w, cfg, bias)
            finite = bool(torch.isfinite(tw).all().item())
            ids_match = bool(torch.equal(tid.cpu(), rid.cpu()))
            wdiff = (tw.float() - rw.float()).abs().max().item()
            w_close = bool(torch.allclose(tw.float(), rw.float(),
                                          atol=1e-3, rtol=1e-3))
            passed = finite and ids_match and w_close
            details.append({"shape_id": i + 1, "shape": shape, "cap": cfg["cap"],
                            "bias": cfg["bias"], "ids_match": ids_match,
                            "w_diff": wdiff, "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} {shape} cap={cfg['cap']} "
                               f"bias={cfg['bias']}: ids_match={ids_match} "
                               f"w_diff={wdiff:.4e} finite={finite}"), details
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
        params = {"shape": _shape_of(cfg), "cap": cfg["cap"], "bias": cfg["bias"]}
        try:
            torch.manual_seed(42 + ti)
            x, w, bias = make_inputs(cfg, "cuda")

            def fn():
                mod.fused_moe_router_shim(
                    cfg["cap"], x, w, cfg["topk"], False, correction_bias=bias)

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
                print(f"  shape {d['shape_id']} {d['shape']} cap={d['cap']} "
                      f"bias={d['bias']}: ids_match={d['ids_match']} "
                      f"w_diff={d['w_diff']:.4e} -> {'PASS' if d['passed'] else 'FAIL'}")
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
