#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/chunk_scaled_dot_kkt_fwd.

Standalone harness for the GDN scaled-dot K@K^T Triton kernel
(chunk_scaled_dot_kkt_fwd_kernel). Exercises the regular (non-varlen) path;
the per-chunk math is identical to the varlen branch, only offsets differ.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton chunk_scaled_dot_kkt_fwd vs torch fp32 reference.
  --full-benchmark : graph-first GPU timing, write build/performance_report.json

Reference per chunk/head:
  A = tril_strict( beta[:,None] * (k_c @ k_c^T) * safe_exp(g_c[:,None]-g_c[None,:]) )
where safe_exp(x) = exp(x) if x<=0 else 0; k_c is bf16 (fp32-accumulated matmul).
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

TASK_NAME = "triton2flydsl/sglang/chunk_scaled_dot_kkt_fwd"
SOURCE_FILE = os.path.join(TASK_DIR, "chunk_scaled_dot_kkt_fwd.py")
BT = 64  # CHUNK_SIZE

# Test configs: (B, T, Hg, H, K, use_g). real Qwen3-Next / Kimi-Linear GDN:
# K(=head_k_dim)=128, Hg/H = 16/32 (grouped when Hg<H). T multiple/non-multiple of 64.
TEST_SHAPES = [
    (1, 1024, 16, 16, 128, True),
    (1, 1024, 8, 16, 128, True),    # grouped (Hg < H)
    (2, 512, 16, 32, 128, True),
    (1, 2048, 32, 32, 128, True),
    (1, 1000, 16, 16, 128, True),   # T % BT != 0
    (4, 256, 16, 16, 128, True),
    (1, 1024, 16, 16, 64, True),    # K=64
    (1, 1024, 16, 16, 128, False),  # USE_G=False branch
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5
DTYPE_NAME = os.environ.get("GDN_DTYPE", "bfloat16")


def load_module():
    spec = importlib.util.spec_from_file_location("chunk_kkt_src", SOURCE_FILE)
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


def _chunk_local_cumsum(g, bt):
    import torch
    B, T, H = g.shape
    out = g.clone()
    for t0 in range(0, T, bt):
        t1 = min(t0 + bt, T)
        out[:, t0:t1] = torch.cumsum(g[:, t0:t1], dim=1)
    return out


def make_test_data(B, T, Hg, H, K, use_g, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = getattr(torch, DTYPE_NAME)
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    if use_g:
        g_raw = torch.nn.functional.logsigmoid(
            torch.randn(B, T, H, device=device, dtype=torch.float32)
        )
        g = _chunk_local_cumsum(g_raw, BT)
    else:
        g = None
    return {"B": B, "T": T, "Hg": Hg, "H": H, "K": K, "use_g": use_g,
            "k": k, "beta": beta, "g": g}


def reference_kkt(inp):
    import torch
    B, T, Hg, H, K = inp["B"], inp["T"], inp["Hg"], inp["H"], inp["K"]
    k, beta, g = inp["k"], inp["beta"], inp["g"]
    dtype = k.dtype
    rep = H // Hg
    NT = (T + BT - 1) // BT
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    for b in range(B):
        for hh in range(H):
            hg = hh // rep
            for it in range(NT):
                t0 = it * BT
                t1 = min(t0 + BT, T)
                L = t1 - t0
                k_c = k[b, t0:t1, hg].float()           # [L,K] (exact bf16 values)
                kk = k_c @ k_c.T                          # [L,L]
                b_A = kk
                if g is not None:
                    g_c = g[b, t0:t1, hh].float()        # [L]
                    diff = g_c[:, None] - g_c[None, :]
                    se = torch.where(diff <= 0, torch.exp(diff), torch.zeros_like(diff))
                    b_A = b_A * se
                beta_c = beta[b, t0:t1, hh].float()       # [L]
                b_A = b_A * beta_c[:, None]
                ii = torch.arange(L, device=k.device)
                b_A = torch.where(ii[:, None] > ii[None, :], b_A, torch.zeros_like(b_A))
                A[b, t0:t1, hh, 0:L] = b_A
    return A


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "chunk_scaled_dot_kkt_fwd"), \
            "Missing entry chunk_scaled_dot_kkt_fwd"
        assert hasattr(mod, "chunk_scaled_dot_kkt_fwd_kernel"), \
            "Missing @triton.jit chunk_scaled_dot_kkt_fwd_kernel"
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
    atol = 1e-4 if dtype == torch.float32 else 3e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2
    details = []
    for i, (B, T, Hg, H, K, use_g) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            inp = make_test_data(B, T, Hg, H, K, use_g, "cuda", dtype)
            A_t = _retry_oom(lambda: mod.chunk_scaled_dot_kkt_fwd(
                k=inp["k"], beta=inp["beta"], g_cumsum=inp["g"], cu_seqlens=None,
                chunk_size=BT))
            torch.cuda.synchronize()
            A_r = reference_kkt(inp)
            diff = (A_t.float() - A_r.float()).abs().max().item()
            isclose = torch.isclose(A_t.float(), A_r.float(), atol=atol, rtol=rtol)
            err_ratio = (~isclose).float().mean().item()
            passed = err_ratio <= 0.02
            details.append({"shape_id": i + 1, "shape": [B, T, Hg, H, K], "use_g": use_g,
                            "max_diff": diff, "err_ratio": err_ratio, "passed": bool(passed)})
            if not passed:
                return False, (f"Shape {i+1} {TEST_SHAPES[i]}: max_diff={diff:.4e} "
                               f"err_ratio={err_ratio:.4f}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": [B, T, Hg, H, K], "error": str(e)})
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
    for ti, (B, T, Hg, H, K, use_g) in enumerate(TEST_SHAPES):
        params = {"B": B, "T": T, "Hg": Hg, "H": H, "K": K, "use_g": use_g}
        try:
            torch.manual_seed(42 + ti)
            inp = make_test_data(B, T, Hg, H, K, use_g, "cuda", dtype)

            def fn():
                mod.chunk_scaled_dot_kkt_fwd(
                    k=inp["k"], beta=inp["beta"], g_cumsum=inp["g"], cu_seqlens=None,
                    chunk_size=BT)

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
                print(f"  shape {d['shape_id']} {d['shape']} use_g={d['use_g']}: "
                      f"max_diff={d['max_diff']:.4e} err_ratio={d['err_ratio']:.4f} "
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
