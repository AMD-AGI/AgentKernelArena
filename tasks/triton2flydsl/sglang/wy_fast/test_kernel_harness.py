#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/wy_fast.

Standalone harness for the GDN WY-representation recompute Triton kernel
(recompute_w_u_fwd_kernel). Exercises the regular (non-varlen) path; the
per-chunk math is identical to the varlen branch, only offsets differ.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton recompute_w_u_fwd vs torch fp32 reference, assert close.
  --full-benchmark : graph-first GPU timing, write build/performance_report.json

Reference per chunk/head (A = solved (I+tril(beta*KK^T))^{-1}, bf16):
  u = A @ (v * beta[:,None])
  w = A @ (k * beta[:,None] * exp(g)[:,None])
operands cast to bf16 before the fp32-accumulated matmuls.
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

TASK_NAME = "triton2flydsl/sglang/wy_fast"
SOURCE_FILE = os.path.join(TASK_DIR, "wy_fast.py")
BT = 64  # CHUNK_SIZE = A.shape[-1]

# Test configs: (B, T, Hg, H, K, V). real Qwen3-Next / Kimi-Linear GDN:
# K=V=128, Hg/H = 16/32 (grouped when Hg<H). T multiple/non-multiple of 64.
TEST_SHAPES = [
    (1, 1024, 16, 16, 128, 128),
    (1, 1024, 8, 16, 128, 128),    # grouped (Hg < H)
    (2, 512, 16, 32, 128, 128),
    (1, 2048, 32, 32, 128, 128),
    (1, 1000, 16, 16, 128, 128),   # T % BT != 0
    (4, 256, 16, 16, 128, 128),
    (1, 1024, 16, 16, 128, 64),    # V != K
    (1, 1024, 16, 16, 64, 128),    # K != V
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5
DTYPE_NAME = os.environ.get("GDN_DTYPE", "bfloat16")


def load_module():
    spec = importlib.util.spec_from_file_location("wy_fast_src", SOURCE_FILE)
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


def make_test_data(B, T, Hg, H, K, V, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = getattr(torch, DTYPE_NAME)
    NT = (T + BT - 1) // BT
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    g_raw = torch.nn.functional.logsigmoid(
        torch.randn(B, T, H, device=device, dtype=torch.float32)
    )
    g = _chunk_local_cumsum(g_raw, BT)
    # A = solved (I + tril(beta*KK^T))^{-1}: per chunk a unit-diagonal lower-tri
    # [L, L] block stored at A[:, t0:t1, :, 0:L] (bf16, as produced upstream).
    A = torch.zeros(B, T, H, BT, device=device, dtype=dtype)
    for t0 in range(0, T, BT):
        t1 = min(t0 + BT, T)
        L = t1 - t0
        block = 0.1 * torch.randn(B, L, H, L, device=device, dtype=torch.float32)
        ii = torch.arange(L, device=device)
        strict_lower = (ii[:, None] > ii[None, :]).float()  # [L,L]
        block = block * strict_lower[None, :, None, :]
        eye = torch.eye(L, device=device)
        block = block + eye[None, :, None, :]
        A[:, t0:t1, :, 0:L] = block.to(dtype)
    return {"B": B, "T": T, "Hg": Hg, "H": H, "K": K, "V": V, "NT": NT,
            "k": k, "v": v, "beta": beta, "g": g, "A": A}


def reference_wu(inp):
    import torch
    B, T, Hg, H, K, V = inp["B"], inp["T"], inp["Hg"], inp["H"], inp["K"], inp["V"]
    k, v, beta, g, A = inp["k"], inp["v"], inp["beta"], inp["g"], inp["A"]
    dtype = k.dtype
    rep = H // Hg
    NT = inp["NT"]
    w = torch.zeros(B, T, H, K, device=k.device, dtype=torch.float32)
    u = torch.zeros(B, T, H, V, device=k.device, dtype=torch.float32)
    for b in range(B):
        for hh in range(H):
            hg = hh // rep
            for it in range(NT):
                t0 = it * BT
                t1 = min(t0 + BT, T)
                L = t1 - t0
                A_c = A[b, t0:t1, hh, 0:L]                       # [L,L] bf16
                beta_c = beta[b, t0:t1, hh].float()             # [L]
                g_exp = torch.exp(g[b, t0:t1, hh].float())      # [L]
                # u = A @ (v*beta)
                v_c = v[b, t0:t1, hh].float()                   # [L,V]
                vb = (v_c * beta_c[:, None]).to(dtype).float()  # bf16-rounded operand
                u[b, t0:t1, hh] = A_c.float() @ vb
                # w = A @ (k*beta*exp(g))
                k_c = k[b, t0:t1, hg].float()                   # [L,K]
                kb = (k_c * beta_c[:, None] * g_exp[:, None]).to(dtype).float()
                w[b, t0:t1, hh] = A_c.float() @ kb
    return w, u


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "recompute_w_u_fwd"), "Missing entry recompute_w_u_fwd"
        assert hasattr(mod, "recompute_w_u_fwd_kernel"), \
            "Missing @triton.jit recompute_w_u_fwd_kernel"
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
    for i, (B, T, Hg, H, K, V) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            inp = make_test_data(B, T, Hg, H, K, V, "cuda", dtype)
            w_t, u_t = _retry_oom(lambda: mod.recompute_w_u_fwd(
                k=inp["k"], v=inp["v"], beta=inp["beta"], g_cumsum=inp["g"],
                A=inp["A"], cu_seqlens=None))
            torch.cuda.synchronize()
            w_r, u_r = reference_wu(inp)
            w_diff = (w_t.float() - w_r.float()).abs().max().item()
            u_diff = (u_t.float() - u_r.float()).abs().max().item()
            w_close = torch.isclose(w_t.float(), w_r.float(), atol=atol, rtol=rtol)
            u_close = torch.isclose(u_t.float(), u_r.float(), atol=atol, rtol=rtol)
            w_err = (~w_close).float().mean().item()
            u_err = (~u_close).float().mean().item()
            passed = (w_err <= 0.02) and (u_err <= 0.02)
            details.append({"shape_id": i + 1, "shape": [B, T, Hg, H, K, V],
                            "w_max_diff": w_diff, "u_max_diff": u_diff,
                            "w_err_ratio": w_err, "u_err_ratio": u_err,
                            "passed": bool(passed)})
            if not passed:
                return False, (f"Shape {i+1} {TEST_SHAPES[i]}: w_diff={w_diff:.4e} "
                               f"(err {w_err:.4f}) u_diff={u_diff:.4e} (err {u_err:.4f})"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": [B, T, Hg, H, K, V], "error": str(e)})
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
    for ti, (B, T, Hg, H, K, V) in enumerate(TEST_SHAPES):
        params = {"B": B, "T": T, "Hg": Hg, "H": H, "K": K, "V": V}
        try:
            torch.manual_seed(42 + ti)
            inp = make_test_data(B, T, Hg, H, K, V, "cuda", dtype)

            def fn():
                mod.recompute_w_u_fwd(
                    k=inp["k"], v=inp["v"], beta=inp["beta"], g_cumsum=inp["g"],
                    A=inp["A"], cu_seqlens=None)

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
                print(f"  shape {d['shape_id']} {d['shape']}: w_diff={d['w_max_diff']:.4e} "
                      f"(err {d['w_err_ratio']:.4f}) u_diff={d['u_max_diff']:.4e} "
                      f"(err {d['u_err_ratio']:.4f}) -> {'PASS' if d['passed'] else 'FAIL'}")
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
