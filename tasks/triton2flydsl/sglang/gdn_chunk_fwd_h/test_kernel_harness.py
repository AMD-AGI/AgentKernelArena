#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/gdn_chunk_fwd_h.

Standalone harness for the GDN chunked recurrent STATE Triton kernel
(chunk_gated_delta_rule_fwd_kernel_h_blockdim64). Exercises the regular
(non-varlen) path: the per-chunk recurrence (delta correction, decay gating,
rank-BT state update, in-place final-state write) is identical to the varlen
branch; only the chunk-offset arithmetic differs.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton fwd_h vs torch fp32 reference (h, v_new, final state).
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

TASK_NAME = "triton2flydsl/sglang/gdn_chunk_fwd_h"
SOURCE_FILE = os.path.join(TASK_DIR, "gdn_chunk_fwd_h.py")
BT = 64  # CHUNK_SIZE

# Test configs: (B, T, Hg, H, K, V, pool). real Qwen3.5-35B GDN prefill:
# Hg=8, H=16, K=V=128 (TP=2); Hg=16,H=32 (TP=1). T multiple of 64.
TEST_SHAPES = [
    (1, 1024, 8, 16, 128, 128, 32),    # real TP=2
    (2, 512, 8, 16, 128, 128, 32),
    (1, 2048, 8, 16, 128, 128, 32),
    (1, 1024, 16, 32, 128, 128, 32),   # TP=1
    (4, 256, 8, 16, 128, 128, 64),
    (1, 1024, 16, 16, 128, 128, 32),   # Hg==H
    (1, 2048, 16, 32, 128, 128, 32),   # long prefill (TP=1)
    (1, 1024, 8, 16, 64, 128, 32),     # K=64 (single K-block)
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5
DTYPE_NAME = os.environ.get("GDN_DTYPE", "bfloat16")


def load_module():
    spec = importlib.util.spec_from_file_location("gdn_chunk_fwd_h_src", SOURCE_FILE)
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


def _chunk_local_cumsum(g, BT):
    import torch
    B, T, H = g.shape
    out = g.clone()
    for t0 in range(0, T, BT):
        t1 = min(t0 + BT, T)
        out[:, t0:t1] = torch.cumsum(g[:, t0:t1], dim=1)
    return out


def make_test_data(B, T, Hg, H, K, V, pool, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = getattr(torch, DTYPE_NAME)
    NT = (T + BT - 1) // BT
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    w = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    u = torch.randn(B, T, H, V, device=device, dtype=dtype)
    # g (gating, log-space) is fp32; scaled so exp(cumsum) stays non-degenerate.
    g_raw = torch.nn.functional.logsigmoid(
        torch.randn(B, T, H, device=device, dtype=torch.float32)
    ) * 0.1
    g = _chunk_local_cumsum(g_raw, BT)
    init = torch.randn(pool, H, V, K, device=device, dtype=dtype) * 0.1
    idx = torch.arange(B, device=device, dtype=torch.int32)
    return {"B": B, "T": T, "Hg": Hg, "H": H, "K": K, "V": V, "NT": NT, "pool": pool,
            "k": k, "w": w, "u": u, "g": g, "init": init, "idx": idx}


def reference_h(inp):
    """Pure-torch fp32 golden mirroring the blockdim64 fwd_h kernel (non-varlen).

    Returns (h[B,NT,H,V,K], v_new[B,T,H,V], updated_init[pool,H,V,K]) in dtype.
    """
    import torch
    B, T, Hg, H, K, V = inp["B"], inp["T"], inp["Hg"], inp["H"], inp["K"], inp["V"]
    NT = inp["NT"]
    dtype = inp["k"].dtype
    k, w, u = inp["k"].float(), inp["w"].float(), inp["u"].float()
    g = inp["g"].float()
    init = inp["init"].clone()
    idx = inp["idx"]

    h_out = torch.zeros(B, NT, H, V, K, device=k.device, dtype=torch.float32)
    v_new = torch.zeros(B, T, H, V, device=k.device, dtype=torch.float32)
    init_f = init.float()
    rep = H // Hg
    for n in range(B):
        index = int(idx[n].item())
        for hh in range(H):
            hg = hh // rep
            state = init_f[index, hh].clone()             # [V, K] fp32
            for it in range(NT):
                t0 = it * BT
                t1 = min(t0 + BT, T)
                h_out[n, it, hh] = state                  # start-of-chunk state
                w_c = w[n, t0:t1, hh]                      # [L, K]
                u_c = u[n, t0:t1, hh]                      # [L, V]
                # inter term casts state to bf16 (matches kernel b_h.to(b_w.dtype))
                state_bf = state.to(dtype).float()
                inter = w_c.to(dtype).float() @ state_bf.T  # [L, V]
                bv = u_c - inter                           # [L, V]
                v_new[n, t0:t1, hh] = bv                   # BEFORE gating
                last = min((it + 1) * BT, T) - 1
                g_last = g[n, last, hh]
                g_blk = g[n, t0:t1, hh]                     # [L]
                diff = g_last - g_blk
                se = torch.where(diff <= 0, torch.exp(diff), torch.zeros_like(diff))
                bv = bv * se[:, None]
                state = state * torch.exp(g_last)
                bv_bf = bv.to(dtype).float()
                k_c = k[n, t0:t1, hg].to(dtype).float()    # [L, K]
                state = state + bv_bf.T @ k_c              # [V, K]
            init_f[index, hh] = state
    return h_out.to(dtype), v_new.to(dtype), init_f.to(dtype)


def _run_triton(mod, inp):
    import torch
    init = inp["init"].clone()
    h, v_new = _retry_oom(lambda: mod.chunk_gated_delta_rule_fwd_h(
        k=inp["k"], w=inp["w"], u=inp["u"], g=inp["g"],
        initial_state=init, initial_state_indices=inp["idx"],
        save_new_value=True, cu_seqlens=None))
    torch.cuda.synchronize()
    return h, v_new, init


def _close(a, b, atol, rtol, max_ratio):
    import torch
    isclose = torch.isclose(a.float(), b.float(), atol=atol, rtol=rtol)
    er = (~isclose).float().mean().item()
    md = (a.float() - b.float()).abs().max().item()
    return er <= max_ratio, er, md


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "chunk_gated_delta_rule_fwd_h"), "Missing entry chunk_gated_delta_rule_fwd_h"
        assert hasattr(mod, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"), "Missing @triton.jit kernel"
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
    # Deep bf16 recurrence: state accumulation drift grows with #chunks, so use
    # bf16-appropriate tolerance (the GDN reference test also treats long-sequence
    # state divergence as expected/informational).
    atol = 1e-4 if dtype == torch.float32 else 5e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2
    max_ratio = 0.0 if dtype == torch.float32 else 0.05
    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        B, T, Hg, H, K, V, pool = cfg
        try:
            torch.manual_seed(42 + i)
            inp = make_test_data(B, T, Hg, H, K, V, pool, "cuda", dtype)
            h_t, vn_t, init_t = _run_triton(mod, inp)
            h_r, vn_r, init_r = reference_h(inp)
            idx = inp["idx"]
            h_ok, h_er, h_md = _close(h_t, h_r, atol, rtol, max_ratio)
            v_ok, v_er, v_md = _close(vn_t, vn_r, atol, rtol, max_ratio)
            s_ok, s_er, s_md = _close(init_t[idx], init_r[idx], atol, rtol, max_ratio)
            passed = bool(h_ok and v_ok and s_ok)
            details.append({"shape_id": i + 1, "shape": list(cfg),
                            "h_err": h_er, "vnew_err": v_er, "state_err": s_er,
                            "h_md": h_md, "vnew_md": v_md, "state_md": s_md,
                            "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} {cfg}: h_err={h_er:.4f} vnew_err={v_er:.4f} "
                               f"state_err={s_er:.4f}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": list(cfg), "error": str(e)})
            return False, f"Shape {i+1} {cfg}: exception: {e}", details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    dtype = getattr(torch, DTYPE_NAME)
    test_cases = []
    for ti, cfg in enumerate(TEST_SHAPES):
        B, T, Hg, H, K, V, pool = cfg
        params = {"B": B, "T": T, "Hg": Hg, "H": H, "K": K, "V": V, "pool": pool}
        try:
            torch.manual_seed(42 + ti)
            inp = make_test_data(B, T, Hg, H, K, V, pool, "cuda", dtype)
            init = inp["init"]
            bench_init = init.clone()

            def prepare_fn():
                bench_init.copy_(init)

            def fn():
                mod.chunk_gated_delta_rule_fwd_h(
                    k=inp["k"], w=inp["w"], u=inp["u"], g=inp["g"],
                    initial_state=bench_init, initial_state_indices=inp["idx"],
                    save_new_value=True, cu_seqlens=None)

            prepare_fn()
            _retry_oom(fn)
            for _ in range(WARMUP_ITERATIONS):
                prepare_fn()
                fn()
            torch.cuda.synchronize()
            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                fn, warmup=0, repetition=BENCHMARK_ITERATIONS,
                prepare_fn=prepare_fn,
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
                print(f"  shape {d['shape_id']} {d['shape']}: h_err={d['h_err']:.4f} "
                      f"vnew_err={d['vnew_err']:.4f} state_err={d['state_err']:.4f} "
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
