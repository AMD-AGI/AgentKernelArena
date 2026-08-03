#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/gdn_chunk_fwd_o.

Standalone harness for the GDN chunked linear-attention OUTPUT Triton kernel
(chunk_fwd_kernel_o). Exercises the regular (non-varlen) path: the per-chunk
compute (q@h^T inter-chunk, q@k^T intra-chunk, decay gating, causal tril,
b_A@v) is identical to the varlen branch; only the offset arithmetic differs.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton chunk_fwd_o vs torch fp32 reference, assert close.
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

TASK_NAME = "triton2flydsl/sglang/gdn_chunk_fwd_o"
SOURCE_FILE = os.path.join(TASK_DIR, "gdn_chunk_fwd_o.py")
BT = 64  # CHUNK_SIZE

# Test configs: (B, T, Hg, H, K, V). real Qwen3.5-35B GDN prefill: Hg=8, H=16,
# K=V=128 (TP=2); Hg=16,H=32 (TP=1). T multiple of 64.
TEST_SHAPES = [
    (1, 1024, 8, 16, 128, 128),    # real TP=2
    (2, 512, 8, 16, 128, 128),
    (1, 2048, 8, 16, 128, 128),
    (1, 1024, 16, 32, 128, 128),   # TP=1
    (4, 256, 8, 16, 128, 128),
    (1, 1024, 16, 16, 128, 128),   # Hg==H (qwen3-next style)
    (1, 4096, 8, 16, 128, 128),    # long prefill
    (1, 1024, 8, 16, 128, 64),     # V != K
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5
DTYPE_NAME = os.environ.get("GDN_DTYPE", "bfloat16")


def load_module():
    spec = importlib.util.spec_from_file_location("gdn_chunk_fwd_o_src", SOURCE_FILE)
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
    """Cumsum of g within each BT-sized chunk along T. g: [B, T, H]."""
    import torch
    B, T, H = g.shape
    out = g.clone()
    for t0 in range(0, T, BT):
        t1 = min(t0 + BT, T)
        out[:, t0:t1] = torch.cumsum(g[:, t0:t1], dim=1)
    return out


def make_test_data(B, T, Hg, H, K, V, device="cuda", dtype=None):
    import torch
    if dtype is None:
        dtype = getattr(torch, DTYPE_NAME)
    NT = (T + BT - 1) // BT
    q = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    h = torch.randn(B, NT, H, V, K, device=device, dtype=dtype) * 0.1
    # g (gating, log-space) is fp32 in the real GDN pipeline (cumsum output).
    g_raw = torch.nn.functional.logsigmoid(
        torch.randn(B, T, H, device=device, dtype=torch.float32)
    )
    g = _chunk_local_cumsum(g_raw, BT)
    return {"B": B, "T": T, "Hg": Hg, "H": H, "K": K, "V": V, "NT": NT,
            "q": q, "k": k, "v": v, "h": h, "g": g, "scale": K ** -0.5}


def reference_o(inp):
    """Pure-torch fp32 golden mirroring chunk_fwd_kernel_o (non-varlen)."""
    import torch
    B, T, Hg, H, K, V = inp["B"], inp["T"], inp["Hg"], inp["H"], inp["K"], inp["V"]
    NT, scale = inp["NT"], inp["scale"]
    dtype = inp["v"].dtype
    q, k, v, h, g = inp["q"].float(), inp["k"].float(), inp["v"].float(), inp["h"].float(), inp["g"].float()
    o = torch.zeros(B, T, H, V, device=inp["v"].device, dtype=torch.float32)
    rep = H // Hg
    for b in range(B):
        for hh in range(H):
            hg = hh // rep
            for it in range(NT):
                t0 = it * BT
                t1 = min(t0 + BT, T)
                q_c = q[b, t0:t1, hg]            # [L,K]
                k_c = k[b, t0:t1, hg]            # [L,K]
                v_c = v[b, t0:t1, hh]            # [L,V]
                h_c = h[b, it, hh]               # [V,K]
                g_c = g[b, t0:t1, hh]            # [L]
                b_o = q_c @ h_c.T               # [L,V]
                b_A = q_c @ k_c.T              # [L,L]
                b_o = b_o * torch.exp(g_c)[:, None]
                diff = g_c[:, None] - g_c[None, :]
                se = torch.where(diff <= 0, torch.exp(diff), torch.zeros_like(diff))
                b_A = b_A * se
                L = t1 - t0
                ii = torch.arange(L, device=q.device)
                b_A = torch.where(ii[:, None] >= ii[None, :], b_A, torch.zeros_like(b_A))
                # intra term uses bf16 b_A and bf16 v (kernel casts to b_v.dtype)
                b_A_bf = b_A.to(dtype).float()
                v_bf = v_c.to(dtype).float()
                b_o = b_o * scale + (b_A_bf @ v_bf) * scale
                o[b, t0:t1, hh] = b_o
    return o.to(dtype)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "chunk_fwd_o"), "Missing entry chunk_fwd_o"
        assert hasattr(mod, "chunk_fwd_kernel_o"), "Missing @triton.jit chunk_fwd_kernel_o"
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
            o_t = _retry_oom(lambda: mod.chunk_fwd_o(
                q=inp["q"], k=inp["k"], v=inp["v"], h=inp["h"], g=inp["g"],
                scale=inp["scale"], cu_seqlens=None))
            torch.cuda.synchronize()
            o_r = reference_o(inp)
            diff = (o_t.float() - o_r.float()).abs().max().item()
            # robust closeness: allow small fraction of mismatched elems (bf16 matmul)
            isclose = torch.isclose(o_t.float(), o_r.float(), atol=atol, rtol=rtol)
            err_ratio = (~isclose).float().mean().item()
            passed = err_ratio <= 0.02
            details.append({"shape_id": i + 1, "shape": [B, T, Hg, H, K, V],
                            "max_diff": diff, "err_ratio": err_ratio, "passed": bool(passed)})
            if not passed:
                return False, (f"Shape {i+1} {TEST_SHAPES[i]}: max_diff={diff:.4e} "
                               f"err_ratio={err_ratio:.4f}"), details
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
                mod.chunk_fwd_o(
                    q=inp["q"], k=inp["k"], v=inp["v"], h=inp["h"], g=inp["g"],
                    scale=inp["scale"], cu_seqlens=None)

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
