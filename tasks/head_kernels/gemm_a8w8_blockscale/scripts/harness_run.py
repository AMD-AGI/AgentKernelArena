#!/usr/bin/env python3
"""Arena harness for triton2triton/gemm_a8w8_blockscale (Triton entry).

Optimization target (per config.yaml):
  source/gemm_a8w8_blockscale.py  ::  gemm_a8w8_blockscale
This is the MiniMax-M2.5 a8w8 128x128 block-scale FP8 GEMM (Triton path) used by
the attention projections.

Golden contract (INDEPENDENT torch reference):
  - The golden is a pure-PyTorch fp32 recomputation of the a8w8 128x128
    block-scale GEMM (`torch_ref_a8w8_blockscale`), NOT a copy of the kernel.
    We compare the EDITABLE live kernel output against the independent fp32
    oracle on identical seeded inputs (cosine >= 0.99, err_ratio <= 0.05,
    cos_diff < 0.01).

WORKLOAD REGIME (token-parallel GEMM): M = B * seqlen = B * 1024.
  concurrency B in {2,32,64}  ->  M in {2048, 32768, 65536}  -> ids c2,c32,c64.
Model dims kept from the captured base case (qkv_proj attention projection):
  N = 4096, K = 3072  (K=3072=hidden; block scale 128x128; fp8 e4m3fnuz -> bf16).
"""
import importlib.util
import math
import os
import sys

import torch

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(TASK_DIR, "source")
SOURCE_PY = os.path.join(SOURCE_DIR, "gemm_a8w8_blockscale.py")

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

BLOCK_SHAPE = (128, 128)          # 128x128 weight block scale
N_MODEL = 4096                    # qkv_proj output dim (TP=2)
K_MODEL = 3072                    # hidden dim, = 24 * 128
SEQLEN = 1024
DTYPE = torch.bfloat16

WARMUP = int(os.environ.get("GEAK_WARMUP", "10"))
ITERS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "100"))

# concurrency -> (id, M)
CASES = [(2, "c2"), (32, "c32"), (64, "c64")]


def _load_live():
    spec = importlib.util.spec_from_file_location("local_gemm_a8w8_blockscale", SOURCE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gemm_a8w8_blockscale


def build_inputs(M, N, K, seed=42):
    """Inputs for the non-shuffled Triton entry:
       x (M,K) fp8, weight (N,K) fp8, x_scale (M, scale_k) f32, w_scale (scale_n, scale_k) f32.
    """
    e4m3 = getattr(torch, "float8_e4m3fnuz")
    torch.manual_seed(seed)
    bn, bk = BLOCK_SHAPE
    scale_k = (K + bk - 1) // bk
    scale_n = (N + bn - 1) // bn
    x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3)
    weight = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(e4m3)
    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")
    return x, weight, x_scale, w_scale


def torch_ref_a8w8_blockscale(x, weight, x_scale, w_scale):
    """Independent fp32 oracle for Y = dequant(x) @ dequant(weight)^T.

    x       (M,K) fp8, per-1x128 activation scale x_scale (M, scale_k)
    weight  (N,K) fp8, 128x128 block scale w_scale (scale_n, scale_k)
    Dequant: x_deq[m,k]    = x[m,k]      * x_scale[m, k//bk]
             w_deq[n,k]    = weight[n,k] * w_scale[n//bn, k//bk]
    Returns fp32 (M,N). This shares NO code with the kernel under test, so an
    edit to the triton kernel that breaks the math cannot pass trivially.
    """
    bn, bk = BLOCK_SHAPE
    M, K = x.shape
    N, _ = weight.shape
    xf = x.to(torch.float32)
    wf = weight.to(torch.float32)
    xs = x_scale.to(torch.float32).repeat_interleave(bk, dim=1)[:, :K]
    ws = (
        w_scale.to(torch.float32)
        .repeat_interleave(bn, dim=0)
        .repeat_interleave(bk, dim=1)
    )[:N, :K]
    return (xf * xs) @ (wf * ws).transpose(0, 1)


def compare(out_ref, out_kernel):
    out_ref = out_ref.float()
    out_kernel = out_kernel.float()
    is_close = torch.isclose(out_ref, out_kernel, rtol=5e-2, atol=5e-2)
    err_ratio = 0.0 if is_close.all() else (~is_close).sum().item() / out_ref.numel()
    x, y = out_ref.double(), out_kernel.double()
    denom = (x * x + y * y).sum().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max(denom, 1e-12)
    cos = 1.0 - cos_diff
    passed = (err_ratio <= 0.05 and cos_diff < 0.01 and cos >= 0.99)
    return passed, err_ratio, cos_diff, cos


def run_correctness():
    live = _load_live()
    all_pass = True
    for B, cid in CASES:
        M = B * SEQLEN
        N, K = N_MODEL, K_MODEL
        try:
            x, w, xs, ws = build_inputs(M, N, K)
            out_live = live(x, w, xs, ws, DTYPE).clone()
            out_gold = torch_ref_a8w8_blockscale(x, w, xs, ws)
            passed, err_ratio, cos_diff, cos = compare(out_gold, out_live)
            print(f"  [{cid}] M={M} N={N} K={K}  err_ratio={err_ratio:.4f} "
                  f"cos={cos:.6f} cos_diff={cos_diff:.2e}  {'PASS' if passed else 'FAIL'}")
            all_pass = all_pass and passed
        except Exception as exc:
            import traceback
            print(f"  [{cid}] M={M} N={N} K={K}  ERROR: {exc}")
            traceback.print_exc()
            all_pass = False
        finally:
            torch.cuda.empty_cache()
    if not all_pass:
        print("CORRECTNESS FAILED")
        return False
    print("ALL CORRECTNESS CHECKS PASSED")
    return True


def _bench_one(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    lat = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        lat.append(s.elapsed_time(e))
    return sum(lat) / len(lat)


def run_benchmark():
    live = _load_live()
    results = []
    for B, cid in CASES:
        M = B * SEQLEN
        N, K = N_MODEL, K_MODEL
        try:
            x, w, xs, ws = build_inputs(M, N, K)
            y = torch.empty((M, N), dtype=DTYPE, device="cuda")

            def fn():
                live(x, w, xs, ws, DTYPE, y)

            ms = _bench_one(fn)
            print(f"  {cid} M={M} N={N} K={K} fp8->bf16  {ms:.4f}ms")
            results.append((cid, M, N, K, ms))
        except Exception as exc:
            import traceback
            print(f"  {cid} M={M} N={N} K={K}  ERROR: {exc}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
    if results:
        geo = math.exp(sum(math.log(r[4]) for r in results) / len(results))
        print(f"GEAK_RESULT_LATENCY_MS={geo:.4f}")
    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--correctness", action="store_true")
    g.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    if args.correctness:
        ok = run_correctness()
        sys.exit(0 if ok else 1)
    else:
        res = run_benchmark()
        sys.exit(0 if res else 1)


if __name__ == "__main__":
    main()
