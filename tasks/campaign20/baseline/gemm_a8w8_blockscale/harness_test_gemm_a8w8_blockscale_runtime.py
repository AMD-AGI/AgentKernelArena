#!/usr/bin/env python3
"""GEAK harness for the MiniMax-M2.5 a8w8 block-scale ASM GEMM (bpreshuffle).

Kernel under test: aiter `_gemm_a8w8_blockscale_kernel_..._bpreshuffle` (ASM), source
`aiter/csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`, invoked via
`aiter.gemm_a8w8_blockscale_bpreshuffle_asm`. This is the dense FP8 (e4m3fnuz) +
128x128 block-scale GEMM used by the MiniMax-M2.5 attention projections:
  - qkv_proj : (M, 3072) x (4096, 3072)^T -> (M, 4096)   [q24+k4+v4 heads * 128, TP=2]
  - o_proj   : (M, 3072) x (3072, 3072)^T -> (M, 3072)
M is the number of tokens in the forward (prefill: ~2048..17918; decode is <0.5% so
excluded). Correctness is checked against a dequant + F.linear torch reference.
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange

REPO_ROOT = os.environ.get("GEAK_WORK_DIR", os.environ.get("GEAK_REPO_ROOT", "/sgl-workspace/aiter"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402

WARMUP = int(os.environ.get("GEAK_WARMUP", "20"))
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "100"))
BLOCK_SHAPE = (128, 128)
# The a8w8 block-scale ASM (bpreshuffle) kernel emits bf16; production casts to the
# fp16 model dtype afterwards (the `direct_copy` Half elementwise kernel in the profile).
DTYPE = torch.bfloat16

# (M tokens, N out, K in, label). K=3072=hidden. Real prefill token counts ~2048/11188/17918.
ALL_CONFIGS = [
    (2048, 4096, 3072, "qkv_proj"),
    (2048, 3072, 3072, "o_proj"),
    (11264, 4096, 3072, "qkv_proj"),
    (11264, 3072, 3072, "o_proj"),
    (17920, 4096, 3072, "qkv_proj"),
    (17920, 3072, 3072, "o_proj"),
    (4096, 4096, 3072, "qkv_proj"),
    (8192, 3072, 3072, "o_proj"),
]
CORRECTNESS_IDX = {0, 1, 6, 7}  # smaller M (torch fp32 ref is costly at large M)


def setup_inputs(cfg):
    m, n, k, _ = cfg
    torch.manual_seed(42)
    bn, bk = BLOCK_SHAPE
    scale_k = (k + bk - 1) // bk
    scale_n = (n + bn - 1) // bn
    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")
    # ASM bpreshuffle layout: weight shuffled (16,16), x_scale transposed.
    weight_asm = shuffle_weight(weight, layout=(16, 16))
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    out = torch.empty((m, n), dtype=DTYPE, device="cuda")
    return {"x": x, "weight": weight, "weight_asm": weight_asm, "x_scale": x_scale,
            "x_scale_t": x_scale_t, "w_scale": w_scale, "out": out, "m": m, "n": n, "k": k}


def run_kernel(b):
    return aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
        b["x"], b["weight_asm"], b["out"], b["x_scale_t"], b["w_scale"]
    )


def run_ref(b):
    bn, bk = BLOCK_SHAPE
    m, k, n = b["m"], b["k"], b["n"]
    x = b["x"].to(dtypes.fp32).view(m, k // bk, bk) * b["x_scale"].unsqueeze(-1)
    x = x.view(m, k)
    scale_n = (n + bn - 1) // bn
    scale_k = (k + bk - 1) // bk
    ws = rearrange(
        b["w_scale"].view(-1, 1).repeat(1, bn * bk).view(scale_n, scale_k, bn, bk),
        "nb kb bn bk -> (nb bn) (kb bk)",
    )[:n, :k]
    weight = b["weight"].to(dtypes.fp32) * ws
    return F.linear(x, weight).to(DTYPE)


def config_str(cfg):
    m, n, k, label = cfg
    return f"{label} M={m} N={n} K={k} fp8->bf16"


def check_correctness_val(out_ref, out_kernel):
    out_ref = out_ref.float()
    out_kernel = out_kernel.float()
    is_close = torch.isclose(out_ref, out_kernel, rtol=5e-2, atol=5e-2)
    err_ratio = 0.0 if is_close.all() else (~is_close).sum().item() / out_ref.numel()
    x, y = out_ref.double(), out_kernel.double()
    denom = (x * x + y * y).sum().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max(denom, 1e-12)
    return (err_ratio <= 0.20 and cos_diff < 0.02), err_ratio, cos_diff


def benchmark_kernel(b):
    def fn():
        run_kernel(b)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    lat = []
    for _ in range(ITERATIONS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        lat.append(s.elapsed_time(e))
    lat.sort()
    return lat[len(lat) // 2]


def mode_correctness(indices):
    print(f"Running correctness check on {len(indices)} configs...")
    all_pass = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            b = setup_inputs(cfg)
            out = run_kernel(b).clone()
            ref = run_ref(b)
            passed, err_ratio, cos_diff = check_correctness_val(ref, out)
            print(f"  [{idx}] {label}  err_ratio={err_ratio:.4f} cos_diff={cos_diff:.2e}  {'PASS' if passed else 'FAIL'}")
            all_pass = all_pass and passed
        except Exception as exc:
            import traceback
            print(f"  [{idx}] {label}  ERROR: {exc}")
            traceback.print_exc()
            all_pass = False
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={indices}")
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


def mode_benchmark(indices):
    print(f"Running benchmark on {len(indices)} configs...")
    lat = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            b = setup_inputs(cfg)
            ms = benchmark_kernel(b)
            print(f"  {label}  {ms:.4f}ms")
            lat.append(ms)
        except Exception as exc:
            print(f"  {label}  ERROR: {exc}")
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={indices}")
    if lat:
        geo = math.exp(sum(math.log(x) for x in lat) / len(lat))
        print(f"GEAK_RESULT_LATENCY_MS={geo:.4f}")
    else:
        print("No successful benchmarks")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="GEAK MiniMax a8w8 block-scale GEMM harness")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--correctness", action="store_true")
    g.add_argument("--benchmark", action="store_true")
    g.add_argument("--full-benchmark", action="store_true")
    g.add_argument("--profile", action="store_true")
    args = p.parse_args()
    print(f"Total configs: {len(ALL_CONFIGS)}")
    corr = sorted(CORRECTNESS_IDX)
    if args.correctness:
        mode_correctness(corr)
    elif args.benchmark or args.full_benchmark:
        mode_benchmark(list(range(len(ALL_CONFIGS))))
    elif args.profile:
        mode_benchmark(corr[:2])


if __name__ == "__main__":
    main()
