#!/usr/bin/env python3
"""GEAK harness for the MiniMax-M2.5 CK block-scale MoE **stage-1** GEMM (gate/up projection).

Kernel under test: aiter CK `kernel_moe_gemm_2lds` / GridwiseMoeGemmBlockScale with
InMemoryDataOp=Set (stage1), invoked via the production callable returned by
aiter.fused_moe.get_2stage_cfgs (identical to run_sglang_test_minimax.sh).

Stage1 computes, per routed token: a1(fp8) @ w1(fp8 gate+up) -> silu(gate)*up -> a2 (fp16),
shapes a1(token,3072) x w1(256,1536,3072) -> a2(token,8,768). Correctness vs torch_moe_stage1.
"""
import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from minimax_moe_common import (  # noqa: E402
    build_base, quant_a1, torch_moe_stage1,
    INTER_DIM, TOPK, DTYPE, QUANT_TYPE, ACT, ALL_TOKENS,
)

WARMUP = int(os.environ.get("GEAK_WARMUP", "10"))
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "100"))

# (token, dtype). dtype is fixed fp16 (MiniMax compute dtype); token sweeps the real range.
ALL_CONFIGS = [(t, DTYPE) for t in ALL_TOKENS]
# Correctness validates an IN-REGIME shape so it agrees with the scored perf cases
# (c2/c32/c64 = 2048/32768/65536). We use the smallest regime token (2048 = c2): the
# fp32 256-expert torch reference is costly at large M, and 2048 already exercises the
# block_m=64 prefill path the patch targets. (Was {16,256,2048} -- 16/256 are NOT in the
# arena regime, which is what made correctness and perf run different test cases.)
# Override via GEAK_CORRECTNESS_TOKENS="t1,t2,..." (stage2-style) if a wider check is wanted.
_env_corr = os.environ.get("GEAK_CORRECTNESS_TOKENS", "").strip()
if _env_corr:
    CORRECTNESS_TOKENS = {int(x) for x in _env_corr.split(",") if x.strip()}
else:
    CORRECTNESS_TOKENS = {2048}


def setup_inputs(cfg):
    token, _ = cfg
    base = build_base(token, seed=42)
    a1, a1_scale = quant_a1(base["inp"])
    a2 = torch.empty((token, TOPK, INTER_DIM), dtype=DTYPE, device=base["dev"])
    base.update({"a1": a1, "a1_scale": a1_scale, "a2": a2})
    return base


def run_kernel(b):
    # Identical to fused_moe_2stages' stage1 invocation.
    return b["metadata"].stage1(
        b["a1"], b["w1q"], b["w2q"],
        b["sorted_ids"], b["sorted_expert_ids"], b["num_valid_ids"],
        b["a2"], TOPK,
        block_m=b["block_m"],
        a1_scale=b["a1_scale"],
        w1_scale=b["w1_scale"],
        sorted_weights=None,
    )


def run_ref(b):
    return torch_moe_stage1(
        b["a1"], b["w1_qt"], b["w2_qt"], b["topk_w"], b["topk_ids"],
        dtype=DTYPE, activation=ACT, quant_type=QUANT_TYPE,
        a1_scale=b["a1_scale"], w1_scale=b["w1_scale"], doweight=False,
    )


def config_str(cfg):
    token, dtype = cfg
    return f"token={token} model_dim=3072 inter_dim=768 E=256 topk=8 {dtype}"


def check_correctness_val(out_ref, out_kernel):
    # fp8 block-scale GEMM: use err-ratio + cosine like the reference harness.
    out_ref = out_ref.float()
    out_kernel = out_kernel.float()
    rtol, atol, max_err_ratio = 5e-2, 5e-2, 0.20
    is_close = torch.isclose(out_ref, out_kernel, rtol=rtol, atol=atol)
    err_ratio = 0.0 if is_close.all() else (~is_close).sum().item() / out_ref.numel()
    x, y = out_ref.double(), out_kernel.double()
    denom = (x * x + y * y).sum().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max(denom, 1e-12)
    return (err_ratio <= max_err_ratio and cos_diff < 0.02), err_ratio, cos_diff


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
    return sum(lat) / len(lat)


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
    p = argparse.ArgumentParser(description="GEAK MiniMax MoE stage1 (gate/up) harness")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--correctness", action="store_true")
    g.add_argument("--benchmark", action="store_true")
    g.add_argument("--full-benchmark", action="store_true")
    g.add_argument("--profile", action="store_true")
    args = p.parse_args()
    print(f"Total configs: {len(ALL_CONFIGS)}")
    corr = [i for i, c in enumerate(ALL_CONFIGS) if c[0] in CORRECTNESS_TOKENS]
    if args.correctness:
        mode_correctness(corr)
    elif args.benchmark:
        mode_benchmark(list(range(len(ALL_CONFIGS))))
    elif args.full_benchmark:
        mode_benchmark(list(range(len(ALL_CONFIGS))))
    elif args.profile:
        mode_benchmark(corr[:2])


if __name__ == "__main__":
    main()
