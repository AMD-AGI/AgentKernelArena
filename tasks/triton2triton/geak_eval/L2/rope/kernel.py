#!/usr/bin/env python3
"""
RoPE (Rotary Position Embedding) SBHD Forward Kernel

Based on aiter's RoPE implementation (aiter/ops/triton/_triton_kernels/rope/rope.py):
- Supports NEOX and GPTJ rotation styles
- Handles SBHD (sequence, batch, head, dim) layout
- Supports NOPE (No Position Embedding) dimensions via reuse_freqs_front_part/nope_first
- Supports cached cos/sin values for efficiency
"""

import torch
import triton
import triton.language as tl
from enum import IntEnum

# ============================================================================
# ROTATE STYLE ENUM (from aiter/ops/triton/rope/rope.py)
# ============================================================================


class RotateStyle(IntEnum):
    NEOX = 0
    GPTJ = 1


# ============================================================================
# TRITON KERNELS (from aiter/ops/triton/_triton_kernels/rope/rope.py)
# ============================================================================


@triton.jit
def _get_neox_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    IS_BWD: tl.constexpr = False,
):
    if IS_BWD:
        x_rotated = tl.where(x_rotated_mask, -x, x)
    else:
        x_rotated = tl.where(x_rotated_mask, x, -x)

    x_rotated = tl.reshape(x_rotated, (BLOCK_T, 2, BLOCK_D_HALF))
    x_rotated = tl.flip(x_rotated, 2)
    x_rotated = tl.reshape(
        x_rotated,
        (
            BLOCK_T,
            BLOCK_D,
        ),
    )
    x_rotated = tl.flip(x_rotated, 1)
    return x_rotated


@triton.jit
def _get_gptj_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    IS_BWD: tl.constexpr = False,
):
    if IS_BWD:
        x_rotated = tl.where(x_rotated_mask, -x, x)
    else:
        x_rotated = tl.where(x_rotated_mask, x, -x)

    x_rotated = tl.reshape(x_rotated, (BLOCK_T, BLOCK_D_HALF, 2))
    x_rotated = tl.flip(x_rotated, 2)
    x_rotated = tl.reshape(
        x_rotated,
        (
            BLOCK_T,
            BLOCK_D,
        ),
    )
    return x_rotated


@triton.jit
def _rope_kernel_sbhd_fwd(
    x_ptr,
    freqs_ptr,
    out_ptr,
    stride_x_s,
    stride_x_b,
    stride_x_h,
    stride_x_d,
    stride_freqs_s,
    stride_freqs_b,
    stride_freqs_h,
    stride_freqs_d,
    stride_out_s,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    S,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    s_mask = s_offs < S

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_freqs_offs = tl.where(
                (d_offs >= BLOCK_D_HALF) & (d_offs < BLOCK_D),
                d_offs - BLOCK_D_HALF,
                d_offs,
            ).to(d_offs.dtype)
            d_freqs_mask = d_freqs_offs < BLOCK_D
        else:
            d_freqs_offs = d_offs // 2
            d_freqs_mask = d_freqs_offs < BLOCK_D_HALF
    else:
        d_freqs_offs = d_offs
        d_freqs_mask = d_freqs_offs < BLOCK_D

    freqs_mask = s_mask[:, None] & d_freqs_mask[None, :]
    freqs_offs = (
        s_offs[:, None] * stride_freqs_s + d_freqs_offs[None, :] * stride_freqs_d
    )

    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)
    cos = tl.cos(freqs.to(tl.float32))
    sin = tl.sin(freqs.to(tl.float32))

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_offs = (
        b * stride_x_b
        + s_offs[:, None] * stride_x_s
        + h * stride_x_h
        + (d_offs + nope_offs)[None, :] * stride_x_d
    )
    x_mask = s_mask[:, None] & (d_offs < BLOCK_D)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
        x_rotated = _get_neox_rotated_x(
            x, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF
        )
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]
        x_rotated = _get_gptj_rotated_x(
            x, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF
        )

    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = (
        b * stride_out_b
        + s_offs[:, None] * stride_out_s
        + h * stride_out_h
        + (d_offs + nope_offs)[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask=x_mask)


# ============================================================================
# PYTHON WRAPPERS (from aiter/ops/triton/rope/rope.py)
# ============================================================================


def _rope_fwd(
    x: torch.Tensor,
    out: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    inplace: bool,
) -> torch.Tensor:
    s, b, h, d = x.shape

    if freqs.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif freqs.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    if have_nope:
        BLOCK_D = d // 2
        BLOCK_D_HALF = d // 4
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2

    BLOCK_S = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(s, BLOCK_S))

    _rope_kernel_sbhd_fwd[grid](
        x,
        freqs,
        out,
        *x.stride(),
        *freqs.stride(),
        *out.stride(),
        s,
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        INPLACE=inplace,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=(rotate_style == RotateStyle.NEOX),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return out


def rope_fwd(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
) -> torch.Tensor:
    s, b, h, d = x.shape
    out = torch.empty(
        (s, b, h, d), dtype=x.dtype, device=x.device, requires_grad=False
    )
    _rope_fwd(x, out, freqs, rotate_style, reuse_freqs_front_part, nope_first, False)
    return out


# ============================================================================
# REFERENCE IMPLEMENTATIONS
# ============================================================================


def _rotate_half_neox(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_gptj(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _ref_rope_sbhd_fwd(x_, freqs_, rotate_style, reuse_freqs_front_part, nope_first):
    """Pure PyTorch reference for RoPE SBHD forward, handling all modes."""
    x = x_.to(dtype=torch.float32)
    freqs = freqs_.to(dtype=torch.float32)
    rotate_half = (
        _rotate_half_neox if rotate_style == RotateStyle.NEOX else _rotate_half_gptj
    )
    rotate_dim = freqs.shape[-1] * (2 if reuse_freqs_front_part else 1)
    if nope_first:
        d = x.shape[-1]
        x, x_forward = x[..., d - rotate_dim :], x[..., : d - rotate_dim]
    else:
        x, x_forward = x[..., :rotate_dim], x[..., rotate_dim:]
    if reuse_freqs_front_part:
        if rotate_style == RotateStyle.NEOX:
            freqs = freqs.repeat([1] * (freqs.dim() - 1) + [2])
        elif rotate_style == RotateStyle.GPTJ:
            freqs = freqs.repeat_interleave(2, dim=-1)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return (
        torch.cat((x_forward, x_embed.to(dtype=x.dtype)), dim=-1).to(dtype=x_.dtype)
        if nope_first
        else torch.cat((x_embed.to(dtype=x.dtype), x_forward), dim=-1).to(
            dtype=x_.dtype
        )
    )


# ============================================================================
# ENTRY POINTS
# ============================================================================


def triton_op(
    x, freqs, is_neox=True, reuse_freqs_front_part=True, nope_first=False
):
    rotate_style = RotateStyle.NEOX if is_neox else RotateStyle.GPTJ
    return rope_fwd(x, freqs, rotate_style, reuse_freqs_front_part, nope_first)


def torch_op(
    x, freqs, is_neox=True, reuse_freqs_front_part=True, nope_first=False
):
    rotate_style = RotateStyle.NEOX if is_neox else RotateStyle.GPTJ
    return _ref_rope_sbhd_fwd(
        x, freqs, rotate_style, reuse_freqs_front_part, nope_first
    )


# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

import math

# (B, S, H, D, is_neox, reuse_freqs_front_part, nope_first)
EVAL_CONFIGS = [
    # (B, S, H, D, is_neox, reuse_front, nope_first)
    # All 27 shapes from test_harness_rope.py, NEOX style, reuse_front=True
    (1, 2, 1, 4, True, True, False),
    (1, 4, 1, 4, True, True, False),
    (1, 2, 8, 4, True, True, False),
    (1, 2, 1, 64, True, True, False),
    (1, 4, 1, 64, True, True, False),
    (2, 2, 1, 64, True, True, False),
    (2, 4, 1, 64, True, True, False),
    (1, 10, 1, 64, True, True, False),
    (1, 2, 8, 64, True, True, False),
    (1, 32, 1, 64, True, True, False),
    (1, 10, 8, 64, True, True, False),
    (1, 2, 32, 128, True, True, False),
    (2, 10, 8, 64, True, True, False),
    (1, 32, 8, 64, True, True, False),
    (2, 32, 8, 64, True, True, False),
    (1, 10, 32, 128, True, True, False),
    (15, 10, 8, 64, True, True, False),
    (1, 32, 32, 128, True, True, False),
    (32, 10, 8, 64, True, True, False),
    (15, 32, 8, 64, True, True, False),
    (2, 32, 32, 128, True, True, False),
    (57, 10, 8, 64, True, True, False),
    (32, 32, 8, 64, True, True, False),
    (57, 32, 8, 64, True, True, False),
    (15, 32, 32, 128, True, True, False),
    (32, 32, 32, 128, True, True, False),
    (57, 32, 32, 128, True, True, False),
]

PROFILE_CONFIGS = [
    (1, 2, 1, 4, True, True, False),
    (2, 2, 1, 64, True, True, False),
    (1, 10, 8, 64, True, True, False),
    (1, 10, 32, 128, True, True, False),
    (2, 32, 32, 128, True, True, False),
]

RTOL, ATOL = 1e-2, 1e-2


# ============================================================================
# TEST HARNESS
# ============================================================================


def get_inputs(
    B, S, H, D, is_neox, reuse_front, nope_first, dtype=torch.float16, device="cuda"
):
    x = torch.randn(S, B, H, D, dtype=dtype, device=device)
    freqs = torch.randn(S, 1, 1, D // 2, dtype=dtype, device=device)
    return x, freqs


def check_correctness(
    B, S, H, D, is_neox, reuse_front, nope_first
) -> dict:
    try:
        x, freqs = get_inputs(B, S, H, D, is_neox, reuse_front, nope_first)
        res = triton_op(x, freqs, is_neox, reuse_front, nope_first)
        ref = torch_op(x, freqs, is_neox, reuse_front, nope_first)
        correct = torch.allclose(res, ref, rtol=RTOL, atol=ATOL)
        max_diff = torch.max(torch.abs(res - ref)).item() if not correct else 0.0
        return {"correct": correct, "max_diff": max_diff, "error": None}
    except Exception as e:
        return {"correct": False, "max_diff": float("inf"), "error": str(e)}


def _config_label(B, S, H, D, is_neox, reuse_front, nope_first):
    style = "NEOX" if is_neox else "GPTJ"
    nope = ""
    if not reuse_front:
        nope = ",NOPE" + (",first" if nope_first else ",last")
    return f"({B},{S},{H},{D},{style}{nope})"


BASELINE_LATENCIES = {
    (1, 2, 1, 4, True, True, False): 0.021,
    (1, 4, 1, 4, True, True, False): 0.021,
    (1, 2, 8, 4, True, True, False): 0.0206,
    (1, 2, 1, 64, True, True, False): 0.0209,
    (1, 4, 1, 64, True, True, False): 0.021,
    (2, 2, 1, 64, True, True, False): 0.021,
    (2, 4, 1, 64, True, True, False): 0.021,
    (1, 10, 1, 64, True, True, False): 0.0211,
    (1, 2, 8, 64, True, True, False): 0.021,
    (1, 32, 1, 64, True, True, False): 0.0208,
    (1, 10, 8, 64, True, True, False): 0.0209,
    (1, 2, 32, 128, True, True, False): 0.0208,
    (2, 10, 8, 64, True, True, False): 0.0209,
    (1, 32, 8, 64, True, True, False): 0.0209,
    (2, 32, 8, 64, True, True, False): 0.0209,
    (1, 10, 32, 128, True, True, False): 0.0209,
    (15, 10, 8, 64, True, True, False): 0.021,
    (1, 32, 32, 128, True, True, False): 0.0209,
    (32, 10, 8, 64, True, True, False): 0.0211,
    (15, 32, 8, 64, True, True, False): 0.0213,
    (2, 32, 32, 128, True, True, False): 0.0228,
    (57, 10, 8, 64, True, True, False): 0.0212,
    (32, 32, 8, 64, True, True, False): 0.021,
    (57, 32, 8, 64, True, True, False): 0.022,
    (15, 32, 32, 128, True, True, False): 0.0214,
    (32, 32, 32, 128, True, True, False): 0.0212,
    (57, 32, 32, 128, True, True, False): 0.0212,
}


def benchmark_config(
    B, S, H, D, is_neox, reuse_front, nope_first, warmup=500, iters=2000
) -> dict:
    import time

    cfg_key = (B, S, H, D, is_neox, reuse_front, nope_first)
    x, freqs = get_inputs(B, S, H, D, is_neox, reuse_front, nope_first)

    for _ in range(warmup):
        triton_op(x, freqs, is_neox, reuse_front, nope_first)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        triton_op(x, freqs, is_neox, reuse_front, nope_first)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) * 1000 / iters

    baseline_ms = BASELINE_LATENCIES.get(cfg_key, triton_ms)
    return {
        "torch_ms": baseline_ms,
        "triton_ms": triton_ms,
        "speedup": baseline_ms / triton_ms if triton_ms > 0 else 1.0,
    }


def evaluate(
    configs=None, warmup: int = 500, iters: int = 2000, verbose: bool = True
) -> dict:
    configs = configs or EVAL_CONFIGS
    results, failures = [], []

    if verbose:
        print(
            f"{'Config':<35} {'Correct':>8} {'Torch':>10} {'Triton':>10} {'Speedup':>10}"
        )
        print("-" * 75)

    for cfg in configs:
        B, S, H, D, is_neox, reuse_front, nope_first = cfg
        label = _config_label(*cfg)
        corr = check_correctness(*cfg)
        if not corr["correct"]:
            failures.append({"config": cfg, **corr})
            if verbose:
                err = corr["error"] or f"max_diff={corr['max_diff']:.2e}"
                print(f"{label:<35} {'FAIL':>8}   {err[:25]}")
            continue

        bench = benchmark_config(*cfg, warmup=warmup, iters=iters)
        results.append({"config": cfg, "correct": True, **bench})
        if verbose:
            marker = " *" if bench["speedup"] > 1.0 else ""
            print(
                f"{label:<35} {'PASS':>8} {bench['torch_ms']:>8.4f}ms "
                f"{bench['triton_ms']:>8.4f}ms {bench['speedup']:>8.2f}x{marker}"
            )

    total_baseline = sum(r["torch_ms"] for r in results)
    total_evolved = sum(r["triton_ms"] for r in results)
    speedup = total_baseline / total_evolved if total_evolved > 0 else 0.0

    if verbose:
        print("-" * 75)
        status = (
            "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        )
        print(f"{'Status:':<35} {status}")
        if results:
            print(f"{'Speedup (total):':<35} {speedup:.2f}x")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
        "speedup_geomean": speedup,
    }


def run_profile(configs=None, warmup: int = 3, iters: int = 1, verbose: bool = True):
    configs = configs or PROFILE_CONFIGS
    if verbose:
        print(
            f"Profile: {len(configs)} config(s), {warmup} warmup, {iters} iter(s)"
        )

    for cfg in configs:
        B, S, H, D, is_neox, reuse_front, nope_first = cfg
        x, freqs = get_inputs(B, S, H, D, is_neox, reuse_front, nope_first)
        for _ in range(warmup):
            triton_op(x, freqs, is_neox, reuse_front, nope_first)
        torch.cuda.synchronize()
        for _ in range(iters):
            triton_op(x, freqs, is_neox, reuse_front, nope_first)
        torch.cuda.synchronize()
        if verbose:
            print(f"  {_config_label(*cfg)} done")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoPE SBHD Forward Kernel Test Harness")
    parser.add_argument(
        "--profile", action="store_true", help="Run minimal profiling workload"
    )
    args = parser.parse_args()

    print("=" * 75)
    print("RoPE SBHD Forward Kernel")
    print("=" * 75)

    if args.profile:
        print("\n[Profile Mode]")
        run_profile()
    else:
        print("\n[Evaluation]")
        evaluate()

    print("=" * 75)
