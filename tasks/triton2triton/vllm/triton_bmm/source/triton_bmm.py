"""Batched matrix multiplication kernel using Triton, adapted from vLLM batch_invariant.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """Batched GEMM: (B, M, K) x (B, K, N) -> (B, M, N)"""
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)

    if pid_b >= B:
        return

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    if A_LARGE or B_LARGE or C_LARGE:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = tl.where(mask_n, offs_n, 0)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    a_batch_ptr = a_ptr + pid_b * stride_ab
    b_batch_ptr = b_ptr + pid_b * stride_bb
    c_batch_ptr = c_ptr + pid_b * stride_cb

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_mask = tl.arange(0, BLOCK_SIZE_K)

    for ki in range(k_tiles):
        if A_LARGE or B_LARGE:
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        else:
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_batch_ptr + (
            offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_batch_ptr + (
            offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )

        k_valid = offs_k_mask < (K - ki * BLOCK_SIZE_K)
        a_mask = mask_m[:, None] & k_valid[None, :]
        b_mask = k_valid[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    c_m = offs_m
    c_n = offs_n
    if C_LARGE:
        c_m = c_m.to(tl.int64)
        c_n = c_n.to(tl.int64)

    c_ptrs = c_batch_ptr + stride_cm * c_m[:, None] + stride_cn * c_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


def _launch_config(dtype_name, batch, m, n, k):
    """Return the bmm_kernel tile sizes, num_warps and num_stages.

    Tune these values, optionally per dtype or problem size. The protected
    wrapper validates them and derives the grid from the tile sizes.
    """
    if dtype_name == "float32":
        return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                "num_stages": 2, "num_warps": 4}
    return {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
            "num_stages": 2, "num_warps": 8}


_DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}
_TILE_KEYS = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")


def _checked_launch_config(cfg):
    """Accept only tile sizes, num_warps and num_stages within supported bounds."""
    expected = set(_TILE_KEYS) | {"num_warps", "num_stages"}
    if not isinstance(cfg, dict) or set(cfg) != expected:
        raise ValueError(f"Launch configuration must set exactly {sorted(expected)}")
    checked = {}
    for key in sorted(expected):
        value = cfg[key]
        if type(value) is not int:
            raise ValueError(f"Launch setting {key} must be an int")
        checked[key] = value
    for key in _TILE_KEYS:
        if not (16 <= checked[key] <= 256 and checked[key] & (checked[key] - 1) == 0):
            raise ValueError(f"{key} must be a power of two in [16, 256]")
    if checked["num_warps"] not in (1, 2, 4, 8, 16):
        raise ValueError("num_warps must be 1, 2, 4, 8 or 16")
    if not 1 <= checked["num_stages"] <= 4:
        raise ValueError("num_stages must be in [1, 4]")
    return checked


def bmm_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for batched matrix multiplication kernel."""
    assert a.ndim == 3 and b.ndim == 3, "Inputs must be 3D"
    assert a.shape[0] == b.shape[0], "Batch dimensions must match"
    assert a.shape[2] == b.shape[1], "Incompatible inner dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    B_size, M, K = a.shape
    _, _, N = b.shape
    dtype = a.dtype

    c = torch.empty((B_size, M, N), device=a.device, dtype=dtype)

    cfg = _checked_launch_config(_launch_config(_DTYPE_NAMES[dtype], B_size, M, N, K))
    grid = (
        B_size,
        triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
    )

    bmm_kernel[grid](
        a,
        b,
        c,
        B_size,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        **cfg,
    )

    return c
