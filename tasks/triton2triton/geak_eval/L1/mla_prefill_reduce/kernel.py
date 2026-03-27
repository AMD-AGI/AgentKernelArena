#!/usr/bin/env python3
"""
MLA Prefill Reduce Kernel — exact copy from aiter/mla.py L577-L780.

Triton kernel that merges partial attention tiles (log-sum-exp weighted
combination) into final output for Multi-head Latent Attention prefill.
"""

import math
import os
import sys

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Resolve aiter so we can reuse get_cu_num for the default MAX_PARTIALS
# ---------------------------------------------------------------------------
_repo_root = os.environ.get(
    "GEAK_REPO_ROOT",
    "/home/upandey/AIG-Eval/external_repos/aiter",
)
for _p in [_repo_root, os.path.join(_repo_root, "aiter")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from aiter.jit.utils.chip_info import get_cu_num
except ImportError:
    def get_cu_num():
        return 304


# ============================================================================
# TRITON KERNEL — verbatim from aiter/mla.py L577-L712
# ============================================================================


@triton.jit
def _mla_prefill_reduce_kernel(
    # Input tensors
    partial_output_ptr,  # [padded_num_tokens * available_tgs, num_head_q, v_head_dim]
    partial_lse_ptr,  # [padded_num_tokens * available_tgs, num_head_q]
    # Metadata tensors
    reduce_indptr_ptr,  # [num_reduce_groups + 1]
    reduce_final_map_ptr,  # [num_reduce_groups, 2]: [qo_start, qo_end]
    reduce_partial_map_ptr,  # [num_partial_tiles]: [partial_qo_loc]
    # Output tensor
    output_ptr,  # [total_tokens, num_head_q, v_head_dim]
    # Strides
    stride_po_tok,
    stride_po_head,
    stride_po_dim,
    stride_lse_tok,
    stride_lse_head,
    stride_o_tok,
    stride_o_head,
    stride_o_dim,
    # Constants
    TILE_Q: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    MAX_PARTIALS: tl.constexpr,
):
    """
    Each program processes one (reduce_group, head, token) combination.
    Grid: (num_reduce_groups, num_heads, TILE_Q)

    All heads are uniformly split and reduced together.
    """

    group_id = tl.program_id(0)
    head_id = tl.program_id(1)
    tok_offset = tl.program_id(2)  # q_tile

    # Load reduce group metadata (read once per block)
    start_idx = tl.load(reduce_indptr_ptr + group_id)
    end_idx = tl.load(reduce_indptr_ptr + group_id + 1)
    num_partials = end_idx - start_idx

    if num_partials == 0:
        return

    # Load final map: [qo_start, qo_end]
    final_map_offset = group_id * 2
    qo_start = tl.load(reduce_final_map_ptr + final_map_offset + 0)
    qo_end = tl.load(reduce_final_map_ptr + final_map_offset + 1)

    q_len = qo_end - qo_start
    tok_id = tok_offset

    # Skip if beyond valid range
    if tok_id >= q_len:
        return

    # compute max LSE and collect LSE values
    max_lse = -float("inf")
    lse_values = tl.zeros((MAX_PARTIALS,), dtype=tl.float32) - float("inf")

    for p_idx in range(num_partials):
        if p_idx < num_partials:
            partial_qo_loc = tl.load(reduce_partial_map_ptr + start_idx + p_idx)

            lse_offset = (
                partial_qo_loc + tok_id
            ) * stride_lse_tok + head_id * stride_lse_head
            lse = tl.load(partial_lse_ptr + lse_offset)

            is_valid = lse == lse
            lse = tl.where(is_valid, lse, -float("inf"))

            lse_values = tl.where(tl.arange(0, MAX_PARTIALS) == p_idx, lse, lse_values)

            # Update max
            max_lse = tl.maximum(max_lse, lse)

    # compute sum_exp
    sum_exp = 0.0
    for p_idx in tl.static_range(MAX_PARTIALS):
        if p_idx < num_partials:
            # Extract the lse value for this partition
            lse = tl.sum(tl.where(tl.arange(0, MAX_PARTIALS) == p_idx, lse_values, 0.0))
            exp_val = tl.exp(lse - max_lse)
            sum_exp += exp_val

    final_lse = max_lse + tl.log(sum_exp)

    # accumulate weighted outputs in chunks
    # Process V_HEAD_DIM in chunks of BLOCK_DIM
    num_dim_blocks = tl.cdiv(V_HEAD_DIM, BLOCK_DIM)

    for dim_block_id in range(num_dim_blocks):
        dim_offs = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offs < V_HEAD_DIM

        acc = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

        for p_idx in tl.static_range(MAX_PARTIALS):
            if p_idx < num_partials:
                partial_qo_loc = tl.load(reduce_partial_map_ptr + start_idx + p_idx)

                # Extract lse value
                lse = tl.sum(
                    tl.where(tl.arange(0, MAX_PARTIALS) == p_idx, lse_values, 0.0)
                )

                scale = tl.exp(lse - final_lse)

                # load partial output
                out_offset = (
                    (partial_qo_loc + tok_id) * stride_po_tok
                    + head_id * stride_po_head
                    + dim_offs * stride_po_dim
                )
                partial_out = tl.load(
                    partial_output_ptr + out_offset, mask=dim_mask, other=0.0
                )

                # Handle NaN in output (NaN != NaN)
                is_valid_out = partial_out == partial_out
                partial_out = tl.where(is_valid_out, partial_out, 0.0)

                acc += scale * partial_out

        output_offset = (
            (qo_start + tok_id) * stride_o_tok
            + head_id * stride_o_head
            + dim_offs * stride_o_dim
        )
        tl.store(
            output_ptr + output_offset,
            acc.to(output_ptr.dtype.element_ty),
            mask=dim_mask,
        )


# ============================================================================
# PYTHON WRAPPER — verbatim from aiter/mla.py L715-L780
# ============================================================================


def mla_prefill_reduce_triton(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    output: torch.Tensor,
    tile_q: int = 256,
    max_partials_static: int = None,
) -> None:
    """Triton version of mla_prefill_reduce.
    All heads are uniformly split and reduced together.
    """
    MAX_PARTIALS_STATIC = (
        max_partials_static if max_partials_static is not None else get_cu_num()
    )

    num_reduce_groups = reduce_indptr.shape[0] - 1
    _, num_heads, v_head_dim = partial_output.shape

    if num_reduce_groups == 0:
        return

    max_partials = 0
    for i in range(num_reduce_groups):
        num_p = (reduce_indptr[i + 1] - reduce_indptr[i]).item()
        max_partials = max(max_partials, num_p)

    if max_partials > MAX_PARTIALS_STATIC:
        raise ValueError(
            f"max_partials={max_partials} exceeds MAX_PARTIALS_STATIC={MAX_PARTIALS_STATIC}. "
            "Consider increasing MAX_PARTIALS_STATIC."
        )

    BLOCK_DIM = 64
    if v_head_dim <= 64:
        BLOCK_DIM = triton.next_power_of_2(v_head_dim)

    grid = (num_reduce_groups, num_heads, tile_q)

    _mla_prefill_reduce_kernel[grid](
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        output,
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        TILE_Q=tile_q,
        V_HEAD_DIM=v_head_dim,
        BLOCK_DIM=BLOCK_DIM,
        MAX_PARTIALS=MAX_PARTIALS_STATIC,
        num_warps=4,
    )


# ============================================================================
# REFERENCE IMPLEMENTATION (pure PyTorch)
# ============================================================================


def ref_mla_prefill_reduce(
    partial_output, partial_lse, reduce_indptr, reduce_final_map,
    reduce_partial_map, output, tile_q,
):
    """Pure PyTorch reference for the reduce kernel."""
    num_reduce_groups = reduce_indptr.shape[0] - 1
    device = partial_output.device
    dtype = partial_output.dtype
    _, num_heads, v_head_dim = partial_output.shape

    for group_id in range(num_reduce_groups):
        start_idx = reduce_indptr[group_id].item()
        end_idx = reduce_indptr[group_id + 1].item()
        num_partials = end_idx - start_idx
        if num_partials == 0:
            continue

        qo_start = reduce_final_map[group_id, 0].item()
        qo_end = reduce_final_map[group_id, 1].item()
        q_len = qo_end - qo_start

        partial_indices = []
        for pidx in range(start_idx, end_idx):
            partial_indices.append(reduce_partial_map[pidx].item())

        for head_idx in range(num_heads):
            stacked_lse = torch.stack([
                partial_lse[pqo:pqo + tile_q, head_idx]
                for pqo in partial_indices
            ], dim=0)
            stacked_out = torch.stack([
                partial_output[pqo:pqo + tile_q, head_idx, :]
                for pqo in partial_indices
            ], dim=0)

            nan_mask = torch.isnan(stacked_lse)
            clean_lse = torch.where(nan_mask, torch.tensor(float("-inf"), device=device), stacked_lse)
            max_lse = torch.max(clean_lse, dim=0)[0]
            exp_vals = torch.where(nan_mask, torch.zeros_like(stacked_lse), torch.exp(stacked_lse - max_lse.unsqueeze(0)))
            sum_exp = exp_vals.sum(dim=0)
            final_lse = max_lse + torch.log(sum_exp)
            scales = torch.exp(clean_lse - final_lse.unsqueeze(0)).unsqueeze(-1)

            clean_out = torch.where(torch.isnan(stacked_out), torch.zeros_like(stacked_out), stacked_out)
            final_output = (clean_out * scales).sum(dim=0)
            output[qo_start:qo_end, head_idx, :] = final_output[:q_len, :]

    return output


# ============================================================================
# SYNTHETIC INPUT BUILDER
# ============================================================================


def _build_reduce_inputs(batch_size, ctx_len, num_heads, v_head_dim, device="cuda"):
    """Build synthetic inputs that exercise the reduce kernel."""
    torch.manual_seed(42)
    tile_q = 256

    tiles_per_batch = max(1, math.ceil(ctx_len / tile_q))
    num_partials_per_group = min(4, max(1, math.ceil(ctx_len / 1024)))
    num_reduce_groups = batch_size * tiles_per_batch
    total_tokens = batch_size * ctx_len
    total_partial_tiles = num_reduce_groups * num_partials_per_group
    padded_partial_size = total_partial_tiles * tile_q

    max_partials_static = triton.next_power_of_2(num_partials_per_group)

    partial_output = torch.randn(
        (padded_partial_size, num_heads, v_head_dim),
        dtype=torch.float32, device=device,
    )
    partial_lse = (
        torch.randn((padded_partial_size, num_heads), dtype=torch.float32, device=device) * 3.0 + 5.0
    )

    reduce_indptr = torch.zeros(num_reduce_groups + 1, dtype=torch.int32, device="cpu")
    for g in range(num_reduce_groups):
        reduce_indptr[g + 1] = reduce_indptr[g] + num_partials_per_group
    reduce_indptr = reduce_indptr.to(device)

    reduce_final_map = torch.zeros((num_reduce_groups, 2), dtype=torch.int32, device="cpu")
    for b in range(batch_size):
        for t in range(tiles_per_batch):
            g = b * tiles_per_batch + t
            qo_start = b * ctx_len + t * tile_q
            qo_end = min(b * ctx_len + (t + 1) * tile_q, b * ctx_len + ctx_len)
            reduce_final_map[g, 0] = qo_start
            reduce_final_map[g, 1] = qo_end
    reduce_final_map = reduce_final_map.to(device)

    reduce_partial_map = torch.zeros(total_partial_tiles, dtype=torch.int32, device="cpu")
    for g in range(num_reduce_groups):
        for p in range(num_partials_per_group):
            idx = g * num_partials_per_group + p
            reduce_partial_map[idx] = idx * tile_q
    reduce_partial_map = reduce_partial_map.to(device)

    output = torch.zeros(
        (total_tokens, num_heads, v_head_dim), dtype=torch.bfloat16, device=device,
    )

    return (partial_output, partial_lse, reduce_indptr, reduce_final_map,
            reduce_partial_map, output, tile_q, max_partials_static)


# ============================================================================
# ENTRY POINTS (for GEAK harness)
# ============================================================================


def triton_op(partial_output, partial_lse, reduce_indptr, reduce_final_map,
              reduce_partial_map, output, tile_q, max_partials_static):
    mla_prefill_reduce_triton(
        partial_output, partial_lse, reduce_indptr, reduce_final_map,
        reduce_partial_map, output, tile_q,
        max_partials_static=max_partials_static,
    )
    return output


def torch_op(partial_output, partial_lse, reduce_indptr, reduce_final_map,
             reduce_partial_map, output, tile_q, max_partials_static):
    return ref_mla_prefill_reduce(
        partial_output, partial_lse, reduce_indptr, reduce_final_map,
        reduce_partial_map, output, tile_q,
    )


# ============================================================================
# CONFIG SPACE — shared with harness
# Derived from test_mla_prefill_ps.py config product:
#   causal × num_heads × ctx_len × batch_size
# Collapsed: dtype=fp8, kv_dtype=fp8, block_size=1, varlen=False
# qk_head_dim=192, v_head_dim=128
#
# For the standalone kernel.py test we only need (batch_size, ctx_len,
# num_heads, v_head_dim) since the reduce kernel is agnostic to causal/fp8.
# ============================================================================

EVAL_CONFIGS = [
    # (batch_size, ctx_len, num_heads, v_head_dim)
    # Full 120-config space matching test_mla_harness.py ALL_CONFIGS:
    # product(causal=[T,F], num_heads=[1,16], ctx_len=[21..16384], batch_size=[1,4,16])
    # The reduce kernel is causal-agnostic so causal=T / causal=F pairs are
    # identical here; the harness produces different metadata via aiter.
    (1, 21, 1, 128),       # [  0] causal=T nhead= 1 ctx=   21 bs= 1
    (4, 21, 1, 128),       # [  1] causal=T nhead= 1 ctx=   21 bs= 4
    (16, 21, 1, 128),      # [  2] causal=T nhead= 1 ctx=   21 bs=16
    (1, 64, 1, 128),       # [  3] causal=T nhead= 1 ctx=   64 bs= 1
    (4, 64, 1, 128),       # [  4] causal=T nhead= 1 ctx=   64 bs= 4
    (16, 64, 1, 128),      # [  5] causal=T nhead= 1 ctx=   64 bs=16
    (1, 256, 1, 128),      # [  6] causal=T nhead= 1 ctx=  256 bs= 1
    (4, 256, 1, 128),      # [  7] causal=T nhead= 1 ctx=  256 bs= 4
    (16, 256, 1, 128),     # [  8] causal=T nhead= 1 ctx=  256 bs=16
    (1, 512, 1, 128),      # [  9] causal=T nhead= 1 ctx=  512 bs= 1
    (4, 512, 1, 128),      # [ 10] causal=T nhead= 1 ctx=  512 bs= 4
    (16, 512, 1, 128),     # [ 11] causal=T nhead= 1 ctx=  512 bs=16
    (1, 1200, 1, 128),     # [ 12] causal=T nhead= 1 ctx= 1200 bs= 1
    (4, 1200, 1, 128),     # [ 13] causal=T nhead= 1 ctx= 1200 bs= 4
    (16, 1200, 1, 128),    # [ 14] causal=T nhead= 1 ctx= 1200 bs=16
    (1, 3200, 1, 128),     # [ 15] causal=T nhead= 1 ctx= 3200 bs= 1
    (4, 3200, 1, 128),     # [ 16] causal=T nhead= 1 ctx= 3200 bs= 4
    (16, 3200, 1, 128),    # [ 17] causal=T nhead= 1 ctx= 3200 bs=16
    (1, 5200, 1, 128),     # [ 18] causal=T nhead= 1 ctx= 5200 bs= 1
    (4, 5200, 1, 128),     # [ 19] causal=T nhead= 1 ctx= 5200 bs= 4
    (16, 5200, 1, 128),    # [ 20] causal=T nhead= 1 ctx= 5200 bs=16
    (1, 8192, 1, 128),     # [ 21] causal=T nhead= 1 ctx= 8192 bs= 1
    (4, 8192, 1, 128),     # [ 22] causal=T nhead= 1 ctx= 8192 bs= 4
    (16, 8192, 1, 128),    # [ 23] causal=T nhead= 1 ctx= 8192 bs=16
    (1, 10000, 1, 128),    # [ 24] causal=T nhead= 1 ctx=10000 bs= 1
    (4, 10000, 1, 128),    # [ 25] causal=T nhead= 1 ctx=10000 bs= 4
    (16, 10000, 1, 128),   # [ 26] causal=T nhead= 1 ctx=10000 bs=16
    (1, 16384, 1, 128),    # [ 27] causal=T nhead= 1 ctx=16384 bs= 1
    (4, 16384, 1, 128),    # [ 28] causal=T nhead= 1 ctx=16384 bs= 4
    (16, 16384, 1, 128),   # [ 29] causal=T nhead= 1 ctx=16384 bs=16
    (1, 21, 16, 128),      # [ 30] causal=T nhead=16 ctx=   21 bs= 1
    (4, 21, 16, 128),      # [ 31] causal=T nhead=16 ctx=   21 bs= 4
    (16, 21, 16, 128),     # [ 32] causal=T nhead=16 ctx=   21 bs=16
    (1, 64, 16, 128),      # [ 33] causal=T nhead=16 ctx=   64 bs= 1
    (4, 64, 16, 128),      # [ 34] causal=T nhead=16 ctx=   64 bs= 4
    (16, 64, 16, 128),     # [ 35] causal=T nhead=16 ctx=   64 bs=16
    (1, 256, 16, 128),     # [ 36] causal=T nhead=16 ctx=  256 bs= 1
    (4, 256, 16, 128),     # [ 37] causal=T nhead=16 ctx=  256 bs= 4
    (16, 256, 16, 128),    # [ 38] causal=T nhead=16 ctx=  256 bs=16
    (1, 512, 16, 128),     # [ 39] causal=T nhead=16 ctx=  512 bs= 1
    (4, 512, 16, 128),     # [ 40] causal=T nhead=16 ctx=  512 bs= 4
    (16, 512, 16, 128),    # [ 41] causal=T nhead=16 ctx=  512 bs=16
    (1, 1200, 16, 128),    # [ 42] causal=T nhead=16 ctx= 1200 bs= 1
    (4, 1200, 16, 128),    # [ 43] causal=T nhead=16 ctx= 1200 bs= 4
    (16, 1200, 16, 128),   # [ 44] causal=T nhead=16 ctx= 1200 bs=16
    (1, 3200, 16, 128),    # [ 45] causal=T nhead=16 ctx= 3200 bs= 1
    (4, 3200, 16, 128),    # [ 46] causal=T nhead=16 ctx= 3200 bs= 4
    (16, 3200, 16, 128),   # [ 47] causal=T nhead=16 ctx= 3200 bs=16
    (1, 5200, 16, 128),    # [ 48] causal=T nhead=16 ctx= 5200 bs= 1
    (4, 5200, 16, 128),    # [ 49] causal=T nhead=16 ctx= 5200 bs= 4
    (16, 5200, 16, 128),   # [ 50] causal=T nhead=16 ctx= 5200 bs=16
    (1, 8192, 16, 128),    # [ 51] causal=T nhead=16 ctx= 8192 bs= 1
    (4, 8192, 16, 128),    # [ 52] causal=T nhead=16 ctx= 8192 bs= 4
    (16, 8192, 16, 128),   # [ 53] causal=T nhead=16 ctx= 8192 bs=16
    (1, 10000, 16, 128),   # [ 54] causal=T nhead=16 ctx=10000 bs= 1
    (4, 10000, 16, 128),   # [ 55] causal=T nhead=16 ctx=10000 bs= 4
    (16, 10000, 16, 128),  # [ 56] causal=T nhead=16 ctx=10000 bs=16
    (1, 16384, 16, 128),   # [ 57] causal=T nhead=16 ctx=16384 bs= 1
    (4, 16384, 16, 128),   # [ 58] causal=T nhead=16 ctx=16384 bs= 4
    (16, 16384, 16, 128),  # [ 59] causal=T nhead=16 ctx=16384 bs=16
    (1, 21, 1, 128),       # [ 60] causal=F nhead= 1 ctx=   21 bs= 1
    (4, 21, 1, 128),       # [ 61] causal=F nhead= 1 ctx=   21 bs= 4
    (16, 21, 1, 128),      # [ 62] causal=F nhead= 1 ctx=   21 bs=16
    (1, 64, 1, 128),       # [ 63] causal=F nhead= 1 ctx=   64 bs= 1
    (4, 64, 1, 128),       # [ 64] causal=F nhead= 1 ctx=   64 bs= 4
    (16, 64, 1, 128),      # [ 65] causal=F nhead= 1 ctx=   64 bs=16
    (1, 256, 1, 128),      # [ 66] causal=F nhead= 1 ctx=  256 bs= 1
    (4, 256, 1, 128),      # [ 67] causal=F nhead= 1 ctx=  256 bs= 4
    (16, 256, 1, 128),     # [ 68] causal=F nhead= 1 ctx=  256 bs=16
    (1, 512, 1, 128),      # [ 69] causal=F nhead= 1 ctx=  512 bs= 1
    (4, 512, 1, 128),      # [ 70] causal=F nhead= 1 ctx=  512 bs= 4
    (16, 512, 1, 128),     # [ 71] causal=F nhead= 1 ctx=  512 bs=16
    (1, 1200, 1, 128),     # [ 72] causal=F nhead= 1 ctx= 1200 bs= 1
    (4, 1200, 1, 128),     # [ 73] causal=F nhead= 1 ctx= 1200 bs= 4
    (16, 1200, 1, 128),    # [ 74] causal=F nhead= 1 ctx= 1200 bs=16
    (1, 3200, 1, 128),     # [ 75] causal=F nhead= 1 ctx= 3200 bs= 1
    (4, 3200, 1, 128),     # [ 76] causal=F nhead= 1 ctx= 3200 bs= 4
    (16, 3200, 1, 128),    # [ 77] causal=F nhead= 1 ctx= 3200 bs=16
    (1, 5200, 1, 128),     # [ 78] causal=F nhead= 1 ctx= 5200 bs= 1
    (4, 5200, 1, 128),     # [ 79] causal=F nhead= 1 ctx= 5200 bs= 4
    (16, 5200, 1, 128),    # [ 80] causal=F nhead= 1 ctx= 5200 bs=16
    (1, 8192, 1, 128),     # [ 81] causal=F nhead= 1 ctx= 8192 bs= 1
    (4, 8192, 1, 128),     # [ 82] causal=F nhead= 1 ctx= 8192 bs= 4
    (16, 8192, 1, 128),    # [ 83] causal=F nhead= 1 ctx= 8192 bs=16
    (1, 10000, 1, 128),    # [ 84] causal=F nhead= 1 ctx=10000 bs= 1
    (4, 10000, 1, 128),    # [ 85] causal=F nhead= 1 ctx=10000 bs= 4
    (16, 10000, 1, 128),   # [ 86] causal=F nhead= 1 ctx=10000 bs=16
    (1, 16384, 1, 128),    # [ 87] causal=F nhead= 1 ctx=16384 bs= 1
    (4, 16384, 1, 128),    # [ 88] causal=F nhead= 1 ctx=16384 bs= 4
    (16, 16384, 1, 128),   # [ 89] causal=F nhead= 1 ctx=16384 bs=16
    (1, 21, 16, 128),      # [ 90] causal=F nhead=16 ctx=   21 bs= 1
    (4, 21, 16, 128),      # [ 91] causal=F nhead=16 ctx=   21 bs= 4
    (16, 21, 16, 128),     # [ 92] causal=F nhead=16 ctx=   21 bs=16
    (1, 64, 16, 128),      # [ 93] causal=F nhead=16 ctx=   64 bs= 1
    (4, 64, 16, 128),      # [ 94] causal=F nhead=16 ctx=   64 bs= 4
    (16, 64, 16, 128),     # [ 95] causal=F nhead=16 ctx=   64 bs=16
    (1, 256, 16, 128),     # [ 96] causal=F nhead=16 ctx=  256 bs= 1
    (4, 256, 16, 128),     # [ 97] causal=F nhead=16 ctx=  256 bs= 4
    (16, 256, 16, 128),    # [ 98] causal=F nhead=16 ctx=  256 bs=16
    (1, 512, 16, 128),     # [ 99] causal=F nhead=16 ctx=  512 bs= 1
    (4, 512, 16, 128),     # [100] causal=F nhead=16 ctx=  512 bs= 4
    (16, 512, 16, 128),    # [101] causal=F nhead=16 ctx=  512 bs=16
    (1, 1200, 16, 128),    # [102] causal=F nhead=16 ctx= 1200 bs= 1
    (4, 1200, 16, 128),    # [103] causal=F nhead=16 ctx= 1200 bs= 4
    (16, 1200, 16, 128),   # [104] causal=F nhead=16 ctx= 1200 bs=16
    (1, 3200, 16, 128),    # [105] causal=F nhead=16 ctx= 3200 bs= 1
    (4, 3200, 16, 128),    # [106] causal=F nhead=16 ctx= 3200 bs= 4
    (16, 3200, 16, 128),   # [107] causal=F nhead=16 ctx= 3200 bs=16
    (1, 5200, 16, 128),    # [108] causal=F nhead=16 ctx= 5200 bs= 1
    (4, 5200, 16, 128),    # [109] causal=F nhead=16 ctx= 5200 bs= 4
    (16, 5200, 16, 128),   # [110] causal=F nhead=16 ctx= 5200 bs=16
    (1, 8192, 16, 128),    # [111] causal=F nhead=16 ctx= 8192 bs= 1
    (4, 8192, 16, 128),    # [112] causal=F nhead=16 ctx= 8192 bs= 4
    (16, 8192, 16, 128),   # [113] causal=F nhead=16 ctx= 8192 bs=16
    (1, 10000, 16, 128),   # [114] causal=F nhead=16 ctx=10000 bs= 1
    (4, 10000, 16, 128),   # [115] causal=F nhead=16 ctx=10000 bs= 4
    (16, 10000, 16, 128),  # [116] causal=F nhead=16 ctx=10000 bs=16
    (1, 16384, 16, 128),   # [117] causal=F nhead=16 ctx=16384 bs= 1
    (4, 16384, 16, 128),   # [118] causal=F nhead=16 ctx=16384 bs= 4
    (16, 16384, 16, 128),  # [119] causal=F nhead=16 ctx=16384 bs=16
]

PROFILE_CONFIGS = [
    # Matches _pick(ALL_CONFIGS, 5) from test_mla_harness.py (indices [0,30,60,89,119]).
    (1, 21, 1, 128),       # [  0] causal=T nhead= 1 ctx=   21 bs= 1
    (1, 21, 16, 128),      # [ 30] causal=T nhead=16 ctx=   21 bs= 1
    (1, 21, 1, 128),       # [ 60] causal=F nhead= 1 ctx=   21 bs= 1
    (16, 16384, 1, 128),   # [ 89] causal=F nhead= 1 ctx=16384 bs=16
    (16, 16384, 16, 128),  # [119] causal=F nhead=16 ctx=16384 bs=16
]

RTOL, ATOL = 1e-2, 1e-2


# ============================================================================
# SELF-TEST HARNESS
# ============================================================================


def get_inputs(batch_size, ctx_len, num_heads, v_head_dim, device="cuda"):
    return _build_reduce_inputs(batch_size, ctx_len, num_heads, v_head_dim, device)


def check_correctness(batch_size, ctx_len, num_heads, v_head_dim) -> dict:
    try:
        (partial_output, partial_lse, reduce_indptr, reduce_final_map,
         reduce_partial_map, output_triton, tile_q, mps) = get_inputs(
            batch_size, ctx_len, num_heads, v_head_dim
        )
        output_ref = output_triton.clone()

        ref_mla_prefill_reduce(
            partial_output, partial_lse, reduce_indptr, reduce_final_map,
            reduce_partial_map, output_ref, tile_q,
        )
        mla_prefill_reduce_triton(
            partial_output, partial_lse, reduce_indptr, reduce_final_map,
            reduce_partial_map, output_triton, tile_q,
            max_partials_static=mps,
        )
        torch.cuda.synchronize()

        correct = torch.allclose(output_triton.float(), output_ref.float(), rtol=RTOL, atol=ATOL)
        if not correct:
            x = output_triton.double().flatten()
            y = output_ref.double().flatten()
            cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
            if cos_diff < 1e-4:
                correct = True
            max_diff = torch.max(torch.abs(output_triton.float() - output_ref.float())).item()
            return {"correct": correct, "max_diff": max_diff, "error": None}

        return {"correct": True, "max_diff": 0.0, "error": None}
    except Exception as e:
        return {"correct": False, "max_diff": float("inf"), "error": str(e)}


def _config_label(batch_size, ctx_len, num_heads, v_head_dim):
    return f"(B={batch_size},ctx={ctx_len},H={num_heads},D={v_head_dim})"


def benchmark_config(batch_size, ctx_len, num_heads, v_head_dim,
                     warmup=50, iters=200) -> dict:
    import time
    (partial_output, partial_lse, reduce_indptr, reduce_final_map,
     reduce_partial_map, output, tile_q, mps) = get_inputs(
        batch_size, ctx_len, num_heads, v_head_dim
    )
    for _ in range(warmup):
        out_t = output.clone()
        mla_prefill_reduce_triton(partial_output, partial_lse, reduce_indptr,
                                  reduce_final_map, reduce_partial_map, out_t, tile_q,
                                  max_partials_static=mps)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out_t = output.clone()
        mla_prefill_reduce_triton(partial_output, partial_lse, reduce_indptr,
                                  reduce_final_map, reduce_partial_map, out_t, tile_q,
                                  max_partials_static=mps)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) * 1000 / iters
    return {"triton_ms": triton_ms}


def evaluate(configs=None, warmup=50, iters=200, verbose=True) -> dict:
    configs = configs or EVAL_CONFIGS
    results, failures = [], []

    if verbose:
        print(f"{'Config':<40} {'Correct':>8} {'Triton':>10}")
        print("-" * 60)

    for cfg in configs:
        label = _config_label(*cfg)
        corr = check_correctness(*cfg)
        if not corr["correct"]:
            failures.append({"config": cfg, **corr})
            if verbose:
                err = corr["error"] or f"max_diff={corr['max_diff']:.2e}"
                print(f"{label:<40} {'FAIL':>8}   {err[:25]}")
            continue

        bench = benchmark_config(*cfg, warmup=warmup, iters=iters)
        results.append({"config": cfg, "correct": True, **bench})
        if verbose:
            print(f"{label:<40} {'PASS':>8} {bench['triton_ms']:>8.4f}ms")

    if verbose:
        print("-" * 60)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<40} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
    }


def run_profile(configs=None, warmup=3, iters=1, verbose=True):
    configs = configs or PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s)")
    for cfg in configs:
        (partial_output, partial_lse, reduce_indptr, reduce_final_map,
         reduce_partial_map, output, tile_q, mps) = get_inputs(*cfg)
        for _ in range(warmup):
            out = output.clone()
            mla_prefill_reduce_triton(partial_output, partial_lse, reduce_indptr,
                                      reduce_final_map, reduce_partial_map, out, tile_q,
                                      max_partials_static=mps)
        torch.cuda.synchronize()
        for _ in range(iters):
            out = output.clone()
            mla_prefill_reduce_triton(partial_output, partial_lse, reduce_indptr,
                                      reduce_final_map, reduce_partial_map, out, tile_q,
                                      max_partials_static=mps)
        torch.cuda.synchronize()
        if verbose:
            print(f"  {_config_label(*cfg)} done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLA Prefill Reduce Kernel")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("MLA Prefill Reduce Kernel")
    print("=" * 60)

    if args.profile:
        print("\n[Profile Mode]")
        run_profile()
    else:
        print("\n[Evaluation]")
        evaluate()

    print("=" * 60)
