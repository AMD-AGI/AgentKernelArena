# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Host launchers vendored from ROCm/aiter (commit 652f2d4b768239aefd515c929e84817681ff598d):
#   aiter/ops/triton/moe/moe_op_gemm_a4w4.py         (can_overflow_int32, should_upcast_indices,
#                                                     allocate_output, get_kernel_config, mxfp4_quant,
#                                                     moe_gemm_a4w4)
#   aiter/ops/triton/utils/gemm_config_utils.py      (pick_gemm_num_stages and its LDS estimate)
#   aiter/ops/triton/moe/moe_routing/routing.py      (sort_tokens, sort_tokens_fused,
#                                                     _compute_expt_data_internal, the block_m and
#                                                     HIST_BLOCK_M choices of routing())
#   aiter/ops/triton/moe/moe_routing/bitmatrix.py    (Bitmatrix.sum, sum_bitmatrix_rows)
#   aiter/ops/triton/moe/reduce.py                   (reduce_grouped, Triton path)
# and from vllm-project/vllm (commit edbc4969a76b25c531f7b9bb16b79984e65ef023, Apache-2.0),
#   vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py (make_routing_data's
#   pack_bitmatrix launch).
#
# The operator: bf16 tokens quantized to MXFP4 in the op, MXFP4 x MXFP4 expert GEMMs, SiLU(gate) * up,
# the bf16 intermediate requantized to MXFP4, routing weights applied in the down projection and the top-k
# rows summed. Pipeline (all Triton):
#   pack_bitmatrix -> bitmatrix row sums -> combined routing (expert histogram, offsets, tile map,
#   gather/scatter permutations, sorted gate weights) -> MXFP4 quantization of the tokens ->
#   GEMM1 (gather, fused SiLU-and-mul) -> MXFP4 quantization of the intermediate ->
#   GEMM2 (gate-weighted) -> grouped top-k reduction into ``out``.
import triton

from .moe_gemm_a4w4 import _moe_gemm_a4w4, _mxfp4_quant_kernel
from .pack_bitmatrix import pack_bitmatrix
from .reduce import _reduce_grouped
from .routing import _combined_routing, _combined_routing_fused, _sum_bitmatrix_rows

import torch


# ----------------------------------------------------------------------------- routing data
class ExptData:
    def __init__(self, hist, token_offs_raw, token_offs_pad, block_pid_map):
        self.hist = hist
        self.token_offs_raw = token_offs_raw
        self.token_offs_pad = token_offs_pad
        self.block_pid_map = block_pid_map


class RoutingData:
    def __init__(self, block_m, gate_scal, expt_hist, n_expts_tot, n_expts_act, expt_data):
        self.block_m = block_m
        self.gate_scal = gate_scal
        self.expt_hist = expt_hist
        self.n_expts_tot = n_expts_tot
        self.n_expts_act = n_expts_act
        self.expt_data = expt_data

    def n_blocks(self, n_rows, block_m):
        if n_rows <= self.n_expts_tot:
            return n_rows
        else:
            return (
                triton.cdiv(max(n_rows - self.n_expts_tot + 1, 0), block_m)
                + self.n_expts_tot
                - 1
            )


def log2_power_of_two(x):
    assert x > 0 and (x & (x - 1)) == 0, "x must be a power of two"
    return x.bit_length() - 1


def _compute_expt_data_internal(n_expts_tot, n_gates, block_m, device):
    BLOCK = triton.next_power_of_2(n_expts_tot)
    cdiv = triton.cdiv
    block_m_log2 = log2_power_of_two(block_m)
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // block_m)

    # allocate memory
    def pad(x):
        return cdiv(x, BLOCK) * BLOCK

    dtype = torch.int32

    token_offs_combined = torch.empty(
        (2, pad(n_expts_tot + 1)), dtype=dtype, device=device
    )

    token_offs_raw = token_offs_combined[0][: n_expts_tot + 1]
    token_offs_pad = token_offs_combined[1][: n_expts_tot + 1]

    block_pid_map = torch.empty((max_n_tiles,), dtype=dtype, device=device)

    blocks1 = n_expts_tot
    return token_offs_raw, token_offs_pad, block_pid_map, blocks1, BLOCK, block_m_log2


class Bitmatrix:
    """Packed top-k membership bits [n_rows, n_cols]; the sum scratchpads start zeroed."""

    def __init__(self, data, shape, scratchpad, scratchpad_partials):
        self.data = data
        self.shape = shape
        self.scratchpad = scratchpad
        self.scratchpad_partials = scratchpad_partials


def sum_bitmatrix_rows(x, out_ret, partials_block_size=None):
    assert partials_block_size is not None
    cdiv = triton.cdiv
    PARTIALS_BLOCK_M = partials_block_size
    n_rows, n_cols = x.shape
    assert out_ret.shape == (n_cols,)

    TILE_SIZE = 8
    BLOCK_MM = PARTIALS_BLOCK_M * TILE_SIZE

    pids_x = cdiv(n_rows, BLOCK_MM)
    pids_y = cdiv(n_cols, 32)
    out_partials = x.scratchpad_partials

    # output tensors
    _sum_bitmatrix_rows[(pids_x, pids_y)](
        x.data,
        n_rows,
        x.data.stride(0),
        x.data.stride(1),  # input
        out_ret,  # output [final reduction]
        out_partials,
        out_partials.stride(0),
        out_partials.stride(1),
        out_partials.shape[1],
        pids_x,  # output [partial reductions]
        BLOCK_M=PARTIALS_BLOCK_M,
        BLOCK_MM=BLOCK_MM,  # constants
        num_warps=8,
    )

    out_partials = out_partials[: cdiv(n_rows, PARTIALS_BLOCK_M), :]

    return out_ret, out_partials


def make_bitmatrix(topk_ids, n_expts_tot, HIST_BLOCK_M):
    """vLLM make_routing_data's pack_bitmatrix, with the scratchpads AITER's top-k kernel zeroes."""
    n_rows, num_topk = topk_ids.shape
    device = topk_ids.device
    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32
    bm_cols = triton.cdiv(n_expts_tot, BLOCK_SIZE_K)  # n_bitpacks
    bitmatrix = torch.zeros((n_rows, bm_cols), dtype=torch.uint32, device=device)
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    n_cols = bm_cols * 32
    # AITER topk(): scratchpad [n_cols] and scratchpad_partials [pids_x * TILE_SIZE, n_cols], zeroed.
    TILE_SIZE = 8
    BLOCK_MM = HIST_BLOCK_M * TILE_SIZE
    pids_x = triton.cdiv(n_rows, BLOCK_MM)
    scratchpad = torch.zeros((n_cols,), dtype=torch.int32, device=device)
    scratchpad_partials = torch.zeros((n_cols, pids_x * TILE_SIZE), dtype=torch.int32, device=device)
    scratchpad_partials = torch.transpose(scratchpad_partials, 0, 1)
    return Bitmatrix(bitmatrix, [n_rows, n_cols], scratchpad, scratchpad_partials)


def sort_tokens(expt_scal, expt_indx, n_expts_tot, bitmatrix, block_m, HIST_BLOCK_M):
    cdiv = triton.cdiv

    device = expt_scal.device
    dtype = expt_scal.dtype
    n_tokens, n_expts_act = expt_scal.shape
    n_gates = n_tokens * n_expts_act
    n_expts_act_pad = triton.next_power_of_2(n_expts_act)

    # Bitmatrix.sum(): the zeroed scratchpad receives the histogram.
    hist, partial_hist = sum_bitmatrix_rows(bitmatrix, bitmatrix.scratchpad[: bitmatrix.shape[1]],
                                            partials_block_size=HIST_BLOCK_M)
    hist = hist[:n_expts_tot]
    assert hist.dtype == torch.int32
    # scratchpad
    if n_gates <= 65536 - 1:  # save one token id for OOB checks
        combined_indx = torch.empty(n_gates * 2, dtype=torch.uint16, device=device)
    else:
        combined_indx = torch.empty(n_gates * 2, dtype=torch.int32, device=device)
    # output
    topk_indx = combined_indx[:n_gates]
    gate_indx = combined_indx[n_gates:]
    gate_scal = torch.empty(n_gates, dtype=dtype, device=device)

    token_offs_raw, token_offs_pad, block_pid_map, blocks1a, BLOCK_A, block_m_log2 = (
        _compute_expt_data_internal(n_expts_tot, n_gates, block_m, device)
    )

    blocks1b = cdiv(n_tokens, HIST_BLOCK_M)

    indx_offs = partial_hist

    _combined_routing[(blocks1a + blocks1b,)](
        topk_indx,
        gate_indx,
        gate_scal,  # outputs
        expt_scal,
        expt_indx,
        indx_offs,
        indx_offs.stride(0),
        indx_offs.stride(1),  # inputs
        n_gates,  # input shape
        HIST_BLOCK_M,
        n_tokens % HIST_BLOCK_M == 0,
        n_expts_act,  # constants
        n_expts_act_pad,
        hist,
        n_expts_tot,
        token_offs_raw,
        token_offs_pad,
        blocks1a,
        block_pid_map,
        block_pid_map.shape[0],
        block_m_log2,
        BLOCK_A=BLOCK_A,
        EQUAL_A=(hist.shape[0] == BLOCK_A),  # optimization parameters
        USE_TDM=False,  # tensor descriptors exist only on gfx1250
        num_warps=1,
    )

    return (
        hist,
        topk_indx,
        gate_indx,
        gate_scal,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
    )


def sort_tokens_fused(
    expt_scal, expt_indx, n_expts_tot, bitmatrix, block_m, HIST_BLOCK_M
):
    cdiv = triton.cdiv

    device = expt_scal.device
    dtype = expt_scal.dtype
    n_tokens, n_expts_act = expt_scal.shape
    n_gates = n_tokens * n_expts_act
    n_expts_act_pad = triton.next_power_of_2(n_expts_act)

    hist = bitmatrix.scratchpad
    hist = hist[:n_expts_tot]
    assert hist.dtype == torch.int32
    num_blocks_bitmatrix = cdiv(bitmatrix.shape[1], 32)
    # scratchpad
    if n_gates <= 65536:
        combined_indx = torch.empty(n_gates * 2, dtype=torch.uint16, device=device)
    else:
        combined_indx = torch.empty(n_gates * 2, dtype=torch.int32, device=device)
    # output
    topk_indx = combined_indx[:n_gates]
    gate_indx = combined_indx[n_gates:]
    gate_scal = torch.empty(n_gates, dtype=dtype, device=device)

    token_offs_raw, token_offs_pad, block_pid_map, blocks1a, BLOCK_A, block_m_log2 = (
        _compute_expt_data_internal(n_expts_tot, n_gates, block_m, device)
    )

    blocks1b = cdiv(n_tokens, HIST_BLOCK_M)

    _combined_routing_fused[(blocks1a + blocks1b,)](
        topk_indx,
        gate_indx,
        gate_scal,  # outputs
        expt_scal,
        expt_indx,
        bitmatrix.data,
        bitmatrix.shape[0],
        bitmatrix.data.stride(0),
        bitmatrix.data.stride(1),
        num_blocks_bitmatrix,
        n_gates,  # input shape
        HIST_BLOCK_M,
        n_tokens % HIST_BLOCK_M == 0,
        n_expts_act,  # constants
        n_expts_act_pad,
        n_expts_tot,
        hist,
        token_offs_raw,
        token_offs_pad,
        blocks1a,
        block_pid_map,
        block_pid_map.shape[0],
        block_m_log2,
        BLOCK_A=BLOCK_A,
        EQUAL_A=(hist.shape[0] == BLOCK_A),  # optimization parameters
        USE_TDM=False,  # tensor descriptors exist only on gfx1250
        num_warps=1,
    )

    return (
        hist,
        topk_indx,
        gate_indx,
        gate_scal,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
    )


def routing_from_topk(expt_scal, expt_indx, n_expts_tot):
    """AITER routing() for precomputed top-k (the top-k kernel replaced by vLLM's pack_bitmatrix)."""
    num_tokens, n_expts_act = expt_scal.shape
    # block_m heuristic of AITER routing()
    m = num_tokens * n_expts_act
    tokens_per_expt = max(1, m // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    if num_tokens <= 16:
        HIST_BLOCK_M = triton.next_power_of_2(num_tokens)
        sort_fn = sort_tokens_fused
    else:
        HIST_BLOCK_M = 32
        sort_fn = sort_tokens
    bitmatrix = make_bitmatrix(expt_indx, n_expts_tot, HIST_BLOCK_M)
    (
        hist,
        topk_indx,
        gate_indx,
        gate_scal,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
    ) = sort_fn(expt_scal, expt_indx, n_expts_tot, bitmatrix, block_m, HIST_BLOCK_M)
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)
    return (
        RoutingData(block_m, gate_scal, hist, n_expts_tot, n_expts_act, expt_data),
        topk_indx,
        gate_indx,
    )



# ----------------------------------------------------------------------------- MoE GEMM
def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def allocate_output(
    x,
    w,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
    out=None,
):
    # ---- output ------
    N = w.shape[-1]
    # by default - M is number of rows in the activations
    M = x.shape[-2]
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.shape[0]
    # final output
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = (
            scatter_indx.shape[0] // routing_data.n_expts_act
        )  # compressed number of rows
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=x.device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        if out is not None:
            assert tuple(out.shape) == final_shape and out.dtype == out_dtype
            final_output = out
        else:
            final_output = torch.empty(final_shape, device=x.device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output



# gfx950 LDS capacity of AITER's arch_info._LDS_CAP_BYTES (the task runs on gfx950 only).
_LDS_CAP_BYTES_GFX950 = 163840


def _padded_size_32_4(n):
    pad = (n >> 5) << 2
    if (n & 31) == 0 and pad >= 4:
        pad -= 4
    return n + pad


def _padded_size_pow2(n, interval, padding):
    log2_i = (interval - 1).bit_length()
    log2_p = (padding - 1).bit_length() if padding else 0
    pad = (n >> log2_i) << log2_p
    if n % interval == 0 and pad >= padding:
        pad -= padding
    return n + pad


def _gemm_lds_bytes(
    block_m, block_n, block_k, bits_a, bits_b, num_stages, use_async_padding
):
    elem_a = block_m * block_k
    elem_b = block_k * block_n
    if use_async_padding:
        # Padded shared encoding + N buffers (matches TensorAtlas
        # _estimate_triton_lds_async_copy / tritonBLAS origami).
        pa = _padded_size_32_4(elem_a)
        pb = _padded_size_32_4(elem_b)
        if block_k & (block_k - 1) == 0:
            pa = max(pa, _padded_size_pow2(elem_a, block_k, 8))
        if block_n & (block_n - 1) == 0:
            pb = max(pb, _padded_size_pow2(elem_b, block_n, 8))
        return num_stages * (pa * bits_a + pb * bits_b) // 8
    # Non-async: (N-1) extra buffer pairs beyond the active stage.
    LDSA = elem_a * bits_a
    LDSB = elem_b * bits_b
    if num_stages <= 1:
        return max(LDSA, LDSB) // 8
    return (LDSA + LDSB) * (num_stages - 1) // 8


def pick_gemm_num_stages(
    block_m, block_n, block_k, bits_a, bits_b, use_async_padding=False
):
    assert min(block_m, block_n, block_k, bits_a, bits_b) > 0
    # bits_a / bits_b: element bit-widths (8 for fp8, 4 for mxfp4).
    # use_async_padding: True when the kernel lowers to async direct-to-LDS
    # with padded shared encoding (e.g. a4w4 on gfx950).
    cap = _LDS_CAP_BYTES_GFX950
    lds = _gemm_lds_bytes(
        block_m, block_n, block_k, bits_a, bits_b, 2, use_async_padding
    )
    return 2 if lds <= cap else 1


def get_kernel_config(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None

    split_k = 1
    if block_m == 16:
        block_n = 128
        block_k = 256
        num_warps = 4

        grid_m = routing_data.n_blocks(m, block_m)
        grid_n = triton.cdiv(n, block_n)
        grid = grid_m * grid_n * split_k
        while block_n >= 64 and grid < 256:
            block_n = block_n // 2
            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k
    else:
        # for scale preshuffling
        block_n = 512
        block_k = 256
        num_warps = 8
    num_stages = pick_gemm_num_stages(
        block_m, block_n, block_k, 4, 4, use_async_padding=True
    )

    ret = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "group_m": group_m,
        "xcd_swizzle": xcd_swizzle,
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }
    return ret


def reduce_grouped(
    x,
    indx,
    out,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    reduction_n=1,
    out_dtype=None,
    swiglu_add_residual=True,
):
    """Sum, for each group g, the K rows of ``x`` named by ``indx[g, :]`` (fp32 accumulation)."""
    if indx is None and x.shape[0] == 1:
        return x.squeeze(0)
    if indx is not None:
        num_groups = indx.shape[0]
    else:
        num_groups = x.shape[-2]
    K = 1 if indx is None else indx.shape[1]
    out_dtype = x.dtype if out_dtype is None else out_dtype
    assert x.shape[-1] % reduction_n == 0

    BLOCK_N = 512
    num_blocks = triton.cdiv(x.shape[-1], BLOCK_N)

    _reduce_grouped[(num_blocks * num_groups,)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        indx,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        num_blocks,
        apply_swiglu,
        alpha,
        limit,
        reduction_n,
        BLOCK_N=BLOCK_N,
        EVEN_N=(x.shape[-1] % BLOCK_N == 0),
        K=K,
        SWIGLU_ADD_RESIDUAL=swiglu_add_residual,
        USE_TDM=False,  # tensor descriptors exist only on gfx1250
        Residual=None,
        stride_extres_m=0,
        stride_extres_n=0,
        HAS_EXT_RESIDUAL=False,
        num_warps=2,
    )
    return out




# This is fixed by spec for MXFP4. Do not tune this.
MXFP4_QUANT_BLOCK_SIZE = 32


def mxfp4_quant(
    x: torch.Tensor,
    block_size_m: int = 16,
    block_size_n: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor `x` of shape [M, K] (bf16/fp16/fp32) to MXFP4 (E2M1) format
    quantized along the K dimension.

    Returns:
    - A packed MXFP4 tensor `x_fp4` of shape [M, N // 2] (stored as uint8), where
        each byte stores two 4-bit values.
    - A block-scale tensor `x_scale` of shape [M, N / 32], where each entry
        corresponds to one MXFP4 quantization block of 32 elements along the K dimension.
    """
    M, N = x.shape
    assert N % MXFP4_QUANT_BLOCK_SIZE == 0
    assert block_size_n % MXFP4_QUANT_BLOCK_SIZE == 0

    # Upstream first copies x to float32 on the host; the kernel converts each
    # loaded tile to float32 itself, so x is passed as is.
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    x_scale = torch.empty(
        (M, N // MXFP4_QUANT_BLOCK_SIZE), dtype=torch.uint8, device=x.device
    )

    grid = (
        triton.cdiv(M, block_size_m),
        triton.cdiv(N, block_size_n),
    )

    _mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        x_scale,
        x.stride(0),
        x.stride(1),
        x_fp4.stride(0),
        x_fp4.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        M,
        N,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        EVEN_M_N=(M % block_size_m == 0) and (N % block_size_n == 0),
    )

    return x_fp4, x_scale


def moe_gemm_a4w4(
    x,
    w,
    x_scales,
    w_scales,
    x_static_scale=None,
    quant_static_scale=None,
    bias=None,
    routing_data=None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    swizzle_mx_scale=None,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    swiglu_add_residual=True,
    unpadded_N=None,
    unpadded_K=None,
    out=None,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """
    assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    x_has_mx = x_scales is not None
    if x_has_mx:
        assert x.stride(-1) == 1, "'x' must be row-major when it has data-type mxfp"
    if x_has_mx:
        stride_x_mx_m = x_scales.stride(0)
        stride_x_mx_k = x_scales.stride(1)
    else:
        stride_x_mx_m = 0
        stride_x_mx_k = 0
    # determine shapes
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1] * 2, w.shape[-1]
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K
    # compute optimization flags
    config = get_kernel_config(M, N, K, routing_data)
    if apply_swiglu and config["split_k"] > 1:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = True
        reduction_n_reduction = 2
    elif apply_swiglu:
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    else:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    # allocate output memory
    y, y_final = allocate_output(
        x,
        w,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        config["block_m"],
        config["split_k"],
        out=out,
    )
    stride_bias = None if bias is None else bias.stride(0)
    # moe metadata
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map
    # spmd grid
    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]
    # launch kernel
    _moe_gemm_a4w4[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        x_scales,
        stride_x_mx_m,
        stride_x_mx_k,
        w,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scales,
        w_scales.stride(0),
        w_scales.stride(1),
        w_scales.stride(2),
        x_static_scale,
        quant_static_scale,
        bias,
        stride_bias,
        gammas,
        N,
        K,
        gather_indx,
        expt_hist,
        expt_token_offs_raw,
        expt_hist_sum,
        expt_block_pid_map,
        grid_m,
        grid_n,
        apply_swiglu_matmul,
        alpha,
        limit,
        reduction_n_matmul,
        swiglu_add_residual,
        routing_data.n_expts_act,
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["group_m"],
        XCD_SWIZZLE=config["xcd_swizzle"],
        SWIZZLE_MX_SCALE=swizzle_mx_scale,
        SPLIT_K=config["split_k"],
        EVEN_K=K % config["block_k"] == 0,
        MASK_K_LIMIT=K % config["block_k"],
        W_CACHE_MODIFIER=config["w_cache_modifier"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        UPCAST_INDICES=should_upcast_indices(x, w, y),
        waves_per_eu=config["waves_per_eu"],
        matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
        kpack=config["kpack"],
    )
    # Build grouped reduction inputs in a uniform way
    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        apply_swiglu_reduction,
        alpha,
        limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
        swiglu_add_residual=swiglu_add_residual,
    )
    return y_final


# ----------------------------------------------------------------------------- entry point
def run(*, hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids, out):
    n_expts_tot = w1.shape[0]
    routing_data, gather_indx, scatter_indx = routing_from_topk(topk_weights, topk_ids, n_expts_tot)
    # Activations are quantized to MXFP4 per 1x32 block along the hidden axis.
    x_fp4, x_scale = mxfp4_quant(hidden_states)
    # w1 [E, 2*inter (gate/up rows interleaved), hidden/2] and w2 [E, hidden, inter/2] hold K-packed e2m1
    # codes; the kernels take them as column-major [E, K/2, N] views. The scales are stored in the gfx950
    # MX-scale tile order as [E, N/32, K] and read through their [E, K, N/32] transposed views, the
    # layout of AITER's shuffle_scale_moe (see inputs.py).
    intermediate = moe_gemm_a4w4(
        x_fp4,
        w1.transpose(1, 2),
        x_scale,
        w1_scale.transpose(1, 2),
        routing_data=routing_data,
        gather_indx=gather_indx,
        swizzle_mx_scale="CDNA4_SCALE",
        out_dtype=hidden_states.dtype,
        apply_swiglu=True,
        alpha=1.0,
        limit=None,
        swiglu_add_residual=False,
    )
    # The SiLU(gate) * up intermediate (bfloat16, expert-sorted rows) is requantized to MXFP4.
    inter_fp4, inter_scale = mxfp4_quant(intermediate)
    moe_gemm_a4w4(
        inter_fp4,
        w2.transpose(1, 2),
        inter_scale,
        w2_scale.transpose(1, 2),
        routing_data=routing_data,
        scatter_indx=scatter_indx,
        gammas=routing_data.gate_scal,
        swizzle_mx_scale="CDNA4_SCALE",
        out_dtype=out.dtype,
        out=out,
    )
