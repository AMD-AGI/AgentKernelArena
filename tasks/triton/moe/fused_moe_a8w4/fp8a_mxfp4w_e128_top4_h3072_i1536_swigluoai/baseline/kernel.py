# Host launchers vendored from ROCm/aiter (commit 652f2d4b768239aefd515c929e84817681ff598d, MIT, see LICENSE; the upstream files carry no
# copyright header, so the repository LICENSE applies):
#   aiter/ops/triton/moe/moe_op_gemm_a8w4.py       (_get_a8w4_dispatch, can_overflow_int32, should_upcast_indices,
#                                                   allocate_output, get_kernel_config_triton, moe_gemm_a8w4: the
#                                                   Triton path)
#   aiter/ops/triton/moe/quant_moe.py              (downcast_to_static_fp8)
#   aiter/ops/triton/configs/moe/gfx950-A8W4.json  (as configs.py)
#   aiter/ops/triton/moe/moe_routing/routing.py    (sort_tokens, sort_tokens_fused,
#                                                   _compute_expt_data_internal, the block_m and
#                                                   HIST_BLOCK_M choices of routing())
#   aiter/ops/triton/moe/moe_routing/bitmatrix.py  (Bitmatrix.sum, sum_bitmatrix_rows)
#   aiter/ops/triton/moe/reduce.py                 (reduce_grouped, Triton path)
# and from vllm-project/vllm (commit edbc4969a76b25c531f7b9bb16b79984e65ef023, Apache-2.0),
#   vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py (make_routing_data's
#   pack_bitmatrix launch) and
#   vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py (triton_kernel_fused_mxfp4_w4a8_experts:
#   static FP8 quantization of the hidden states, the fused-SwiGLU gather GEMM that re-quantizes its output to FP8,
#   then the gate-weighted scatter GEMM).
#
# The operator: static-FP8 tokens x MXFP4 experts with expert biases and the GPT-OSS SwiGLU (alpha 1.702, limit 7,
# up + 1), routed by precomputed top-k ids/weights. Pipeline (all Triton):
#   static FP8 quantization -> pack_bitmatrix -> bitmatrix row sums -> combined routing (expert histogram,
#   offsets, tile map, gather/scatter permutations, sorted gate weights) -> GEMM1 (gather, bias, fused SwiGLU,
#   static FP8 output) -> GEMM2 (bias, gate-weighted) -> grouped top-k reduction into ``out``.
import triton

from .configs import GFX950_A8W4
from .moe_gemm_a8w4 import _moe_gemm_a8w4
from .pack_bitmatrix import pack_bitmatrix
from .quant_moe import _downcast_to_static_fp8
from .reduce import _reduce_grouped
from .routing import _combined_routing, _combined_routing_fused, _sum_bitmatrix_rows

import torch

SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0


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
def downcast_to_static_fp8(x, scale):
    """AITER downcast_to_static_fp8 (gfx950: float8_e4m3fn): y = fp8(x / scale) for a one-element scale tensor."""
    M, N = x.shape
    y = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    BLOCK_M = min(triton.next_power_of_2(M), 128)
    if M <= 4096:
        BLOCK_N = 32
    else:
        BLOCK_N = 64
    grid_m = triton.cdiv(x.shape[0], BLOCK_M)
    grid_n = triton.cdiv(x.shape[1], BLOCK_N)
    _downcast_to_static_fp8[(grid_m, grid_n)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        scale,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        num_warps=8,
    )
    return y


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def allocate_output(
    M,
    N,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
    device,
    out=None,
):
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
    matmul_output = torch.empty(matmul_shape, device=device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        if out is not None:
            assert tuple(out.shape) == final_shape and out.dtype == out_dtype
            final_output = out
        else:
            final_output = torch.empty(final_shape, device=device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def get_kernel_config_triton(m, n, k, routing_data, swizzle_mx_scale=None):
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    split_k = 1

    # Tuned dispatch: per-(block_m, N, K) winners (AITER gfx950-A8W4.json).
    tuned = GFX950_A8W4.get(f"bm{block_m}_n{n}_k{k}")
    if tuned is not None:
        return {
            "block_m": block_m,
            "block_n": tuned["BLOCK_SIZE_N"],
            "block_k": tuned["BLOCK_SIZE_K"],
            "num_warps": tuned["num_warps"],
            "num_stages": tuned["num_stages"],
            "group_m": group_m,
            "xcd_swizzle": xcd_swizzle,
            "w_cache_modifier": w_cache_modifier,
            "split_k": split_k,
            "waves_per_eu": tuned.get("waves_per_eu", 0),
            "matrix_instr_nonkdim": tuned.get("matrix_instr_nonkdim", 16),
            "kpack": tuned.get("kpack", 1),
        }

    # Shapes not in the tuned table: the tile geometry and num_stages of the entry with the same (N, K) and any
    # block_m. Under CDNA4 swizzle, skip BLOCK_K<256 entries since unswizzle can't compile them.
    dispatch = GFX950_A8W4
    proxy = next(
        (
            v
            for bm in (16, 32, 64, 128)
            if (v := dispatch.get(f"bm{bm}_n{n}_k{k}")) is not None
            and (swizzle_mx_scale != "CDNA4_SCALE" or v.get("BLOCK_SIZE_K", 0) >= 256)
        ),
        None,
    )
    assert proxy is not None, f"no tuned A8W4 tile for N={n}, K={k}"
    return {
        "block_m": block_m,
        "block_n": proxy["BLOCK_SIZE_N"],
        "block_k": proxy["BLOCK_SIZE_K"],
        "num_warps": proxy["num_warps"],
        "num_stages": proxy["num_stages"],
        "group_m": group_m,
        "xcd_swizzle": xcd_swizzle,
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": proxy.get("waves_per_eu", 0),
        "matrix_instr_nonkdim": proxy.get("matrix_instr_nonkdim", 16),
        "kpack": proxy.get("kpack", 1),
    }


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


def moe_gemm_a8w4(
    x,
    w,
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
    out=None,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """
    assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    # determine shapes
    num_tokens = x.shape[-2]
    M = num_tokens if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    padded_N = N
    # compute optimization flags
    config = get_kernel_config_triton(M, N, K, routing_data, swizzle_mx_scale)
    # CDNA4 swizzle requires BLOCK_K % 256 == 0; some tuned small-K entries pick BK<256 for utilization.
    if swizzle_mx_scale == "CDNA4_SCALE" and config["block_k"] < 256:
        config["block_k"] = 256
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
        M,
        padded_N,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        config["block_m"],
        config["split_k"],
        x.device,
        out=out,
    )
    stride_bias = None if bias is None else bias.stride(0)
    # moe metadata
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map
    # pid grid
    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]
    # launch kernel
    _moe_gemm_a8w4[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        None,  # x_scales: the activations carry a static per-tensor scale, not MX scales
        0,
        0,
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
def run(*, hidden_states, w1, w1_scale, w1_bias, w1_input_scale, w2, w2_scale, w2_bias, w2_input_scale,
        topk_weights, topk_ids, out):
    n_expts_tot = w1.shape[0]
    routing_data, gather_indx, scatter_indx = routing_from_topk(topk_weights, topk_ids, n_expts_tot)
    # Static FP8 quantization of the hidden states with the GEMM1 input scale.
    hidden_q = downcast_to_static_fp8(hidden_states, w1_input_scale)
    # w1 [E, 2*inter (gate/up rows interleaved), hidden/2] and w2 [E, hidden, inter/2] hold K-packed e2m1
    # codes; the kernels take them as column-major [E, K/2, N] views. Scales are CDNA4-swizzled
    # (see inputs.py), as triton_kernel_fused_mxfp4_w4a8_experts consumes them on gfx950 at tensor-parallel
    # size 2 or below.
    intermediate = moe_gemm_a8w4(
        hidden_q,
        w1.transpose(1, 2),
        w1_scale,
        x_static_scale=w1_input_scale,
        quant_static_scale=w2_input_scale,
        bias=w1_bias,
        routing_data=routing_data,
        gather_indx=gather_indx,
        swizzle_mx_scale="CDNA4_SCALE",
        out_dtype=torch.float8_e4m3fn,
        apply_swiglu=True,
        alpha=SWIGLU_ALPHA,
        limit=SWIGLU_LIMIT,
    )
    moe_gemm_a8w4(
        intermediate,
        w2.transpose(1, 2),
        w2_scale,
        x_static_scale=w2_input_scale,
        bias=w2_bias,
        routing_data=routing_data,
        scatter_indx=scatter_indx,
        gammas=routing_data.gate_scal,
        swizzle_mx_scale="CDNA4_SCALE",
        out_dtype=out.dtype,
        out=out,
    )
