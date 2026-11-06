# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused mHC post + next pre with output RMSNorm, in Triton.

Vendored from ROCm/aiter: the ``mhc_post_pre`` launcher of aiter/ops/triton/fusions/mhc.py (two
launches: the fused post + split-K pre GEMM, then reduce/Sinkhorn/apply), its tuned gfx950
configuration for C=4096 (aiter/ops/triton/configs/gfx950-MHC_FUSED_SINKHORN-C=4096.json with the
selection rule of aiter/ops/triton/utils/mhc_config_utils.py::get_mhc_config), and the forward
launch of AITER's Triton RMSNorm (aiter/ops/triton/normalization/rmsnorm.py::_rmsnorm_forward) for
the output RMSNorm, which the AITER Triton mHC path does not fuse. ``mhc_post_pre`` writes the
un-normalized layer input in float32 and the RMSNorm reads it, so the normalized output is rounded
to bfloat16 once.
"""

import functools

import torch
import triton

from .mhc_kernels import _mhc_post_pre_reduce_apply_kernel, _mhc_post_pre_split_kernel
from .rmsnorm_kernels import _rms_norm_kernel


# aiter/ops/triton/configs/gfx950-MHC_FUSED_SINKHORN-C=4096.json: the file get_mhc_config selects
# for C=4096 on gfx950 (the largest C-specific threshold <= 4096).
_MHC_FUSED_SINKHORN_GFX950_C4096 = {
    "M_LEQ_1": {"BLOCK_M": 16, "BLOCK_K": 256, "BLOCK_C": 128, "NUM_KSPLIT": 32, "num_warps": 4,
                "num_stages": 1, "waves_per_eu": 0},
    "M_LEQ_1024": {"BLOCK_M": 32, "BLOCK_K": 256, "BLOCK_C": 256, "NUM_KSPLIT": 8, "num_warps": 8,
                   "num_stages": 2, "waves_per_eu": 2},
    "M_LEQ_2048": {"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_C": 64, "NUM_KSPLIT": 8, "num_warps": 8,
                   "num_stages": 2, "waves_per_eu": 1},
    "M_LEQ_4096": {"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_C": 64, "NUM_KSPLIT": 4, "num_warps": 8,
                   "num_stages": 2, "waves_per_eu": 1},
    "M_LEQ_8192": {"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_C": 256, "NUM_KSPLIT": 2, "num_warps": 8,
                   "num_stages": 2, "waves_per_eu": 0},
    "M_LEQ_16384": {"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_C": 256, "NUM_KSPLIT": 2, "num_warps": 4,
                    "num_stages": 1, "waves_per_eu": 0},
    "any": {"BLOCK_M": 64, "BLOCK_K": 256, "BLOCK_C": 256, "NUM_KSPLIT": 2, "num_warps": 4,
            "num_stages": 1, "waves_per_eu": 0},
}


@functools.lru_cache(maxsize=1024)
def _get_mhc_config_items(M: int) -> tuple:
    # The M selection of aiter get_mhc_config, unchanged.
    config_dict = _MHC_FUSED_SINKHORN_GFX950_C4096

    # Extract M_LEQ_x keys and their thresholds, sorted ascending
    m_leq_keys = []
    for key in config_dict:
        if key.startswith("M_LEQ_"):
            try:
                threshold = int(key[6:])  # Extract number after "M_LEQ_"
                m_leq_keys.append((threshold, key))
            except ValueError:
                continue
    m_leq_keys.sort()  # Sort by threshold value

    # Find largest threshold <= M
    matched_key = None
    for threshold, key in m_leq_keys:
        if M >= threshold:
            matched_key = key
        else:
            break

    if matched_key is not None:
        return tuple(config_dict[matched_key].items())

    # Fallback to "any" if no matching key found
    return tuple(config_dict["any"].items())


def get_mhc_config(M: int) -> dict:
    """get_mhc_config("MHC_FUSED", M, 4096, mode="sinkhorn")[0] on gfx950."""
    return dict(_get_mhc_config_items(M))


def mhc_post_pre(
    layer_input: torch.Tensor,  # (M, C)        bf16 / fp16
    residual_in: torch.Tensor,  # (M, n, C)     bf16 / fp16
    post_mix: torch.Tensor,  # (M, n) or (M, n, 1) fp32
    comb_mix: torch.Tensor,  # (M, n, n)     fp32
    phi: torch.Tensor,  # (n*C, N=n+n+n*n) bf16 / fp16
    alphas: torch.Tensor,  # (3,) fp32 — [alpha_pre, alpha_post, alpha_res]
    bias: torch.Tensor,  # (N,) fp32
    n: int,
    eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_iters: int = 20,
    asymmetric_exp_domain: bool = False,
    hc_sinkhorn_eps: float = 1e-6,
    residual_out: torch.Tensor | None = None,
    h_post: torch.Tensor | None = None,
    h_res: torch.Tensor | None = None,
    layer_input_out: torch.Tensor | None = None,
    acc_partial: torch.Tensor | None = None,
    acc_sq_partial: torch.Tensor | None = None,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mhc_post + (next) mhc_pre across two Triton launches.

    Launch 1 (``_mhc_post_pre_split_kernel``): per (M-tile, C-tile), one CTA
    does the mhc_post elementwise mix AND contributes one split-K K-slice of
    the next mhc_pre GEMM + sqrsum, sharing the post-output residual tile in
    registers between the two operations.

    Launch 2 (``_mhc_reduce_apply_kernel``, reused as-is): finalizes RMS-norm
    + bias + alpha + sigmoid (pre/post) + log-domain Sinkhorn (res) + the
    apply-pre fold ``layer_input_out[m, c] = sum_i pre_mix[m, i] *
    residual_out[m, i, c]``.

    The C-tile axis of the split kernel IS the pre's split-K axis (one CTA
    per K-split chunk of size ``n * BLOCK_C`` in the flattened pre input).

    Pipeline:
        residual_out[m, j, c] = post_mix[m, j] * layer_input[m, c]
                              + sum_h comb_mix[m, h, j] * residual_in[m, h, c]
        h_post                = hc_post_mult_value * sigmoid(...)        # of next pre
        h_res                 = Sinkhorn(...)                            # of next pre
        layer_input_out[m, c] = sum_i (sigmoid(H^pre[m, i]) + hc_pre_eps)
                                    * residual_out[m, i, c]

    Args:
        layer_input:        (M, C) bf16 / fp16, attn/ffn output (the post step's ``x``)
        residual_in:        (M, n, C) bf16 / fp16, prev-layer multi-stream residual
        post_mix:           (M, n) or (M, n, 1) fp32, from preceding mhc_pre
        comb_mix:           (M, n, n) fp32, from preceding mhc_pre; [src, dst]
        phi:                (n*C, N) bf16 / fp16, next pre's projection matrix
        alphas:             (3,) fp32 — ``[alpha_pre, alpha_post, alpha_res]``.
                            Loaded by the kernel via tl.load; pass the
                            ``hc_scale`` parameter tensor unchanged.
        bias:               (N,) fp32, next pre's bias
        n:                  stream / manifold dimension
        eps:                RMS-norm epsilon (default 1e-6)
        hc_pre_eps:         added to sigmoid(H^pre) (default 0.0)
        hc_post_mult_value: multiplier on sigmoid(H^post) (default 2.0)
        sinkhorn_iters:     SK iterations on next pre's H^res (default 20).
        asymmetric_exp_domain: when True, runs HIP-compatible exp-domain
                            Sinkhorn (``mhc_kernels.cu:493-507``): first iter
                            ``softmax(row) + eps`` then ``div(col + eps)``,
                            followed by ``sinkhorn_iters - 1`` symmetric
                            ``div(row+eps), div(col+eps)`` iters. Default
                            False (canonical log-domain Sinkhorn-Knopp).
        hc_sinkhorn_eps:    eps used during asymmetric-exp-domain Sinkhorn;
                            ignored when ``asymmetric_exp_domain`` is False
                            (default 1e-6).
        residual_out:       optional pre-allocated (M, n, C) output of dtype
                            ``layer_input.dtype``. Allocated if None.
        h_post:             optional pre-allocated post-stream output. Accepts
                            either (M, n) or (M, n, 1). Any float dtype — the
                            kernel implicit-casts on store. Allocated as
                            (M, n, 1) in ``layer_input.dtype`` if None.
        h_res:              optional pre-allocated res-stream output of shape
                            (M, n, n). Any float dtype. Allocated as
                            (M, n, n) in ``layer_input.dtype`` if None.
        layer_input_out:    optional pre-allocated (M, C) output of dtype
                            ``layer_input.dtype``. Allocated if None.
        config:             optional config dict; loaded from
                            get_mhc_config("MHC_FUSED", M, C, mode="sinkhorn")
                            when None. NUM_KSPLIT is ignored — the split kernel's
                            grid uses ``cdiv(C, BLOCK_C)`` K-splits.

    Returns ``(h_post, h_res, layer_input_out, residual_out)``:
        h_post:          (M, n, 1)  hc_post_mult_value * sigmoid(H^post)
        h_res:           (M, n, n)  Sinkhorn output (raw logits if sk_iters == 0)
        layer_input_out: (M, C)     sum_i (sigmoid(H^pre[m, i]) + hc_pre_eps)
                                          * residual_out[m, i, c]
        residual_out:    (M, n, C)  new mHC residual stream (must be kept;
                                    consumed by the next layer's hc_post)

    Reference: arXiv:2512.24880.
    """
    M, C = layer_input.shape
    M_res, n_res, C_res = residual_in.shape
    assert (M, n, C) == (
        M_res,
        n_res,
        C_res,
    ), f"shape mismatch: layer_input=({M},{C}), residual_in=({M_res},{n_res},{C_res}), n={n}"

    K_phi, N_phi = phi.shape
    n_squared = n * n
    N_total = 2 * n + n_squared
    K = n * C
    assert K_phi == K, f"phi K mismatch: expected {K}, got {K_phi}"
    assert N_phi == N_total, f"phi N mismatch: expected {N_total}, got {N_phi}"

    if post_mix.ndim == 3:
        post_mix = post_mix.squeeze(-1)
    assert post_mix.shape == (
        M,
        n,
    ), f"post_mix shape mismatch: expected ({M}, {n}), got {tuple(post_mix.shape)}"
    assert comb_mix.shape == (
        M,
        n,
        n,
    ), f"comb_mix shape mismatch: expected ({M}, {n}, {n}), got {tuple(comb_mix.shape)}"
    assert (
        bias.shape[0] == N_total
    ), f"bias shape mismatch: expected ({N_total},), got {tuple(bias.shape)}"
    assert (
        layer_input.device
        == residual_in.device
        == phi.device
        == bias.device
        == post_mix.device
        == comb_mix.device
    ), "All tensors must be on the same device"
    assert layer_input.device.type == "cuda", "mHC kernel requires CUDA device"
    assert sinkhorn_iters >= 0, f"sinkhorn_iters must be >= 0, got {sinkhorn_iters}"

    device = layer_input.device
    dtype = layer_input.dtype

    N_POW2 = triton.next_power_of_2(n)
    N_POW2_RES = triton.next_power_of_2(n_squared)
    N_TOTAL_POW2 = triton.next_power_of_2(N_total)

    if sinkhorn_iters > 0:
        assert N_POW2_RES == n_squared, (
            f"sinkhorn_iters>0 requires n_squared ({n_squared}) to be a power of 2; "
            f"got N_POW2_RES={N_POW2_RES}. For non-power-of-2 n, run with "
            f"sinkhorn_iters=0."
        )

    # Reuse MHC_FUSED tuning. Two block-sizes serve distinct roles here:
    #   - BLOCK_K  : per-CTA K-axis width in the post_pre split kernel; one
    #                CTA covers BLOCK_K K-elements of the next-pre GEMM,
    #                equivalent to BLOCK_K/n contiguous C-elements per stream.
    #                NUM_KSPLIT = cdiv(K, BLOCK_K).
    #   - BLOCK_C  : C-axis tile for _mhc_reduce_apply_kernel only (apply-pre
    #                output parallelism). Independent of BLOCK_K.
    if config is None:
        config = get_mhc_config(M)
    config = dict(config)
    BLOCK_M = config.pop("BLOCK_M", 64 if M >= 64 else 32)
    BLOCK_K = config.pop("BLOCK_K", min(512, triton.next_power_of_2(K)))
    BLOCK_K = min(BLOCK_K, triton.next_power_of_2(K))
    BLOCK_C = config.pop("BLOCK_C", min(128, triton.next_power_of_2(C)))
    BLOCK_C = min(BLOCK_C, triton.next_power_of_2(C))
    # Knobs irrelevant to mhc_post_pre.
    config.pop("BLOCK_N", None)
    config.pop("NUM_KSPLIT", None)

    assert BLOCK_K % n == 0, (
        f"BLOCK_K ({BLOCK_K}) must be divisible by n ({n}) — each CTA owns "
        f"n contiguous BLOCK_K/n-wide C-slices of the next-pre GEMM"
    )
    # Per-CTA C-slice width (constexpr inside the split kernel).
    BLOCK_C_SPLIT = BLOCK_K // n
    NUM_KSPLIT = triton.cdiv(C, BLOCK_C_SPLIT)  # == cdiv(K, BLOCK_K)
    KSPLIT_POW2 = triton.next_power_of_2(NUM_KSPLIT)

    # `_mhc_reduce_apply_kernel` uses a (num_M_tiles, NUM_C_BLOCKS + 2) grid.
    NUM_C_BLOCKS = triton.cdiv(C, BLOCK_C)

    assert alphas.shape == (
        3,
    ), f"alphas shape mismatch: expected (3,), got {tuple(alphas.shape)}"

    if residual_out is None:
        residual_out = torch.empty(M, n, C, dtype=dtype, device=device)
    else:
        assert residual_out.shape == (
            M,
            n,
            C,
        ), f"residual_out shape mismatch: expected ({M}, {n}, {C}), got {tuple(residual_out.shape)}"
        assert (
            residual_out.dtype == dtype
        ), f"residual_out dtype mismatch: expected {dtype}, got {residual_out.dtype}"

    # Split-K reduce-apply scratch. Write-before-read within this call (the
    # split kernel writes every entry the reduce kernel reads, masked the same
    # way), so callers may pass persistent buffers — keeping `.data_ptr()`
    # stable lets CUDAGraph capture/replay work without surprise re-allocation.
    if acc_partial is None:
        acc_partial = torch.empty(
            (NUM_KSPLIT, M, N_total), dtype=torch.float32, device=device
        )
    else:
        assert acc_partial.shape == (NUM_KSPLIT, M, N_total), (
            f"acc_partial shape mismatch: expected ({NUM_KSPLIT}, {M}, {N_total}), "
            f"got {tuple(acc_partial.shape)}"
        )
        assert (
            acc_partial.dtype == torch.float32
        ), f"acc_partial dtype mismatch: expected float32, got {acc_partial.dtype}"
    if acc_sq_partial is None:
        acc_sq_partial = torch.empty(
            (NUM_KSPLIT, M), dtype=torch.float32, device=device
        )
    else:
        assert acc_sq_partial.shape == (NUM_KSPLIT, M), (
            f"acc_sq_partial shape mismatch: expected ({NUM_KSPLIT}, {M}), "
            f"got {tuple(acc_sq_partial.shape)}"
        )
        assert acc_sq_partial.dtype == torch.float32, (
            f"acc_sq_partial dtype mismatch: expected float32, "
            f"got {acc_sq_partial.dtype}"
        )

    # --- Launch 1: fused post + partial pre GEMM/sqrsum, one CTA per (M-tile, C-tile).
    grid_split = (triton.cdiv(M, BLOCK_M), NUM_KSPLIT)
    _mhc_post_pre_split_kernel[grid_split](
        layer_input,
        residual_in,
        post_mix,
        comb_mix,
        residual_out,
        phi,
        acc_partial,
        acc_sq_partial,
        M=M,
        N=N_total,
        n=n,
        C=C,
        stride_x_m=layer_input.stride(0),
        stride_x_c=layer_input.stride(1),
        stride_resin_m=residual_in.stride(0),
        stride_resin_n=residual_in.stride(1),
        stride_resin_c=residual_in.stride(2),
        stride_post_m=post_mix.stride(0),
        stride_post_n=post_mix.stride(1),
        stride_comb_m=comb_mix.stride(0),
        stride_comb_src=comb_mix.stride(1),
        stride_comb_dst=comb_mix.stride(2),
        stride_resout_m=residual_out.stride(0),
        stride_resout_n=residual_out.stride(1),
        stride_resout_c=residual_out.stride(2),
        stride_phi_k=phi.stride(0),
        stride_phi_n=phi.stride(1),
        stride_acc_k=acc_partial.stride(0),
        stride_acc_m=acc_partial.stride(1),
        stride_acc_n=acc_partial.stride(2),
        stride_acc_sq_k=acc_sq_partial.stride(0),
        stride_acc_sq_m=acc_sq_partial.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C_SPLIT,
        N_TOTAL_POW2=N_TOTAL_POW2,
        **config,
    )

    # --- Launch 2: reduce-apply kernel writes h_post and h_res directly.
    # Allocate any caller-omitted outputs in `layer_input.dtype`. Callers can
    # pre-allocate them in any float dtype (e.g. fp32) to skip a downstream
    # cast — the kernel implicit-casts at the store.
    if h_post is None:
        h_post = torch.empty(M, n, 1, dtype=dtype, device=device)
    else:
        if h_post.ndim == 2:
            assert h_post.shape == (
                M,
                n,
            ), f"h_post 2D shape mismatch: expected ({M}, {n}), got {tuple(h_post.shape)}"
        else:
            assert h_post.shape == (
                M,
                n,
                1,
            ), f"h_post 3D shape mismatch: expected ({M}, {n}, 1), got {tuple(h_post.shape)}"
    if h_res is None:
        h_res = torch.empty(M, n, n, dtype=dtype, device=device)
    else:
        assert h_res.shape == (
            M,
            n,
            n,
        ), f"h_res shape mismatch: expected ({M}, {n}, {n}), got {tuple(h_res.shape)}"
    if layer_input_out is None:
        layer_input_out = torch.empty(M, C, dtype=dtype, device=device)
    else:
        assert layer_input_out.shape == (M, C), (
            f"layer_input_out shape mismatch: expected ({M}, {C}), "
            f"got {tuple(layer_input_out.shape)}"
        )

    # 2D (M, n) view for the kernel — h_post.ndim is 2 or 3, both OK after view.
    h_post_2d = h_post.view(M, n)
    # 2D (M, n*n) view for the kernel — h_res is contiguous (M, n, n).
    h_res_2d = h_res.view(M, n_squared)

    # residual_out (M, n, C) is contiguous after `torch.empty`; flatten gives
    # the (M, n*C) layout the reduce-apply kernel expects as ``x``.
    x_flat = residual_out.view(M, n * C)

    # Grid: NUM_C_BLOCKS apply-pre CTAs + 2 dedicated CTAs for post & res.
    BLOCK_M_POST_RES = 1
    grid_reduce_apply = (
        triton.cdiv(M, BLOCK_M) * NUM_C_BLOCKS + triton.cdiv(M, BLOCK_M_POST_RES) * 2,
    )
    _mhc_post_pre_reduce_apply_kernel[grid_reduce_apply](
        acc_partial,
        acc_sq_partial,
        alphas,
        bias,
        x_flat,
        h_post_2d,
        h_res_2d,
        layer_input_out,
        M=M,
        K=K,
        n=n,
        n_squared=n_squared,
        C=C,
        eps=eps,
        hc_pre_eps=hc_pre_eps,
        hc_post_mult_value=hc_post_mult_value,
        stride_acc_k=acc_partial.stride(0),
        stride_acc_m=acc_partial.stride(1),
        stride_acc_n=acc_partial.stride(2),
        stride_acc_sq_k=acc_sq_partial.stride(0),
        stride_acc_sq_m=acc_sq_partial.stride(1),
        stride_xm=x_flat.stride(0),
        stride_xk=x_flat.stride(1),
        stride_hp_m=h_post_2d.stride(0),
        stride_hp_n=h_post_2d.stride(1),
        stride_hr_m=h_res_2d.stride(0),
        stride_hr_n=h_res_2d.stride(1),
        stride_li_m=layer_input_out.stride(0),
        stride_li_c=layer_input_out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        N_POW2=N_POW2,
        N_POW2_RES=N_POW2_RES,
        ACTUAL_KSPLIT=NUM_KSPLIT,
        KSPLIT_POW2=KSPLIT_POW2,
        BLOCK_M_POST_RES=BLOCK_M_POST_RES,
        NUM_SINKHORN_ITERS=sinkhorn_iters,
        ASYMMETRIC_EXP_DOMAIN=asymmetric_exp_domain,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        **config,
    )

    # Reshape h_post to (M, n, 1) on return for compatibility with downstream
    # callers that expect the trailing singleton (e.g. HIP mhc_post).
    if h_post.ndim == 2:
        h_post = h_post.unsqueeze(-1)
    return h_post, h_res, layer_input_out, residual_out


# aiter/ops/triton/normalization/rmsnorm.py launch helpers. get_num_sms (AITER's CU count) falls
# back to the device's multiprocessor count, as aiter/ops/triton/utils/device_info.py does.
@functools.lru_cache(maxsize=None)
def get_num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def num_programs(x):
    return min(x.shape[0], get_num_sms(x.device.index))


def block_size(x):
    return min(65536 // x.element_size(), triton.next_power_of_2(x.shape[1]))


def use_blocked(x):
    return x.shape[1] > block_size(x)


def _rmsnorm_forward(y: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, epsilon: float):
    """aiter _rmsnorm_forward, writing into the preallocated ``y``.

    rms_norm / rmsnorm_forward_inference take this path for N=4096 (their large-M, small-N kernel
    is only used for N <= 2048).
    """
    n_rows, n_cols = x.shape

    rsigma = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    blk_size = block_size(x)
    USE_BLOCKED = use_blocked(x)
    NUM_PRGMS = num_programs(x)

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _rms_norm_kernel[grid](
        x,
        y,
        weight,
        rsigma,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        epsilon,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )

    return y, rsigma


def run(*, x, residual, post_mix, comb_mix, proj_weight, mix_scale, mix_bias, norm_weight,
        rms_eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters, norm_eps,
        next_post_mix, next_comb_mix, layer_input, next_residual) -> None:
    tokens, hidden_size = x.shape
    hc_mult = residual.shape[1]
    # The un-normalized layer input stays in float32 (mhc_post_pre casts at its store) so that
    # layer_input is a single bfloat16 rounding of the normalized value.
    pre_norm = torch.empty((tokens, hidden_size), dtype=torch.float32, device=x.device)
    mhc_post_pre(
        x,
        residual,
        post_mix,
        comb_mix,
        proj_weight.t(),  # (hc_mult * hidden_size, mixes) view of the (mixes, K) float32 weight
        mix_scale,
        mix_bias,
        hc_mult,
        eps=rms_eps,
        hc_pre_eps=pre_eps,
        hc_post_mult_value=post_multiplier,
        sinkhorn_iters=sinkhorn_iters,
        asymmetric_exp_domain=True,  # softmax + eps, then row/column Sinkhorn, as the operator does
        hc_sinkhorn_eps=sinkhorn_eps,
        residual_out=next_residual,
        h_post=next_post_mix,
        h_res=next_comb_mix.view(tokens, hc_mult, hc_mult),  # the launcher takes (M, n, n)
        layer_input_out=pre_norm,
    )
    _rmsnorm_forward(layer_input, pre_norm, norm_weight, norm_eps)
