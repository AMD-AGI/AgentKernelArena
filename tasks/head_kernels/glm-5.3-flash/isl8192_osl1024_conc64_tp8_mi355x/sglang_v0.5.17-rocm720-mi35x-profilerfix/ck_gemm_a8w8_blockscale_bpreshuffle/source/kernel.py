# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Stock function body from AITER d9e5ef7ce08ee7045d583aed768cff41aa9210fe.
# The protected harness retains native backend dependencies and the fixed ABI.
import torch
from torch import Tensor
from aiter.ops.gemm_op_a8w8 import (
    AITER_CONFIGS, dtypes, get_gfx, get_CKGEMM_config, logger,
    is_flydsl_available, _hip_blockscale_supported,
    gemm_a8w8_blockscale_bpreshuffle_flydsl,
    gemm_a8w8_blockscale_bpreshuffle_cktile,
    gemm_a8w8_blockscale_bpreshuffle_ck,
    gemm_a8w8_blockscale_bpreshuffle_asm,
)


def gemm_a8w8_blockscale_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[1]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)

    use_gfx1250_flydsl_blockscale = (
        get_gfx() == "gfx1250"
        and x_scale.dtype == dtypes.fp8_e8m0
        and w_scale.dtype == dtypes.fp8_e8m0
    )
    if use_gfx1250_flydsl_blockscale:
        if not is_flydsl_available():
            raise RuntimeError(
                "gfx1250 a8w8 blockscale bpreshuffle with fp8_e8m0 scales "
                "requires FlyDSL"
            )
        config = get_CKGEMM_config(
            m,
            n,
            k,
            AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE,
        )
        if config is not None and config["libtype"] == "flydsl":
            return gemm_a8w8_blockscale_bpreshuffle_flydsl(
                XQ, WQ, x_scale, w_scale, Y, config
            )

        from ..ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_bpreshuffle_wmma_common import (
            kernel_fits_shape,
            kernels_list,
        )

        fits = [ki for ki in kernels_list.values() if kernel_fits_shape(ki, m, n, k)]
        if fits:
            want_tm = min(256, max(16, 1 << (m - 1).bit_length()))
            ki = min(
                fits, key=lambda x: (abs(x.tile_m - want_tm), -x.tile_n, -x.tile_k)
            )
            logger.warning(
                f"[gfx1250] gemm_a8w8_blockscale_bpreshuffle untuned "
                f"M={m}, N={n}, K={k}; falling back to flydsl kernel '{ki.name}'."
            )
            return gemm_a8w8_blockscale_bpreshuffle_flydsl(
                XQ, WQ, x_scale, w_scale, Y, {"kernelName": ki.name}
            )

    # temporarily guard scale that are not fp32.
    if x_scale.dtype == dtypes.fp8_e8m0:
        x_scale = x_scale.to(dtypes.fp32)
    if w_scale.dtype == dtypes.fp8_e8m0:
        w_scale = w_scale.to(dtypes.fp32)

    if not _hip_blockscale_supported():
        # No CK code object for this arch -> triton preshuffle. WQ is already
        # (16,16)-shuffled (the only blockscale layout) == triton's (N//16, K*16)
        # view; x_scale is column-major. Direct fit.
        from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale_preshuffle as _gemm_a8w8_blockscale_preshuffle_triton,
        )

        xq = XQ if XQ.dtype != torch.uint8 else XQ.view(dtypes.fp8)
        wq = WQ if WQ.dtype != torch.uint8 else WQ.view(dtypes.fp8)
        # Explicit config (no PRESHUFFLED tuning file on main yet); mirrors the
        # gfx1201 non-preshuffle M_LEQ_8 default.
        _fallback_cfg = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 8,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 1,
            "kpack": 2,
        }
        return _gemm_a8w8_blockscale_preshuffle_triton(
            xq,
            wq.reshape(n // 16, k * 16),
            x_scale,
            w_scale,
            dtype=dtype,
            config=_fallback_cfg,
            is_x_scale_tranposed=x_scale.stride(0) != 1,
        )
    config = get_CKGEMM_config(
        m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    )
    # Triton path first: it allocates its own output, so skip the Y buffer the
    # ck/asm paths below need.
    if (config is not None and config["libtype"] == "triton") or get_gfx() == "gfx1250":
        # kernelName optionally carries the backend hint ("triton"/"gluon");
        # anything else -> None (auto gluon->triton detection). config=None lets
        # the triton impl load its own tuned config internally. WQ is already
        # (16,16)-shuffled == triton's (N//16, K*16) view; x_scale is
        # column-major -> direct fit.
        from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale_preshuffle as _gemm_a8w8_blockscale_preshuffle_triton,
        )

        kernelName = str(config.get("kernelName", "")) if config is not None else ""
        backend = kernelName if kernelName in ("triton", "gluon") else None
        xq = XQ if XQ.dtype != torch.uint8 else XQ.view(dtypes.fp8)
        wq = WQ if WQ.dtype != torch.uint8 else WQ.view(dtypes.fp8)
        return _gemm_a8w8_blockscale_preshuffle_triton(
            xq,
            wq.reshape(n // 16, k * 16),
            x_scale,
            w_scale,
            dtype=dtype,
            backend=backend,
            is_x_scale_tranposed=x_scale.stride(0) != 1,
        )
    if config is not None:
        libtype = config["libtype"]
        kernelName = str(config.get("kernelName", ""))
        if libtype == "cktile":
            return gemm_a8w8_blockscale_bpreshuffle_cktile(
                XQ, WQ, x_scale, w_scale, Y, kernelName=kernelName
            )
        elif libtype == "ck":
            return gemm_a8w8_blockscale_bpreshuffle_ck(
                XQ, WQ, x_scale, w_scale, Y, kernelName=kernelName
            )
        elif libtype == "asm":
            splitK = config["splitK"]
            return gemm_a8w8_blockscale_bpreshuffle_asm(
                XQ, WQ, Y, x_scale, w_scale, splitK=splitK, kernelName=kernelName
            )
        elif libtype == "opus":
            kernelId = int(config["kernelId"])
            from aiter.ops.opus.gemm_op_a8w8 import (
                opus_gemm_a8w8_blockscale_bpreshuffle_tune,
            )

            return opus_gemm_a8w8_blockscale_bpreshuffle_tune(
                XQ, WQ, x_scale, w_scale, Y, kernelId=kernelId
            )
        elif libtype == "flydsl" and is_flydsl_available():
            return gemm_a8w8_blockscale_bpreshuffle_flydsl(
                XQ, WQ, x_scale, w_scale, Y, config
            )
    try:
        return gemm_a8w8_blockscale_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y)
    except RuntimeError as e:
        raise RuntimeError(
            f"gemm_a8w8_blockscale_bpreshuffle failed for shape M={m}, N={n}, K={k}, "
            f"{dtype=}, config={config}: {e}"
        ) from e
