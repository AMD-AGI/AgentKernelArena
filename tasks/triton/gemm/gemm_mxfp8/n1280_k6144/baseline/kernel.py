# SPDX-License-Identifier: Apache-2.0
"""Dense MXFP8 GEMM via Triton ``tl.dot_scaled`` for AMD CDNA4 (gfx950).

Vendored from SGLang (sglang/kernels/ops/quantization/mxfp8_amd_gfx95.py): the kernel `_mxfp8_linear_kernel`, which
consumes FP8 E4M3 operands and E8M0 block scales directly (no dequantization to BF16, lowering to the CDNA4 native
MX matrix-core ops), and the tile selection of its launcher `_run_mxfp8_linear_kernel`. The activation quantizer and
the dequantize fallback of the original file are not part of this baseline.
"""
import triton
import triton.language as tl


@triton.jit
def _mxfp8_linear_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        xs_ptrs += (BLOCK_K // 32) * stride_xsk
        ws_ptrs += (BLOCK_K // 32) * stride_wsk

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def run(*, x, x_scale, w, w_scale, out):
    """out[M, N] = (x * x_scale) @ (w * w_scale)^T, written into the preallocated bf16 output.

    x [M, K] fp8 e4m3, x_scale [M, K//32] uint8 (E8M0), w [N, K] fp8 e4m3, w_scale [N, K//32] uint8 (E8M0).
    Launcher body of SGLang `_run_mxfp8_linear_kernel` (tile selection kept as is).
    """
    M, K = x.shape
    N = w.shape[0]
    BLOCK_M, BLOCK_K = 64, 128
    if M <= 512 and (K >= 4096 or (N == 6144 and K in (2048, 3072))):
        BLOCK_N, num_warps = 64, 4
    else:
        BLOCK_N, num_warps = 128, 8
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mxfp8_linear_kernel[grid](
        x,
        x_scale,
        w,
        w_scale,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w.stride(0),
        w.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
