# SPDX-License-Identifier: BSD-3-Clause
# Triton rewrite of the kernel TorchInductor generates for x / sqrt(mean(x ** 2, dim=1, keepdim=True) + eps).
"""RMS normalization across the channel axis of an NCHW tensor, in Triton."""
import triton
import triton.language as tl


@triton.jit
def _rms_norm_channel_kernel(x_ptr, out_ptr, n_pixels, eps, C: tl.constexpr, BLOCK: tl.constexpr):
    # Program (p, b) normalizes pixels [p * BLOCK, (p + 1) * BLOCK) of sample b. The channel stride is
    # n_pixels (H * W), so a [C, BLOCK] tile is C contiguous runs of BLOCK elements.
    pid_b = tl.program_id(1).to(tl.int64)
    pix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pix < n_pixels
    chan = tl.arange(0, C).to(tl.int64)
    offsets = pid_b * C * n_pixels + chan[:, None] * n_pixels + pix[None, :]
    x = tl.load(x_ptr + offsets, mask=mask[None, :], other=0.0)
    mean_square = tl.sum(x * x, axis=0) / C
    rms = tl.sqrt_rn(mean_square + eps)
    tl.store(out_ptr + offsets, x / rms[None, :], mask=mask[None, :])


def run(*, x, epsilon, out):
    batch, channels, height, width = x.shape
    n_pixels = height * width
    BLOCK = 128
    grid = (triton.cdiv(n_pixels, BLOCK), batch)
    _rms_norm_channel_kernel[grid](x, out, n_pixels, epsilon, C=channels, BLOCK=BLOCK, num_warps=4)
