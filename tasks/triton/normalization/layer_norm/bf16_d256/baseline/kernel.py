# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-row affine LayerNorm forward, from generative-recommenders' ``triton_layer_norm.py``."""
import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import rsqrt as libdevice_rsqrt
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import rsqrt as libdevice_rsqrt
    except ImportError:
        from triton.language.math import rsqrt as libdevice_rsqrt


def _get_layer_norm_fwd_configs():
    """Autotune configs for the multi-row LayerNorm kernels (the AMD list of the upstream helper)."""
    configs = []
    for BLOCK_N in [1, 2, 4, 8]:
        for num_warps in [1, 2, 4, 8]:
            if BLOCK_N == 1 and num_warps == 8:
                # Upstream skips this single-row/eight-warp candidate on AMD GPUs.
                continue
            configs.append(triton.Config({"BLOCK_N": BLOCK_N}, num_warps=num_warps))
    return configs


@triton.autotune(
    configs=_get_layer_norm_fwd_configs(),
    key=["BLOCK_D", "N"],
)
@triton.jit
def _weighted_layer_norm_fwd(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N,
    D,
    eps,
    stride_x,
    stride_y,
    IS_SWISH: tl.constexpr,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
):
    # Get the block ID and calculate starting row
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    # Load weight and bias once (shared across all rows in this block)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)

    # Create block pointers for X and Y
    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N, D),
        strides=(stride_x, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    Y_block_ptr = tl.make_block_ptr(
        base=Y,
        shape=(N, D),
        strides=(stride_y, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )

    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x_block, axis=1) / D
        if TRAINING:
            tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
    else:
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)

    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)

    if COMPUTE_MEAN_AND_RSTD:
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = libdevice_rsqrt(var + eps)
        if TRAINING:
            tl.store(Rstd + rows, rstd, row_mask)
    else:
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    y = y * w[None, :] + b[None, :]

    if IS_SWISH:
        y = tl.sigmoid(y) * x_block

    tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


def run(*, x, weight, bias, epsilon, out, mean, rstd):
    N, D = x.shape
    BLOCK_D = triton.next_power_of_2(D)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
    _weighted_layer_norm_fwd[grid](
        x,
        out,
        weight,
        bias,
        mean,
        rstd,
        N,
        D,
        epsilon,
        x.stride(0),
        out.stride(0),
        IS_SWISH=False,
        TRAINING=True,
        BLOCK_D=BLOCK_D,
        COMPUTE_MEAN_AND_RSTD=True,
    )
