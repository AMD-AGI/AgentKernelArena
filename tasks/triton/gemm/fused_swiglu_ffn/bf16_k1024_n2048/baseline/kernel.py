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

"""Fused SwiGLU forward (Triton), vendored from generative-recommenders.

Provenance: Meta generative-recommenders,
generative_recommenders/ops/triton/triton_swiglu.py
(https://github.com/meta-recsys/generative-recommenders).

STANDALONE, FORWARD-only extraction depending only on `triton`/`torch`.
Only the portable fused SwiGLU forward kernel is ported:
`_swiglu_fwd_kernel` + the host wrapper `triton_swiglu_fwd`. The kernel computes
`out = silu(x @ W_gate^T) * (x @ W_up^T)` in a single launch, loading `x` from
HBM once and reusing it for both GEMMs (the fusion benefit); the activation is
computed in fp32 registers with no HBM round-trip.

The upstream vendor-specific warp-specialized persistent path
(`_swiglu_fwd_tma_ws_persistent`, which needs GPU-specific Triton extensions not
available on this target) and the dispatcher that selects it are dropped; this
build always runs the portable path. The
`generative_recommenders.common.triton_autotune` dep is inlined.
"""

from typing import List

import torch

import triton
import triton.language as tl


# --- inlined triton_autotune (generative_recommenders.common fallback) ---
# Thin wrapper over the public triton.autotune decorator.
def triton_autotune(
    configs: List[triton.Config],
    key: List[str],
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
):
    kwargs = {}
    if prune_configs_by is not None:
        kwargs["prune_configs_by"] = prune_configs_by
    if reset_to_zero is not None:
        kwargs["reset_to_zero"] = reset_to_zero
    if restore_value is not None:
        kwargs["restore_value"] = restore_value
    return triton.autotune(configs=configs, key=key, **kwargs)


# =============================================================================
# Portable fused SwiGLU kernel (standard Triton path).
#
# Fuses silu(x @ W_gate^T) * (x @ W_up^T) into a single kernel launch.
# Uses standard Triton pointer arithmetic; portable across GPUs incl. AMD/CDNA.
#
# Key optimization: x is loaded from HBM ONCE and reused for both GEMMs.
# Activation (silu * up) is computed in float32 registers, no HBM round-trip.
#
# Weight layout: [K, N] (the weights are stored pre-transposed; the upstream wrapper
# transposes an nn.Linear [N, K] weight on every call).
# =============================================================================


def _get_swiglu_fwd_configs() -> List[triton.Config]:
    """
    Autotune configs for the portable fused SwiGLU kernel.

    Two float32 accumulators (gate + up) double register pressure vs single
    GEMM, so smaller block sizes are included.
    """
    configs = [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]
    if torch.version.hip:
        hip_num_stages = 2 if triton.__version__ >= "3.2.0" else 0
        configs.extend(
            [
                triton.Config(
                    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                    num_stages=hip_num_stages,
                    num_warps=4,
                ),
                triton.Config(
                    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                    num_stages=hip_num_stages,
                    num_warps=4,
                ),
                triton.Config(
                    {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                    num_stages=hip_num_stages,
                    num_warps=4,
                ),
                triton.Config(
                    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                    num_stages=hip_num_stages,
                    num_warps=4,
                ),
            ]
        )
    return configs


@triton_autotune(
    configs=_get_swiglu_fwd_configs(),
    key=["M_BLOCK", "N", "K"],
)
@triton.jit
def _swiglu_fwd_kernel(
    # Pointers to input/output tensors
    x_ptr,  # [M, K] input activation
    w_gate_ptr,  # [K, N] gate weight (already transposed from [N, K])
    w_up_ptr,  # [K, N] up weight (already transposed from [N, K])
    out_ptr,  # [M, N] output = silu(x @ w_gate) * (x @ w_up)
    # Matrix dimensions
    M,  # rows in x (batch_size * seq_len)
    N,  # output dimension (hidden_dim)
    K,  # input/reduction dimension (input_dim)
    M_BLOCK,  # next_power_of_2(M) for stable autotuning
    # Strides
    stride_xm,
    stride_xk,
    stride_wgk,
    stride_wgn,
    stride_wuk,
    stride_wun,
    stride_om,
    stride_on,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Fused SwiGLU forward: out = silu(x @ W_gate) * (x @ W_up).

    Each thread block computes one [BLOCK_M, BLOCK_N] output tile.
    Two accumulators share the same x tile loads (the fusion benefit).
    """
    # -- Step 1: Compute tile coordinates with grouped ordering (L2 reuse) --
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M

    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    # -- Step 2: Set up pointers for x, w_gate, w_up tiles --
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    # [BLOCK_M, BLOCK_K]
    x_ptrs = (
        x_ptr
        + (pid_m.to(tl.int64) * BLOCK_M + offs_m)[:, None] * stride_xm
        + offs_k[None, :] * stride_xk
    )
    # [BLOCK_K, BLOCK_N]
    wg_ptrs = (
        w_gate_ptr
        + offs_k[:, None] * stride_wgk
        + (pid_n.to(tl.int64) * BLOCK_N + offs_n)[None, :] * stride_wgn
    )

    # [BLOCK_K, BLOCK_N]
    wu_ptrs = (
        w_up_ptr
        + offs_k[:, None] * stride_wuk
        + (pid_n.to(tl.int64) * BLOCK_N + offs_n)[None, :] * stride_wun
    )

    # -- Step 3: K-loop - two GEMMs sharing the same x tile --
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_m & mask_k, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        wg = tl.load(wg_ptrs, mask=mask_k & mask_n, other=0.0)
        wu = tl.load(wu_ptrs, mask=mask_k & mask_n, other=0.0)

        acc_gate += tl.dot(x, wg, allow_tf32=ALLOW_TF32)
        acc_up += tl.dot(x, wu, allow_tf32=ALLOW_TF32)

        x_ptrs += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wgk
        wu_ptrs += BLOCK_K * stride_wuk

    # -- Step 4: Apply SwiGLU activation in registers (no HBM round-trip) --
    gate_activated = acc_gate * tl.sigmoid(acc_gate)  # silu
    result = (gate_activated * acc_up).to(out_ptr.dtype.element_ty)

    # -- Step 5: Store result --
    offs_m = pid_m * BLOCK_M + offs_m
    offs_n = pid_n * BLOCK_N + offs_n
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result, mask=mask_m & mask_n)


def run(*, x, w_gate, w_up, out):
    """out = silu(x @ w_gate) * (x @ w_up) with x [M, K] and w_gate, w_up [K, N]."""
    M, K = x.shape
    N = w_gate.shape[1]
    if M == 0 or N == 0:
        return
    M_BLOCK = triton.next_power_of_2(M)
    grid = lambda meta: (  # noqa E731
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    _swiglu_fwd_kernel[grid](
        x,
        w_gate,
        w_up,
        out,
        M,
        N,
        K,
        M_BLOCK,
        x.stride(0),
        x.stride(1),
        w_gate.stride(0),
        w_gate.stride(1),
        w_up.stride(0),
        w_up.stride(1),
        out.stride(0),
        out.stride(1),
        ALLOW_TF32=False,
    )
