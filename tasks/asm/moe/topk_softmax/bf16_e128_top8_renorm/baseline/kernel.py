# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER's assembly top-k softmax routing for bfloat16 logits, 128 experts, top-8, renormalized.

Python translation of the host-side launcher ``topk_softmax_asm`` in ROCm/aiter
``csrc/py_itfs_cu/asm_topksoftmax.cu``: the kernel choice of
``get_heuristic_kernel_topksoftmax`` over ``hsa/gfx950/topksoftmax/topksoftmax.csv``
(the bf16, 128-expert, top-8 rows), the packed ``KernelArgs`` layout and the grid.
The two code objects are vendored byte for byte from AITER.
"""
import aka_asm

N_EXPERTS = 128
TOPK = 8
RENORMALIZE = 1

# topksoftmax.csv rows for dtype bf16, num_experts 128, topk 8: subm (tokens per workgroup) -> kernel.
_KERNELS = {
    4: aka_asm.load("topksoftmax_4x128x8_bf16.co", "_ZN5aiter24topksoftmax_4x128x8_bf16E"),
    12: aka_asm.load("topksoftmax_12x128x8_bf16.co", "_ZN5aiter25topksoftmax_12x128x8_bf16E"),
}


def _pad(words: int) -> list:
    # KernelArgs is a packed struct: every pointer is followed by a p2 (2 x u32) and every
    # u32 field by a p3 (3 x u32), so each field occupies 16 bytes.
    return [aka_asm.u32(0)] * words


def run(*, gating_output, topk_weights, topk_ids) -> None:
    num_tokens = gating_output.shape[0]
    # MAX_SUBM = num_tokens < 10000 ? 4 : 12; the heuristic picks the largest subm <= MAX_SUBM.
    subm = 4 if num_tokens < 10000 else 12
    kernel = _KERNELS[subm]
    out_stride = topk_weights.stride(0) * 4       # bytes per output row
    args = ([topk_ids] + _pad(2) + [topk_weights] + _pad(2) + [gating_output] + _pad(2)
            + [aka_asm.u32(num_tokens)] + _pad(3)
            + [aka_asm.u32(N_EXPERTS)] + _pad(3)
            + [aka_asm.u32(TOPK)] + _pad(3)
            + [aka_asm.u32(RENORMALIZE)] + _pad(3)
            + [aka_asm.u32(out_stride)] + _pad(3))
    kernel.launch(grid=((num_tokens + subm - 1) // subm, 1, 1), block=(256, 1, 1), args=args,
                  shared_bytes=0)
