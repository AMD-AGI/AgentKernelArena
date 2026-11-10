# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/kernels/dims.py
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/deepseek_v41/gfx950/stages/dims.py). Modified for
# AgentKernelArena: imports are package-relative; Dims is a frozen dataclass instead of
# a msgspec Struct.
"""A TP rank's share of V4.1's MoE widths."""

import dataclasses

from .plan import Shard, padded

# the model's full widths
MOE_INTER = 2304  # a routed expert's intermediate width
SHARED_INTER = 2304
# the loader pads a routed expert's rank share to a multiple of this with zero weights
MOE_PAD = 128
ROWS = 16  # GEMV rows a task
UG_PART = 32  # intermediate columns an ug task part: one FP4 group


@dataclasses.dataclass(frozen=True)
class Dims:
    tp: int

    def __post_init__(self):
        shard = Shard(self.tp)
        shard.split(MOE_INTER, "routed intermediate")
        shard.split(SHARED_INTER, "shared intermediate")

    @property
    def inter_real(self) -> int:
        """A routed expert's real intermediate width on this rank."""
        return MOE_INTER // self.tp

    @property
    def inter(self) -> int:
        """``inter_real`` padded as the loader pads it (the rest zero weights)."""
        return padded(self.inter_real, MOE_PAD)

    @property
    def sh_inter(self) -> int:
        """The shared expert's intermediate width on this rank."""
        return SHARED_INTER // self.tp

    @property
    def shared_tasks(self) -> int:
        return self.sh_inter // ROWS

    @property
    def mid_words(self) -> int:
        """MXFP8 words of a pick's routed intermediate."""
        return self.inter // 4

    @property
    def down_scale_cols(self) -> int:
        """Return ``w2_s``'s E8M0 columns."""
        return padded(self.inter // 32, 8)
