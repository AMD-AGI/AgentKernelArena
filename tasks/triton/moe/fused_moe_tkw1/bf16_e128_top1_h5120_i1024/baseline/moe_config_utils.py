# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# (ROCm/aiter aiter/ops/triton/utils/moe_config_utils.py: get_optimal_moe_config for unquantized
# weights, with the tuned gfx950 table aiter/ops/triton/configs/moe/gfx950-MOE-DEFAULT.json inlined
# in place of the JSON file lookup. The task requires gfx950.)

M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024

# aiter/ops/triton/configs/moe/gfx950-MOE-DEFAULT.json
GFX950_MOE_DEFAULT = {
    "small_M": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 1,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    },
    "medium_M": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    },
    "large_M": {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    },
}


def get_optimal_moe_config(M: int = 1):
    configs = GFX950_MOE_DEFAULT
    if M < M_THRESHOLD_SMALL:
        config = configs["small_M"]
    elif M < M_THRESHOLD_MEDIUM:
        config = configs["medium_M"]
    else:
        config = configs["large_M"]
    return dict(config)
