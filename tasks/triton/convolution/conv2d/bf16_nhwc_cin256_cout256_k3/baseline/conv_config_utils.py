# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Conv kernel configuration lookup of aiter/ops/triton/utils/conv_config_utils.py, reduced to gfx950.

Upstream loads configs/conv/<arch>-<CONFIG-NAME>.json for the running architecture. The task runs on gfx950 only
(platform_support), so the seven gfx950 files are inlined below with their upstream content, and the lookup keeps the
upstream three-tier walk: literal shape key, then the M_LEQ_<bound> bucket, then "any". Every gfx950 file contains
only the "any" entry.
"""

import copy
import functools

STANDARD_M_BOUNDS: tuple[int, ...] = (
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
)

_GEMM_TILE = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 4,
    "num_stages": 1,
}

# Content of aiter/ops/triton/configs/conv/gfx950-<name>.json
_GFX950_CONFIGS = {
    "CONV-1X1": {"any": dict(_GEMM_TILE)},
    "CONV-3X3-CBLOCKED": {"any": dict(_GEMM_TILE)},
    "CONV-3X3-NHWC": {"any": dict(_GEMM_TILE)},
    "CONV-GENERAL": {"any": dict(_GEMM_TILE)},
    "CONV-WINO-F4X3-GEMM": {"any": dict(_GEMM_TILE)},
    "CONV-WINO-F4X3-INPUT": {"any": {"BLOCK_C": 64, "num_warps": 4, "num_stages": 1}},
    "CONV-WINO-F4X3-OUTPUT": {"any": {"BLOCK_K": 64, "num_warps": 4, "num_stages": 1}},
}


def format_shape_key(
    N: int,
    C: int,
    H: int,
    W: int,
    K: int,
    R: int,
    S: int,
    sh: int,
    sw: int,
    ph: int,
    pw: int,
    dh: int,
    dw: int,
) -> str:
    """Canonical string key for a user-visible conv2d call. Same format used by
    the loader and the kernel-side _get_config helpers.
    """
    return (
        f"N={N},C={C},H={H},W={W},K={K},R={R},S={S},"
        f"sh={sh},sw={sw},ph={ph},pw={pw},dh={dh},dw={dw}"
    )


@functools.lru_cache(maxsize=512)
def _get_conv_config_cached(
    config_name: str,
    shape_key: str | None,
    M: int | None,
) -> dict:
    """Three-tier walk: literal shape entry -> M_LEQ bucket -> 'any'."""
    config_dict = _GFX950_CONFIGS[config_name]

    # Tier 1: literal shape key.
    shapes = config_dict.get("shapes", {})
    if shape_key is not None and shape_key in shapes:
        return shapes[shape_key]

    # Tier 2: M-bucket walk.
    if M is not None and M >= 0:
        for bound in STANDARD_M_BOUNDS:
            key = f"M_LEQ_{bound}"
            if M <= bound and key in config_dict:
                return config_dict[key]

    # Tier 3: any fallback.
    if "any" in config_dict:
        return config_dict["any"]

    raise KeyError(
        f"No matching config in '{config_name}' for shape_key={shape_key!r}, "
        f"M={M} on arch gfx950 (no literal shape, no bucket, no 'any' fallback)."
    )


def get_conv_config(
    config_name: str,
    shape_key: str | None = None,
    M: int | None = None,
) -> dict:
    """Load a conv kernel config for gfx950.

    Walk order (first hit wins):
        1. ``shapes[shape_key]`` — exact-shape pin from the offline sweep.
        2. ``M_LEQ_<n>`` — row-count bucket walk (M_total for GEMM-like
           kernels, T for Winograd).
        3. ``"any"`` — global fallback.

    Returns a fresh deep-copy of the config dict; safe to mutate.
    """
    config = _get_conv_config_cached(config_name, shape_key, M)
    return copy.deepcopy(config)
