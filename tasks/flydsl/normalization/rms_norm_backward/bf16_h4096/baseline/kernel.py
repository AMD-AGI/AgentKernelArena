# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RMSNorm backward launcher: the staged persistent FlyDSL kernel plus its dweight finalizer.

The program-count selection mirrors _select_rmsnorm_bwd_config in kernels/norm/rmsnorm_kernel.py of the
upstream project for the staged path (token counts of at least 512, hidden size at most 8192).
"""

import functools

import torch

from .rmsnorm_bwd_kernel import build_rmsnorm_bwd_two_stage_module

HIDDEN_SIZE = 4096


@functools.lru_cache(maxsize=None)
def _cu_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=None)
def _launcher(n: int, num_programs: int):
    return build_rmsnorm_bwd_two_stage_module(n, "bf16", num_programs)


def run(*, dy, x, gamma, rsigma, dx, dgamma) -> None:
    m, n = x.shape
    num_cus = _cu_count(x.device.index)
    num_programs = min(m, num_cus if m < 2048 else (3 * num_cus) // 2)
    partial = torch.empty((num_programs * n,), device=x.device, dtype=torch.float32)
    launch = _launcher(n, num_programs)
    launch(x, gamma, dy, rsigma, dx, dgamma, partial, m, stream=torch.cuda.current_stream())
