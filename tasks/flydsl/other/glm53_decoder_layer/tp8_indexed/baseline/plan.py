# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.: atom/mono/plan/execution.py
# Modified for AgentKernelArena: only the execution-model constants of sgl-amd-mono 52486703
# sgl_amd_mono/common/plan.py (its source-hash build keys use modules the candidate rules forbid).
"""The mono kernels' execution model."""
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64
