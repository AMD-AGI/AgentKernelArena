// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>

// Defined in a separate translation unit that alone includes candidate code.
hipError_t aka_matrix_launch(const float* a, const float* b, float* c,
                            unsigned int a_cols, dim3 grid, dim3 block,
                            hipStream_t stream);
