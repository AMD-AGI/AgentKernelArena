// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "../scripts/native/matrix_launch.hpp"
#include "kernel.hpp"

hipError_t aka_matrix_launch(const float* a, const float* b, float* c,
                            unsigned int a_cols, dim3 grid, dim3 block,
                            hipStream_t stream)
{
    matrix_multiplication_kernel<16><<<grid, block, 0, stream>>>(a, b, c, a_cols);
    return hipGetLastError();
}
