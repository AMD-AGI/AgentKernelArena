// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "hip_graph_benchmark.hpp"

// Compile the candidate kernel into this protected benchmark translation unit.
// Renaming its CLI entry point keeps setup/validation code out of timed work.
#define main aka_matrix_candidate_main
#include "../../main.hip"
#undef main

#include <cstdlib>
#include <exception>
#include <string>

namespace
{

unsigned int parse_uint(int argc, char** argv, const char* name, unsigned int fallback)
{
    for(int i = 1; i + 1 < argc; ++i)
    {
        if(std::string(argv[i]) == name)
        {
            return static_cast<unsigned int>(std::strtoul(argv[i + 1], nullptr, 10));
        }
    }
    return fallback;
}

} // namespace

int main(int argc, char** argv)
{
    constexpr unsigned int block_size = 16;
    const unsigned int a_rows = parse_uint(argc, argv, "--A_rows", 2048);
    const unsigned int a_cols = parse_uint(argc, argv, "--A_cols", 1024);
    const unsigned int b_cols = parse_uint(argc, argv, "--B_cols", 1024);
    const unsigned int samples = parse_uint(argc, argv, "--samples", 100);

    if(a_rows == 0 || a_cols == 0 || b_cols == 0 || a_rows % block_size != 0
       || a_cols % block_size != 0 || b_cols % block_size != 0)
    {
        std::cerr << "matrix dimensions must be positive multiples of 16" << std::endl;
        return 2;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    hipStream_t stream = nullptr;
    try
    {
        const std::size_t a_bytes
            = static_cast<std::size_t>(a_rows) * a_cols * sizeof(float);
        const std::size_t b_bytes
            = static_cast<std::size_t>(a_cols) * b_cols * sizeof(float);
        const std::size_t c_bytes
            = static_cast<std::size_t>(a_rows) * b_cols * sizeof(float);

        aka_native_perf::throw_on_hip_error(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_a, a_bytes), "hipMalloc(A)");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_b, b_bytes), "hipMalloc(B)");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_c, c_bytes), "hipMalloc(C)");
        aka_native_perf::throw_on_hip_error(hipMemsetAsync(d_a, 0, a_bytes, stream),
                                             "hipMemsetAsync(A)");
        aka_native_perf::throw_on_hip_error(hipMemsetAsync(d_b, 0, b_bytes, stream),
                                             "hipMemsetAsync(B)");
        aka_native_perf::throw_on_hip_error(hipStreamSynchronize(stream),
                                             "hipStreamSynchronize(setup)");

        const dim3 block(block_size, block_size);
        const dim3 grid(b_cols / block_size, a_rows / block_size);
    auto launch = [&](hipStream_t launch_stream) -> hipError_t
        {
            matrix_multiplication_kernel<block_size>
                <<<grid, block, 0, launch_stream>>>(d_a, d_b, d_c, a_cols);
            return hipGetLastError();
        };

        const auto result = aka_native_perf::benchmark_graph_or_events(
            launch, stream, /*warmup=*/10, samples, /*target_ms=*/1.0, /*max_repeats=*/1000);
        aka_native_perf::print_result_json("matrix_multiplication", result);

        aka_native_perf::throw_on_hip_error(hipFree(d_a), "hipFree(A)");
        d_a = nullptr;
        aka_native_perf::throw_on_hip_error(hipFree(d_b), "hipFree(B)");
        d_b = nullptr;
        aka_native_perf::throw_on_hip_error(hipFree(d_c), "hipFree(C)");
        d_c = nullptr;
        aka_native_perf::throw_on_hip_error(hipStreamDestroy(stream), "hipStreamDestroy");
        stream = nullptr;
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "native benchmark failed: " << error.what() << std::endl;
        if(d_a != nullptr) hipFree(d_a);
        if(d_b != nullptr) hipFree(d_b);
        if(d_c != nullptr) hipFree(d_c);
        if(stream != nullptr) hipStreamDestroy(stream);
        return 1;
    }
}
