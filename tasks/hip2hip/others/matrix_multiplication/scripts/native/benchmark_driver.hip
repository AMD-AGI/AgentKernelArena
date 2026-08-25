// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "hip_graph_benchmark.hpp"

// Compile the candidate kernel into this protected benchmark translation unit.
// Renaming its CLI entry point keeps setup/validation code out of timed work.
#define main aka_matrix_candidate_main
#include "../../main.hip"
#undef main

#include <cstdlib>
#include <cmath>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

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
        std::vector<float> h_a(static_cast<std::size_t>(a_rows) * a_cols);
        std::vector<float> h_b(static_cast<std::size_t>(a_cols) * b_cols);
        std::vector<float> h_c(static_cast<std::size_t>(a_rows) * b_cols);
        for(unsigned int row = 0; row < a_rows; ++row)
        {
            for(unsigned int k = 0; k < a_cols; ++k)
            {
                const int value = static_cast<int>((row * 17U + k * 13U) % 23U) - 11;
                h_a[static_cast<std::size_t>(row) * a_cols + k]
                    = static_cast<float>(value) * 0.03125f;
            }
        }
        for(unsigned int k = 0; k < a_cols; ++k)
        {
            for(unsigned int col = 0; col < b_cols; ++col)
            {
                const int value = static_cast<int>((k * 19U + col * 7U) % 29U) - 14;
                h_b[static_cast<std::size_t>(k) * b_cols + col]
                    = static_cast<float>(value) * 0.015625f;
            }
        }

        aka_native_perf::throw_on_hip_error(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_a, a_bytes), "hipMalloc(A)");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_b, b_bytes), "hipMalloc(B)");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_c, c_bytes), "hipMalloc(C)");
        aka_native_perf::throw_on_hip_error(
            hipMemcpyAsync(d_a, h_a.data(), a_bytes, hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync(A)");
        aka_native_perf::throw_on_hip_error(
            hipMemcpyAsync(d_b, h_b.data(), b_bytes, hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync(B)");
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

        auto validate = [&](hipGraphExec_t graph_exec,
                            hipStream_t validation_stream) -> std::string
        {
            hipError_t error = hipMemsetAsync(d_c, 0xFF, c_bytes, validation_stream);
            if(error == hipSuccess)
            {
                error = graph_exec != nullptr
                    ? hipGraphLaunch(graph_exec, validation_stream)
                    : launch(validation_stream);
            }
            if(error == hipSuccess)
            {
                error = hipMemcpyAsync(h_c.data(), d_c, c_bytes,
                                       hipMemcpyDeviceToHost, validation_stream);
            }
            if(error == hipSuccess)
            {
                error = hipStreamSynchronize(validation_stream);
            }
            if(error != hipSuccess)
            {
                return aka_native_perf::hip_failure("matrix replay validation", error);
            }

            const unsigned int rows[] = {0U, a_rows / 3U, a_rows / 2U, a_rows - 1U};
            const unsigned int cols[] = {0U, b_cols / 3U, b_cols / 2U, b_cols - 1U};
            for(const unsigned int row : rows)
            {
                for(const unsigned int col : cols)
                {
                    double expected = 0.0;
                    for(unsigned int k = 0; k < a_cols; ++k)
                    {
                        expected += static_cast<double>(
                            h_a[static_cast<std::size_t>(row) * a_cols + k])
                            * static_cast<double>(
                                h_b[static_cast<std::size_t>(k) * b_cols + col]);
                    }
                    const float actual
                        = h_c[static_cast<std::size_t>(row) * b_cols + col];
                    const double tolerance
                        = 2.0e-3 * std::max(1.0, std::fabs(expected));
                    if(!std::isfinite(actual)
                       || std::fabs(static_cast<double>(actual) - expected) > tolerance)
                    {
                        std::ostringstream message;
                        message << "C[" << row << "," << col << "]=" << actual
                                << ", expected=" << expected;
                        return message.str();
                    }
                }
            }
            return {};
        };

        const auto result = aka_native_perf::benchmark_graph_or_events(
            launch, stream, /*warmup=*/10, samples, /*target_ms=*/1.0,
            /*max_repeats=*/1000, validate);
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
