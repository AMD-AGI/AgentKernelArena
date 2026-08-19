// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "hip_graph_benchmark.hpp"

#include <rocprim/device/device_search_n.hpp>
#include <rocprim/functional.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace
{

enum class input_mode { normal_half, noise, no_match };

std::size_t parse_size(int argc, char** argv, const char* name, std::size_t fallback)
{
    for(int i = 1; i + 1 < argc; ++i)
    {
        if(std::string(argv[i]) == name)
        {
            return static_cast<std::size_t>(std::strtoull(argv[i + 1], nullptr, 10));
        }
    }
    return fallback;
}

template<class T>
void run_case(const char* id,
              std::size_t bytes,
              std::size_t count,
              input_mode mode,
              std::size_t samples,
              hipStream_t stream)
{
    const std::size_t size = std::max<std::size_t>(bytes / sizeof(T), 4096);
    count = std::max<std::size_t>(1, std::min(count, size));
    const T value = static_cast<T>(1);
    std::vector<T> h_input(size, static_cast<T>(0));

    if(mode == input_mode::normal_half)
    {
        count = size / 2;
        std::fill(h_input.begin() + size / 2, h_input.end(), value);
    }
    else if(mode == input_mode::noise)
    {
        std::fill(h_input.begin(), h_input.end(), value);
        // Break every candidate run except the final one. This mirrors the
        // upstream worst-case/noise benchmark without relying on its timer.
        for(std::size_t pos = count - 1; pos + count < size; pos += count)
        {
            h_input[pos] = static_cast<T>(0);
        }
    }

    T* d_input = nullptr;
    T* d_value = nullptr;
    std::size_t* d_output = nullptr;
    void* d_temp = nullptr;
    std::size_t temp_bytes = 0;
    aka_native_perf::throw_on_hip_error(hipMalloc(&d_input, size * sizeof(T)),
                                         "hipMalloc(input)");
    aka_native_perf::throw_on_hip_error(hipMalloc(&d_value, sizeof(T)),
                                         "hipMalloc(value)");
    aka_native_perf::throw_on_hip_error(hipMalloc(&d_output, sizeof(std::size_t)),
                                         "hipMalloc(output)");
    aka_native_perf::throw_on_hip_error(
        hipMemcpyAsync(d_input,
                       h_input.data(),
                       size * sizeof(T),
                       hipMemcpyHostToDevice,
                       stream),
        "hipMemcpyAsync(input)");
    aka_native_perf::throw_on_hip_error(
        hipMemcpyAsync(d_value, &value, sizeof(T), hipMemcpyHostToDevice, stream),
        "hipMemcpyAsync(value)");
    aka_native_perf::throw_on_hip_error(
        rocprim::search_n(d_temp,
                          temp_bytes,
                          d_input,
                          d_output,
                          size,
                          count,
                          d_value,
                          rocprim::equal_to<T>{},
                          stream,
                          false),
        "temporary storage query");
    aka_native_perf::throw_on_hip_error(hipMalloc(&d_temp, temp_bytes),
                                         "hipMalloc(temp_storage)");
    aka_native_perf::throw_on_hip_error(hipStreamSynchronize(stream),
                                         "hipStreamSynchronize(setup)");

    auto launch = [&](hipStream_t launch_stream) -> hipError_t
    {
        return rocprim::search_n(d_temp,
                                 temp_bytes,
                                 d_input,
                                 d_output,
                                 size,
                                 count,
                                 d_value,
                                 rocprim::equal_to<T>{},
                                 launch_stream,
                                 false);
    };
    const auto result = aka_native_perf::benchmark_graph_or_events(
        launch, stream, /*warmup=*/10, samples, /*target_ms=*/1.0, /*max_repeats=*/1000);
    aka_native_perf::print_result_json(
        id, result, static_cast<double>(size) * sizeof(T));

    hipFree(d_temp);
    hipFree(d_input);
    hipFree(d_value);
    hipFree(d_output);
}

} // namespace

int main(int argc, char** argv)
{
    const std::size_t bytes = parse_size(argc, argv, "--bytes", 64ULL << 20);
    const std::size_t samples = parse_size(argc, argv, "--samples", 20);
    hipStream_t stream = nullptr;
    try
    {
        aka_native_perf::throw_on_hip_error(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        run_case<std::int32_t>(
            "int32_normal_half", bytes, 1, input_mode::normal_half, samples, stream);
        run_case<std::int32_t>(
            "int32_noise_count64", bytes, 64, input_mode::noise, samples, stream);
        run_case<float>(
            "float_noise_count2048", bytes, 2048, input_mode::noise, samples, stream);
        run_case<std::int64_t>(
            "int64_no_match_count4095", bytes, 4095, input_mode::no_match, samples, stream);
        run_case<double>(
            "double_noise_count256", bytes, 256, input_mode::noise, samples, stream);
        aka_native_perf::throw_on_hip_error(hipStreamDestroy(stream), "hipStreamDestroy");
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "native benchmark failed: " << error.what() << std::endl;
        if(stream != nullptr) hipStreamDestroy(stream);
        return 1;
    }
}
