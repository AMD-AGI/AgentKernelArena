// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "hip_graph_benchmark.hpp"

#include <rocprim/device/device_merge_sort.hpp>
#include <rocprim/functional.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

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
std::vector<T> make_keys(std::size_t size)
{
    std::vector<T> result(size);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for(std::size_t i = 0; i < size; ++i)
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        result[i] = static_cast<T>(state >> 17);
    }
    return result;
}

template<class Key>
void run_keys_case(const char* id,
                   std::size_t bytes,
                   std::size_t samples,
                   hipStream_t stream)
{
    const std::size_t size = std::max<std::size_t>(bytes / sizeof(Key), 1024);
    const std::vector<Key> h_keys = make_keys<Key>(size);
    Key* d_input = nullptr;
    Key* d_output = nullptr;
    void* d_temp = nullptr;
    std::size_t temp_bytes = 0;

    aka_native_perf::throw_on_hip_error(hipMalloc(&d_input, size * sizeof(Key)),
                                         "hipMalloc(keys_input)");
    aka_native_perf::throw_on_hip_error(hipMalloc(&d_output, size * sizeof(Key)),
                                         "hipMalloc(keys_output)");
    aka_native_perf::throw_on_hip_error(
        hipMemcpyAsync(d_input,
                       h_keys.data(),
                       size * sizeof(Key),
                       hipMemcpyHostToDevice,
                       stream),
        "hipMemcpyAsync(keys)");
    aka_native_perf::throw_on_hip_error(
        rocprim::merge_sort(d_temp,
                            temp_bytes,
                            d_input,
                            d_output,
                            size,
                            rocprim::less<Key>{},
                            stream,
                            false),
        "temporary storage query");
    aka_native_perf::throw_on_hip_error(hipMalloc(&d_temp, temp_bytes),
                                         "hipMalloc(temp_storage)");
    aka_native_perf::throw_on_hip_error(hipStreamSynchronize(stream),
                                         "hipStreamSynchronize(setup)");

    auto launch = [&](hipStream_t launch_stream) -> hipError_t
    {
        return rocprim::merge_sort(d_temp,
                                   temp_bytes,
                                   d_input,
                                   d_output,
                                   size,
                                   rocprim::less<Key>{},
                                   launch_stream,
                                   false);
    };
    const auto result = aka_native_perf::benchmark_graph_or_events(
        launch, stream, /*warmup=*/5, samples, /*target_ms=*/1.0, /*max_repeats=*/1000);
    aka_native_perf::print_result_json(
        id, result, static_cast<double>(size) * sizeof(Key));

    hipFree(d_temp);
    hipFree(d_input);
    hipFree(d_output);
}

template<class Key, class Value>
void run_pairs_case(const char* id,
                    std::size_t bytes,
                    std::size_t samples,
                    hipStream_t stream)
{
    const std::size_t size = std::max<std::size_t>(bytes / sizeof(Key), 1024);
    const std::vector<Key> h_keys = make_keys<Key>(size);
    std::vector<Value> h_values(size);
    for(std::size_t i = 0; i < size; ++i)
    {
        h_values[i] = static_cast<Value>(i);
    }

    Key* d_keys_input = nullptr;
    Key* d_keys_output = nullptr;
    Value* d_values_input = nullptr;
    Value* d_values_output = nullptr;
    void* d_temp = nullptr;
    std::size_t temp_bytes = 0;
    aka_native_perf::throw_on_hip_error(
        hipMalloc(&d_keys_input, size * sizeof(Key)), "hipMalloc(keys_input)");
    aka_native_perf::throw_on_hip_error(
        hipMalloc(&d_keys_output, size * sizeof(Key)), "hipMalloc(keys_output)");
    aka_native_perf::throw_on_hip_error(
        hipMalloc(&d_values_input, size * sizeof(Value)), "hipMalloc(values_input)");
    aka_native_perf::throw_on_hip_error(
        hipMalloc(&d_values_output, size * sizeof(Value)), "hipMalloc(values_output)");
    aka_native_perf::throw_on_hip_error(
        hipMemcpyAsync(d_keys_input,
                       h_keys.data(),
                       size * sizeof(Key),
                       hipMemcpyHostToDevice,
                       stream),
        "hipMemcpyAsync(keys)");
    aka_native_perf::throw_on_hip_error(
        hipMemcpyAsync(d_values_input,
                       h_values.data(),
                       size * sizeof(Value),
                       hipMemcpyHostToDevice,
                       stream),
        "hipMemcpyAsync(values)");
    aka_native_perf::throw_on_hip_error(
        rocprim::merge_sort(d_temp,
                            temp_bytes,
                            d_keys_input,
                            d_keys_output,
                            d_values_input,
                            d_values_output,
                            size,
                            rocprim::less<Key>{},
                            stream,
                            false),
        "temporary storage query");
    aka_native_perf::throw_on_hip_error(hipMalloc(&d_temp, temp_bytes),
                                         "hipMalloc(temp_storage)");
    aka_native_perf::throw_on_hip_error(hipStreamSynchronize(stream),
                                         "hipStreamSynchronize(setup)");

    auto launch = [&](hipStream_t launch_stream) -> hipError_t
    {
        return rocprim::merge_sort(d_temp,
                                   temp_bytes,
                                   d_keys_input,
                                   d_keys_output,
                                   d_values_input,
                                   d_values_output,
                                   size,
                                   rocprim::less<Key>{},
                                   launch_stream,
                                   false);
    };
    const auto result = aka_native_perf::benchmark_graph_or_events(
        launch, stream, /*warmup=*/5, samples, /*target_ms=*/1.0, /*max_repeats=*/1000);
    const double workload = static_cast<double>(size) * (sizeof(Key) + sizeof(Value));
    aka_native_perf::print_result_json(id, result, workload);

    hipFree(d_temp);
    hipFree(d_keys_input);
    hipFree(d_keys_output);
    hipFree(d_values_input);
    hipFree(d_values_output);
}

} // namespace

int main(int argc, char** argv)
{
    const std::size_t bytes = parse_size(argc, argv, "--bytes", 16ULL << 20);
    const std::size_t samples = parse_size(argc, argv, "--samples", 20);
    hipStream_t stream = nullptr;
    try
    {
        aka_native_perf::throw_on_hip_error(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        run_keys_case<std::int32_t>("keys_int32", bytes, samples, stream);
        run_keys_case<long long>("keys_int64", bytes, samples, stream);
        run_keys_case<std::uint8_t>("keys_uint8", bytes, samples, stream);
        run_pairs_case<std::int32_t, float>("pairs_int32_float", bytes, samples, stream);
        run_pairs_case<long long, double>("pairs_int64_double", bytes, samples, stream);
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
