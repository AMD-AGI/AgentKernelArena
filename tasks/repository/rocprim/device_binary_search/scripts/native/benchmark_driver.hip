// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "hip_graph_benchmark.hpp"

#include <rocprim/device/device_binary_search.hpp>
#include <rocprim/functional.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <string>
#include <vector>

namespace
{

enum class algorithm_kind { binary, lower, upper };

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

template<algorithm_kind Algorithm, class T>
hipError_t dispatch(void* temp,
                    std::size_t& temp_bytes,
                    T* haystack,
                    T* needles,
                    std::size_t* output,
                    std::size_t haystack_size,
                    std::size_t needles_size,
                    hipStream_t stream)
{
    if constexpr(Algorithm == algorithm_kind::binary)
    {
        return rocprim::binary_search(temp,
                                      temp_bytes,
                                      haystack,
                                      needles,
                                      output,
                                      haystack_size,
                                      needles_size,
                                      rocprim::less<T>{},
                                      stream,
                                      false);
    }
    else if constexpr(Algorithm == algorithm_kind::lower)
    {
        return rocprim::lower_bound(temp,
                                    temp_bytes,
                                    haystack,
                                    needles,
                                    output,
                                    haystack_size,
                                    needles_size,
                                    rocprim::less<T>{},
                                    stream,
                                    false);
    }
    else
    {
        return rocprim::upper_bound(temp,
                                    temp_bytes,
                                    haystack,
                                    needles,
                                    output,
                                    haystack_size,
                                    needles_size,
                                    rocprim::less<T>{},
                                    stream,
                                    false);
    }
}

template<algorithm_kind Algorithm, class T>
void run_case(const char* id,
              std::size_t haystack_bytes,
              bool sorted_needles,
              std::size_t samples,
              hipStream_t stream)
{
    const std::size_t haystack_size
        = std::max<std::size_t>(haystack_bytes / sizeof(T), 1024);
    const std::size_t needles_size = std::max<std::size_t>(haystack_size / 10, 1);
    std::vector<T> h_haystack(haystack_size);
    std::vector<T> h_needles(needles_size);
    std::iota(h_haystack.begin(), h_haystack.end(), T{0});
    for(std::size_t i = 0; i < needles_size; ++i)
    {
        h_needles[i] = static_cast<T>((i * 104729ULL + 7919ULL) % haystack_size);
    }
    if(sorted_needles)
    {
        std::sort(h_needles.begin(), h_needles.end());
    }

    T* d_haystack = nullptr;
    T* d_needles = nullptr;
    std::size_t* d_output = nullptr;
    void* d_temp = nullptr;
    std::size_t temp_bytes = 0;
    aka_native_perf::throw_on_hip_error(
        hipMalloc(&d_haystack, haystack_size * sizeof(T)), "hipMalloc(haystack)");
    aka_native_perf::throw_on_hip_error(
        hipMalloc(&d_needles, needles_size * sizeof(T)), "hipMalloc(needles)");
    aka_native_perf::throw_on_hip_error(
        hipMalloc(&d_output, needles_size * sizeof(std::size_t)), "hipMalloc(output)");
    aka_native_perf::throw_on_hip_error(
        hipMemcpyAsync(d_haystack,
                       h_haystack.data(),
                       haystack_size * sizeof(T),
                       hipMemcpyHostToDevice,
                       stream),
        "hipMemcpyAsync(haystack)");
    aka_native_perf::throw_on_hip_error(
        hipMemcpyAsync(d_needles,
                       h_needles.data(),
                       needles_size * sizeof(T),
                       hipMemcpyHostToDevice,
                       stream),
        "hipMemcpyAsync(needles)");
    aka_native_perf::throw_on_hip_error(
        dispatch<Algorithm>(d_temp,
                            temp_bytes,
                            d_haystack,
                            d_needles,
                            d_output,
                            haystack_size,
                            needles_size,
                            stream),
        "temporary storage query");
    aka_native_perf::throw_on_hip_error(hipMalloc(&d_temp, temp_bytes),
                                         "hipMalloc(temp_storage)");
    aka_native_perf::throw_on_hip_error(hipStreamSynchronize(stream),
                                         "hipStreamSynchronize(setup)");

    auto launch = [&](hipStream_t launch_stream) -> hipError_t
    {
        return dispatch<Algorithm>(d_temp,
                                   temp_bytes,
                                   d_haystack,
                                   d_needles,
                                   d_output,
                                   haystack_size,
                                   needles_size,
                                   launch_stream);
    };
    const auto result = aka_native_perf::benchmark_graph_or_events(
        launch, stream, /*warmup=*/5, samples, /*target_ms=*/1.0, /*max_repeats=*/1000);
    const double workload = static_cast<double>(needles_size) * sizeof(T);
    aka_native_perf::print_result_json(id, result, workload);

    hipFree(d_temp);
    hipFree(d_haystack);
    hipFree(d_needles);
    hipFree(d_output);
}

} // namespace

int main(int argc, char** argv)
{
    const std::size_t bytes = parse_size(argc, argv, "--bytes", 32ULL << 20);
    const std::size_t samples = parse_size(argc, argv, "--samples", 20);
    hipStream_t stream = nullptr;
    try
    {
        aka_native_perf::throw_on_hip_error(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        run_case<algorithm_kind::binary, float>(
            "binary_float_random_needles", bytes, false, samples, stream);
        run_case<algorithm_kind::binary, float>(
            "binary_float_sorted_needles", bytes, true, samples, stream);
        run_case<algorithm_kind::lower, std::int32_t>(
            "lower_bound_int32_random_needles", bytes, false, samples, stream);
        run_case<algorithm_kind::lower, std::int32_t>(
            "lower_bound_int32_sorted_needles", bytes, true, samples, stream);
        run_case<algorithm_kind::upper, double>(
            "upper_bound_double_random_needles", bytes, false, samples, stream);
        run_case<algorithm_kind::upper, double>(
            "upper_bound_double_sorted_needles", bytes, true, samples, stream);
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
