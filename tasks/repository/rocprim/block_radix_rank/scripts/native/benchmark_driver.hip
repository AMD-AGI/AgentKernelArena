// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "hip_graph_benchmark.hpp"

#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_radix_rank.hpp>
#include <rocprim/block/block_store_func.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>

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

template<class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBits,
         rocprim::block_radix_rank_algorithm Algorithm,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize) void aka_rank_kernel(const T* input,
                                                             unsigned int* output)
{
    using rank_type = rocprim::block_radix_rank<BlockSize, RadixBits, Algorithm>;
    const unsigned int lid = threadIdx.x;
    const unsigned int offset = blockIdx.x * BlockSize * ItemsPerThread;

    T keys[ItemsPerThread];
    unsigned int ranks[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, input + offset, keys);

    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; ++trial)
    {
        ROCPRIM_SHARED_MEMORY typename rank_type::storage_type storage;
        for(unsigned int begin_bit = 0; begin_bit < sizeof(T) * 8; begin_bit += RadixBits)
        {
            const unsigned int pass_bits
                = min(RadixBits, static_cast<unsigned int>(sizeof(T) * 8 - begin_bit));
            rank_type().rank_keys(keys, ranks, storage, begin_bit, pass_bits);
        }
    }
    rocprim::block_store_direct_striped<BlockSize>(lid, output + offset, ranks);
}

template<class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         rocprim::block_radix_rank_algorithm Algorithm,
         unsigned int Trials = 10>
void run_case(const char* id,
              std::size_t requested_bytes,
              std::size_t samples,
              hipStream_t stream)
{
    constexpr std::size_t items_per_block = BlockSize * ItemsPerThread;
    const std::size_t requested_items
        = std::max<std::size_t>(requested_bytes / sizeof(T), items_per_block);
    const std::size_t grid_size
        = (requested_items + items_per_block - 1) / items_per_block;
    const std::size_t items = grid_size * items_per_block;

    T* d_input = nullptr;
    unsigned int* d_output = nullptr;
    aka_native_perf::throw_on_hip_error(hipMalloc(&d_input, items * sizeof(T)),
                                         "hipMalloc(input)");
    aka_native_perf::throw_on_hip_error(hipMalloc(&d_output, items * sizeof(unsigned int)),
                                         "hipMalloc(output)");
    aka_native_perf::throw_on_hip_error(
        hipMemsetAsync(d_input, 0x5a, items * sizeof(T), stream), "hipMemsetAsync(input)");
    aka_native_perf::throw_on_hip_error(hipStreamSynchronize(stream),
                                         "hipStreamSynchronize(setup)");

    auto launch = [&](hipStream_t launch_stream) -> hipError_t
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(aka_rank_kernel<T,
                                            BlockSize,
                                            ItemsPerThread,
                                            4,
                                            Algorithm,
                                            Trials>),
            dim3(grid_size),
            dim3(BlockSize),
            0,
            launch_stream,
            d_input,
            d_output);
        return hipGetLastError();
    };
    const auto result = aka_native_perf::benchmark_graph_or_events(
        launch, stream, /*warmup=*/10, samples, /*target_ms=*/1.0, /*max_repeats=*/1000);
    const double workload = static_cast<double>(items) * sizeof(T) * Trials;
    aka_native_perf::print_result_json(id, result, workload);

    aka_native_perf::throw_on_hip_error(hipFree(d_input), "hipFree(input)");
    aka_native_perf::throw_on_hip_error(hipFree(d_output), "hipFree(output)");
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
        run_case<int, 128, 4, rocprim::block_radix_rank_algorithm::basic>(
            "int_bs128_ipt4_basic", bytes, samples, stream);
        run_case<int, 256, 8, rocprim::block_radix_rank_algorithm::basic_memoize>(
            "int_bs256_ipt8_basic_memoize", bytes, samples, stream);
        run_case<std::uint8_t, 256, 16, rocprim::block_radix_rank_algorithm::match>(
            "uint8_bs256_ipt16_match", bytes, samples, stream);
        run_case<long long, 256, 4, rocprim::block_radix_rank_algorithm::match>(
            "int64_bs256_ipt4_match", bytes, samples, stream);
        run_case<unsigned long long, 512, 4, rocprim::block_radix_rank_algorithm::basic>(
            "uint64_bs512_ipt4_basic", bytes, samples, stream);
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
