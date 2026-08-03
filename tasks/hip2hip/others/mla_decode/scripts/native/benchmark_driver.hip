// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "hip_graph_benchmark.hpp"

#define main aka_mla_candidate_main
#include "../../mla_decode.hip"
#undef main

#include <cstdlib>
#include <exception>
#include <string>

namespace
{

int parse_int(int argc, char** argv, const char* name, int fallback)
{
    for(int i = 1; i + 1 < argc; ++i)
    {
        if(std::string(argv[i]) == name)
        {
            return std::atoi(argv[i + 1]);
        }
    }
    return fallback;
}

} // namespace

int main(int argc, char** argv)
{
    const int batch   = parse_int(argc, argv, "--batch", 4);
    const int ctx     = parse_int(argc, argv, "--ctx", 1024);
    const int samples = parse_int(argc, argv, "--samples", 100);
    if(batch <= 0 || ctx <= 0 || samples <= 0)
    {
        std::cerr << "batch, ctx, and samples must be positive" << std::endl;
        return 2;
    }

    bf16* d_q = nullptr;
    bf16* d_o = nullptr;
    uint8_t* d_kv = nullptr;
    int32_t* d_r2t = nullptr;
    int32_t* d_bsl = nullptr;
    hipStream_t stream = nullptr;
    try
    {
        const float sm_scale = 1.0f / std::sqrt(static_cast<float>(LK));
        const std::size_t q_e
            = static_cast<std::size_t>(batch) * NHEAD * LK;
        const std::size_t o_e
            = static_cast<std::size_t>(batch) * NHEAD * LV;
        const std::size_t r2t_e = static_cast<std::size_t>(batch) * ctx;
        const std::size_t kv_tokens = r2t_e + 1024;
        const std::size_t kv_e = kv_tokens * LK;

        std::vector<int32_t> h_r2t(r2t_e);
        std::vector<int32_t> h_bsl(batch, ctx);
        for(int b = 0; b < batch; ++b)
        {
            for(int t = 0; t < ctx; ++t)
            {
                h_r2t[static_cast<std::size_t>(b) * ctx + t] = b * ctx + t;
            }
        }

        aka_native_perf::throw_on_hip_error(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_q, q_e * sizeof(bf16)),
                                             "hipMalloc(q)");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_o, o_e * sizeof(bf16)),
                                             "hipMalloc(o)");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_kv, kv_e * sizeof(uint8_t)),
                                             "hipMalloc(kv)");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_r2t, r2t_e * sizeof(int32_t)),
                                             "hipMalloc(req_to_token)");
        aka_native_perf::throw_on_hip_error(hipMalloc(&d_bsl, batch * sizeof(int32_t)),
                                             "hipMalloc(seq_len)");
        aka_native_perf::throw_on_hip_error(
            hipMemsetAsync(d_q, 0, q_e * sizeof(bf16), stream), "hipMemsetAsync(q)");
        aka_native_perf::throw_on_hip_error(
            hipMemsetAsync(d_kv, 0, kv_e * sizeof(uint8_t), stream), "hipMemsetAsync(kv)");
        aka_native_perf::throw_on_hip_error(
            hipMemcpyAsync(d_r2t,
                           h_r2t.data(),
                           r2t_e * sizeof(int32_t),
                           hipMemcpyHostToDevice,
                           stream),
            "hipMemcpyAsync(req_to_token)");
        aka_native_perf::throw_on_hip_error(
            hipMemcpyAsync(d_bsl,
                           h_bsl.data(),
                           batch * sizeof(int32_t),
                           hipMemcpyHostToDevice,
                           stream),
            "hipMemcpyAsync(seq_len)");
        aka_native_perf::throw_on_hip_error(hipStreamSynchronize(stream),
                                             "hipStreamSynchronize(setup)");

        const dim3 grid(batch, HEAD_GROUPS);
        const dim3 block(BLOCK_THREADS);
        const std::size_t smem = BLOCK_H * LK * sizeof(float);
    auto launch = [&](hipStream_t launch_stream) -> hipError_t
        {
            hipLaunchKernelGGL(mla_decode_kernel,
                               grid,
                               block,
                               smem,
                               launch_stream,
                               d_q,
                               d_kv,
                               d_r2t,
                               d_bsl,
                               sm_scale,
                               d_o,
                               static_cast<float*>(nullptr),
                               ctx,
                               ctx);
            return hipGetLastError();
        };

        const auto result = aka_native_perf::benchmark_graph_or_events(
            launch,
            stream,
            /*warmup=*/10,
            static_cast<std::size_t>(samples),
            /*target_ms=*/1.0,
            /*max_repeats=*/1000);
        const double bytes_per_launch
            = static_cast<double>(batch) * ctx * LK
              + static_cast<double>(batch) * NHEAD * (LK + LV) * sizeof(bf16);
        aka_native_perf::print_result_json("mla_decode", result, bytes_per_launch);

        hipFree(d_q); d_q = nullptr;
        hipFree(d_o); d_o = nullptr;
        hipFree(d_kv); d_kv = nullptr;
        hipFree(d_r2t); d_r2t = nullptr;
        hipFree(d_bsl); d_bsl = nullptr;
        aka_native_perf::throw_on_hip_error(hipStreamDestroy(stream), "hipStreamDestroy");
        stream = nullptr;
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "native benchmark failed: " << error.what() << std::endl;
        if(d_q != nullptr) hipFree(d_q);
        if(d_o != nullptr) hipFree(d_o);
        if(d_kv != nullptr) hipFree(d_kv);
        if(d_r2t != nullptr) hipFree(d_r2t);
        if(d_bsl != nullptr) hipFree(d_bsl);
        if(stream != nullptr) hipStreamDestroy(stream);
        return 1;
    }
}
