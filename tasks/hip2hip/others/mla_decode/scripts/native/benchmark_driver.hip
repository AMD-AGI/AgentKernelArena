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
        std::vector<bf16> h_q(q_e);
        std::vector<uint8_t> h_kv(kv_e);
        std::vector<bf16> h_golden(o_e);
        std::vector<bf16> h_actual(o_e);
        for(std::size_t i = 0; i < q_e; ++i)
        {
            const int value = static_cast<int>((i * 17U) % 31U) - 15;
            h_q[i] = __float2bfloat16(static_cast<float>(value) * 0.015625f);
        }
        for(std::size_t i = 0; i < kv_e; ++i)
        {
            const int value = static_cast<int>((i * 13U + 5U) % 29U) - 14;
            h_kv[i] = f32_to_fp8_e4m3fn(static_cast<float>(value) * 0.03125f);
        }
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
            hipMemcpyAsync(d_q, h_q.data(), q_e * sizeof(bf16),
                           hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync(q)");
        aka_native_perf::throw_on_hip_error(
            hipMemcpyAsync(d_kv, h_kv.data(), kv_e * sizeof(uint8_t),
                           hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync(kv)");
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

        aka_native_perf::throw_on_hip_error(
            hipMemsetAsync(d_o, 0xFF, o_e * sizeof(bf16), stream),
            "hipMemsetAsync(o golden)");
        aka_native_perf::throw_on_hip_error(launch(stream), "golden eager launch");
        aka_native_perf::throw_on_hip_error(
            hipMemcpyAsync(h_golden.data(), d_o, o_e * sizeof(bf16),
                           hipMemcpyDeviceToHost, stream),
            "hipMemcpyAsync(golden)");
        aka_native_perf::throw_on_hip_error(
            hipStreamSynchronize(stream), "hipStreamSynchronize(golden)");

        auto validate = [&](hipGraphExec_t graph_exec,
                            hipStream_t validation_stream) -> std::string
        {
            hipError_t error
                = hipMemsetAsync(d_o, 0xFF, o_e * sizeof(bf16), validation_stream);
            if(error == hipSuccess)
            {
                error = graph_exec != nullptr
                    ? hipGraphLaunch(graph_exec, validation_stream)
                    : launch(validation_stream);
            }
            if(error == hipSuccess)
            {
                error = hipMemcpyAsync(h_actual.data(), d_o, o_e * sizeof(bf16),
                                       hipMemcpyDeviceToHost, validation_stream);
            }
            if(error == hipSuccess)
            {
                error = hipStreamSynchronize(validation_stream);
            }
            if(error != hipSuccess)
            {
                return aka_native_perf::hip_failure("MLA replay validation", error);
            }
            for(std::size_t i = 0; i < o_e; ++i)
            {
                const float expected = __bfloat162float(h_golden[i]);
                const float actual = __bfloat162float(h_actual[i]);
                if(!std::isfinite(actual)
                   || std::fabs(actual - expected) > 1.0e-2f)
                {
                    return "output mismatch at element " + std::to_string(i);
                }
            }
            return {};
        };

        const auto result = aka_native_perf::benchmark_graph_or_events(
            launch,
            stream,
            /*warmup=*/10,
            static_cast<std::size_t>(samples),
            /*target_ms=*/1.0,
            /*max_repeats=*/1000,
            validate);
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
