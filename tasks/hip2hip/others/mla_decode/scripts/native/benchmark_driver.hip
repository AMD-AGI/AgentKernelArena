// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "hip_graph_benchmark.hpp"

#define main aka_mla_candidate_main
#include "../../mla_decode.hip"
#undef main
#include "output_validation.hpp"

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
        std::vector<bf16> h_reference(o_e);
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
        mla_routing_inputs(h_r2t, h_bsl, batch, ctx, kv_tokens);

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

        host_reference(h_reference, h_q, h_kv, h_r2t, h_bsl, batch, ctx, sm_scale);

        std::vector<bf16> checked_q(q_e);
        std::vector<uint8_t> checked_kv(kv_e);
        std::vector<int32_t> checked_r2t(r2t_e), checked_bsl(batch);
        auto check_inputs = [&](const std::vector<bf16>& expected_q,
                                const std::vector<uint8_t>& expected_kv,
                                hipStream_t check_stream) -> std::string
        {
            hipError_t status = hipMemcpyAsync(checked_q.data(), d_q, q_e * sizeof(bf16), hipMemcpyDeviceToHost, check_stream);
            if(status == hipSuccess) status = hipMemcpyAsync(checked_kv.data(), d_kv, kv_e, hipMemcpyDeviceToHost, check_stream);
            if(status == hipSuccess) status = hipMemcpyAsync(checked_r2t.data(), d_r2t, r2t_e * sizeof(int32_t), hipMemcpyDeviceToHost, check_stream);
            if(status == hipSuccess) status = hipMemcpyAsync(checked_bsl.data(), d_bsl, batch * sizeof(int32_t), hipMemcpyDeviceToHost, check_stream);
            if(status == hipSuccess) status = hipStreamSynchronize(check_stream);
            if(status != hipSuccess) return aka_native_perf::hip_failure("MLA input-state validation", status);
            if(!mla_same_buffer(checked_q, expected_q) || !mla_same_buffer(checked_kv, expected_kv)
               || !mla_same_buffer(checked_r2t, h_r2t) || !mla_same_buffer(checked_bsl, h_bsl))
                return "MLA modified a caller-owned input";
            return {};
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
            auto restore_inputs = [&]() -> bool
            {
                hipError_t rq = hipMemcpyAsync(d_q, h_q.data(), q_e * sizeof(bf16), hipMemcpyHostToDevice, validation_stream);
                hipError_t rk = hipMemcpyAsync(d_kv, h_kv.data(), kv_e, hipMemcpyHostToDevice, validation_stream);
                hipError_t ri = hipMemcpyAsync(d_r2t, h_r2t.data(), r2t_e * sizeof(int32_t), hipMemcpyHostToDevice, validation_stream);
                hipError_t rl = hipMemcpyAsync(d_bsl, h_bsl.data(), batch * sizeof(int32_t), hipMemcpyHostToDevice, validation_stream);
                hipError_t sync = hipStreamSynchronize(validation_stream);
                return rq == hipSuccess && rk == hipSuccess && ri == hipSuccess && rl == hipSuccess && sync == hipSuccess;
            };
            return mla_with_restored_inputs([&]() -> std::string
            {
                // Validate the output left by the actual measured invocation before
                // changing it, then validate the exact graph after poisoning.
                hipError_t error = hipMemcpyAsync(h_actual.data(), d_o, o_e * sizeof(bf16),
                                                  hipMemcpyDeviceToHost, validation_stream);
                if(error == hipSuccess) error = hipStreamSynchronize(validation_stream);
                if(error != hipSuccess) return aka_native_perf::hip_failure("MLA measured output", error);
                auto decode = [](bf16 value) { return __bfloat162float(value); };
                std::string failure = validate_mla_outputs(h_actual, h_golden, h_reference, decode);
                if(!failure.empty()) return failure;
                failure = check_inputs(h_q, h_kv, validation_stream);
                if(!failure.empty()) return failure;
                error = hipMemsetAsync(d_o, 0xFF, o_e * sizeof(bf16), validation_stream);
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
                failure = validate_mla_outputs(h_actual, h_golden, h_reference, decode);
                if(!failure.empty()) return failure;
                failure = check_inputs(h_q, h_kv, validation_stream);
                if(!failure.empty()) return failure;

                // Fresh Q=0 with mapped, token-varying KV has an independent
                // closed-form mean. This also rejects identity addressing and
                // reading padded tokens. The enclosing guard restores all inputs.
                const std::vector<bf16> zero_q(q_e, __float2bfloat16(0.0f));
                std::vector<uint8_t> control_kv(kv_e);
                const auto means = mla_routing_known_answer(control_kv, h_r2t, h_bsl, batch, ctx, LK, f32_to_fp8_e4m3fn);
                std::vector<bf16> expected(o_e);
                for(int b = 0; b < batch; ++b)
                    std::fill_n(expected.begin() + std::size_t(b) * NHEAD * LV, NHEAD * LV, __float2bfloat16(means[b]));
                error = hipMemcpyAsync(d_q, zero_q.data(), q_e * sizeof(bf16), hipMemcpyHostToDevice, validation_stream);
                if(error == hipSuccess) error = hipMemcpyAsync(d_kv, control_kv.data(), kv_e, hipMemcpyHostToDevice, validation_stream);
                if(error == hipSuccess) error = hipMemsetAsync(d_o, 0xFF, o_e * sizeof(bf16), validation_stream);
                if(error == hipSuccess) error = graph_exec != nullptr ? hipGraphLaunch(graph_exec, validation_stream) : launch(validation_stream);
                if(error == hipSuccess) error = hipMemcpyAsync(h_actual.data(), d_o, o_e * sizeof(bf16), hipMemcpyDeviceToHost, validation_stream);
                if(error == hipSuccess) error = hipStreamSynchronize(validation_stream);
                if(error != hipSuccess) failure = aka_native_perf::hip_failure("MLA fresh known answer", error);
                else
                {
                    // Fresh inputs use the original reference gate. The eager/replay
                    // consistency gate above remains for the original timed inputs.
                    failure = validate_mla_outputs(h_actual, h_actual, expected, decode);
                }
                if(failure.empty()) failure = check_inputs(zero_q, control_kv, validation_stream);
                return failure;
            }, restore_inputs);
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
            = static_cast<double>(std::accumulate(h_bsl.begin(), h_bsl.end(), int64_t{0})) * LK
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
