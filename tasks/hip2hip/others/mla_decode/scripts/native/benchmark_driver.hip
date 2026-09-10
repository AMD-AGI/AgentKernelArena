// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "hip_graph_benchmark.hpp"

#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using bf16 = __hip_bfloat16;

// Keep the candidate in a separate translation unit. In particular, do not
// include mla_decode.hip here: candidate preprocessor definitions and workload
// constants must not affect protected allocation, launch, validation, or timing.
extern "C" __global__ void mla_decode_kernel(
    const bf16* q,
    const uint8_t* kv,
    const int32_t* req_to_token,
    const int32_t* b_seq_len,
    float sm_scale,
    bf16* o,
    float* lse,
    int stride_r2t_b,
    int max_ctx);

namespace
{

// Protected shape and launch contract. Candidate implementations may organize
// the same work differently, but cannot change the buffers or reported work.
constexpr int kNHead        = 128;
constexpr int kBlockH       = 16;
constexpr int kHeadGroups   = kNHead / kBlockH;
constexpr int kLK           = 576;
constexpr int kLV           = 512;
constexpr int kBlockThreads = 64;

enum class Mode
{
    Full,
    Check,
    Bench,
};

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

bool parse_mode(int argc, char** argv, Mode& mode)
{
    mode = Mode::Full;
    for(int i = 1; i + 1 < argc; ++i)
    {
        if(std::string(argv[i]) != "--mode")
        {
            continue;
        }
        const std::string value(argv[i + 1]);
        if(value == "full")
        {
            mode = Mode::Full;
        }
        else if(value == "check")
        {
            mode = Mode::Check;
        }
        else if(value == "bench")
        {
            mode = Mode::Bench;
        }
        else
        {
            std::cerr << "unknown --mode " << value << std::endl;
            return false;
        }
    }
    return true;
}

float fp8_e4m3fn_to_f32(uint8_t value)
{
    if(value == 0x80 || value == 0x00)
    {
        return 0.0f;
    }
    const int sign = (value & 0x80) != 0 ? -1 : 1;
    const int exponent = (value >> 3) & 0xF;
    const int mantissa_bits = value & 0x7;
    float decoded = 0.0f;
    if(exponent == 0)
    {
        decoded = static_cast<float>(mantissa_bits) / 8.0f / 64.0f;
    }
    else
    {
        const float mantissa = 1.0f + static_cast<float>(mantissa_bits) / 8.0f;
        decoded = mantissa * std::exp2(static_cast<float>(exponent) - 7.0f);
    }
    return static_cast<float>(sign) * decoded;
}

uint8_t f32_to_fp8_e4m3fn(float value)
{
    if(value == 0.0f || !std::isfinite(value))
    {
        return 0x00;
    }
    const uint32_t sign_bit = value < 0.0f ? 0x80 : 0x00;
    const float magnitude = std::fabs(value);
    if(magnitude >= 448.0f)
    {
        return static_cast<uint8_t>(sign_bit | 0x7E);
    }
    const int unbiased_exponent = static_cast<int>(std::floor(std::log2(magnitude)));
    int biased_exponent = unbiased_exponent + 7;
    if(biased_exponent <= 0)
    {
        int mantissa = static_cast<int>(std::round(magnitude * 512.0f));
        mantissa = std::min(mantissa, 7);
        return static_cast<uint8_t>(sign_bit | (mantissa & 0x7));
    }
    biased_exponent = std::min(biased_exponent, 15);
    const float fraction
        = magnitude / std::ldexp(1.0f, unbiased_exponent) - 1.0f;
    int mantissa = static_cast<int>(std::round(fraction * 8.0f));
    if(mantissa == 8)
    {
        ++biased_exponent;
        mantissa = 0;
        if(biased_exponent > 15)
        {
            biased_exponent = 15;
            mantissa = 7;
        }
    }
    mantissa = std::max(0, std::min(mantissa, 7));
    return static_cast<uint8_t>(sign_bit | ((biased_exponent & 0xF) << 3)
                                | (mantissa & 0x7));
}

bool bf16_is_finite(bf16 value)
{
    static_assert(sizeof(bf16) == sizeof(uint16_t), "unexpected bf16 size");
    uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7F80U) != 0x7F80U;
}

void fill_inputs(std::vector<bf16>& q,
                 std::vector<uint8_t>& kv,
                 int batch,
                 int ctx)
{
    std::mt19937 q_rng(42U + static_cast<uint32_t>(batch * 31 + ctx));
    std::uniform_real_distribution<float> q_dist(-1.0f, 1.0f);
    for(auto& value : q)
    {
        value = __float2bfloat16(q_dist(q_rng));
    }

    std::mt19937 kv_rng(123U + static_cast<uint32_t>(batch * 17 + ctx));
    std::normal_distribution<float> kv_dist(0.0f, 1.0f);
    for(auto& value : kv)
    {
        value = f32_to_fp8_e4m3fn(kv_dist(kv_rng));
    }
}

void host_reference(std::vector<bf16>& output,
                    const std::vector<bf16>& q,
                    const std::vector<uint8_t>& kv,
                    const std::vector<int32_t>& req_to_token,
                    const std::vector<int32_t>& seq_len,
                    int batch,
                    int max_ctx,
                    float sm_scale)
{
    output.assign(static_cast<std::size_t>(batch) * kNHead * kLV,
                  __float2bfloat16(0.0f));
#if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
#endif
    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < kNHead; ++h)
        {
            const int sequence = seq_len[b];
            std::vector<float> scores(sequence, 0.0f);
            for(int token = 0; token < sequence; ++token)
            {
                const int physical
                    = req_to_token[static_cast<std::size_t>(b) * max_ctx + token];
                const uint8_t* k_row
                    = kv.data() + static_cast<std::size_t>(physical) * kLK;
                float score = 0.0f;
                for(int dim = 0; dim < kLK; ++dim)
                {
                    const float q_value = __bfloat162float(
                        q[(static_cast<std::size_t>(b) * kNHead + h) * kLK + dim]);
                    score += q_value * fp8_e4m3fn_to_f32(k_row[dim]);
                }
                scores[token] = score * sm_scale;
            }

            float maximum = -INFINITY;
            for(float score : scores)
            {
                maximum = std::fmax(maximum, score);
            }
            double sum = 0.0;
            for(float& score : scores)
            {
                score = std::exp(score - maximum);
                sum += score;
            }

            const float inverse_sum = static_cast<float>(1.0 / sum);
            std::vector<float> accumulated(kLV, 0.0f);
            for(int token = 0; token < sequence; ++token)
            {
                const int physical
                    = req_to_token[static_cast<std::size_t>(b) * max_ctx + token];
                const uint8_t* v_row
                    = kv.data() + static_cast<std::size_t>(physical) * kLK;
                const float probability = scores[token] * inverse_sum;
                for(int dim = 0; dim < kLV; ++dim)
                {
                    accumulated[dim]
                        += probability * fp8_e4m3fn_to_f32(v_row[dim]);
                }
            }
            for(int dim = 0; dim < kLV; ++dim)
            {
                output[(static_cast<std::size_t>(b) * kNHead + h) * kLV + dim]
                    = __float2bfloat16(accumulated[dim]);
            }
        }
    }
}

bool compare_to_reference(const std::vector<bf16>& actual,
                          const std::vector<bf16>& expected,
                          double& max_abs,
                          double& max_rel)
{
    max_abs = 0.0;
    max_rel = 0.0;
    if(actual.size() != expected.size())
    {
        max_abs = INFINITY;
        max_rel = INFINITY;
        return false;
    }
    for(std::size_t i = 0; i < actual.size(); ++i)
    {
        if(!bf16_is_finite(actual[i]) || !bf16_is_finite(expected[i]))
        {
            max_abs = INFINITY;
            max_rel = INFINITY;
            return false;
        }
        const double actual_value
            = static_cast<double>(__bfloat162float(actual[i]));
        const double expected_value
            = static_cast<double>(__bfloat162float(expected[i]));
        const double difference = std::fabs(actual_value - expected_value);
        if(!std::isfinite(difference))
        {
            max_abs = INFINITY;
            max_rel = INFINITY;
            return false;
        }
        max_abs = std::fmax(max_abs, difference);
        const double denominator
            = std::fmax(1.0e-8,
                        std::fmax(std::fabs(actual_value), std::fabs(expected_value)));
        max_rel = std::fmax(max_rel, difference / denominator);
    }
    constexpr double kAbsoluteTolerance = 5.0e-2;
    constexpr double kRelativeTolerance = 1.0e-1;
    return max_abs <= kAbsoluteTolerance || max_rel <= kRelativeTolerance;
}

} // namespace

int main(int argc, char** argv)
{
    const int batch   = parse_int(argc, argv, "--batch", 4);
    const int ctx     = parse_int(argc, argv, "--ctx", 1024);
    const int samples = parse_int(argc, argv, "--samples", 100);
    Mode mode;
    if(!parse_mode(argc, argv, mode))
    {
        return 2;
    }
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
        const float sm_scale = 1.0f / std::sqrt(static_cast<float>(kLK));
        const std::size_t q_e
            = static_cast<std::size_t>(batch) * kNHead * kLK;
        const std::size_t o_e
            = static_cast<std::size_t>(batch) * kNHead * kLV;
        const std::size_t r2t_e = static_cast<std::size_t>(batch) * ctx;
        const std::size_t kv_tokens = r2t_e + 1024;
        const std::size_t kv_e = kv_tokens * kLK;

        std::vector<int32_t> h_r2t(r2t_e);
        std::vector<int32_t> h_bsl(batch, ctx);
        std::vector<bf16> h_q(q_e);
        std::vector<uint8_t> h_kv(kv_e);
        std::vector<bf16> h_golden(o_e);
        std::vector<bf16> h_actual(o_e);
        fill_inputs(h_q, h_kv, batch, ctx);
        for(int b = 0; b < batch; ++b)
        {
            for(int token = 0; token < ctx; ++token)
            {
                h_r2t[static_cast<std::size_t>(b) * ctx + token] = b * ctx + token;
            }
        }

        aka_native_perf::throw_on_hip_error(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            "hipStreamCreateWithFlags");
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
            hipMemcpyAsync(d_r2t, h_r2t.data(), r2t_e * sizeof(int32_t),
                           hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync(req_to_token)");
        aka_native_perf::throw_on_hip_error(
            hipMemcpyAsync(d_bsl, h_bsl.data(), batch * sizeof(int32_t),
                           hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync(seq_len)");
        aka_native_perf::throw_on_hip_error(
            hipStreamSynchronize(stream), "hipStreamSynchronize(setup)");

        const dim3 grid(batch, kHeadGroups);
        const dim3 block(kBlockThreads);
        const std::size_t smem = kBlockH * kLK * sizeof(float);
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

        bool check_ok = true;
        if(mode == Mode::Full || mode == Mode::Check)
        {
            std::fill(h_actual.begin(), h_actual.end(),
                      __float2bfloat16(std::numeric_limits<float>::quiet_NaN()));
            aka_native_perf::throw_on_hip_error(
                hipMemcpyAsync(d_o, h_actual.data(), o_e * sizeof(bf16),
                               hipMemcpyHostToDevice, stream),
                "hipMemcpyAsync(poison output)");
            aka_native_perf::throw_on_hip_error(launch(stream), "correctness launch");
            aka_native_perf::throw_on_hip_error(
                hipMemcpyAsync(h_actual.data(), d_o, o_e * sizeof(bf16),
                               hipMemcpyDeviceToHost, stream),
                "hipMemcpyAsync(correctness output)");
            aka_native_perf::throw_on_hip_error(
                hipStreamSynchronize(stream), "hipStreamSynchronize(correctness)");

            host_reference(h_golden, h_q, h_kv, h_r2t, h_bsl,
                           batch, ctx, sm_scale);
            double max_abs = 0.0;
            double max_rel = 0.0;
            check_ok = compare_to_reference(h_actual, h_golden, max_abs, max_rel);
            std::cout << "Check: max_abs=" << max_abs << " max_rel=" << max_rel
                      << " -> " << (check_ok ? "PASS" : "FAIL") << std::endl;
        }

        if(check_ok && (mode == Mode::Full || mode == Mode::Bench))
        {
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
                    if(!bf16_is_finite(h_actual[i]) || !bf16_is_finite(h_golden[i]))
                    {
                        return "non-finite output at element " + std::to_string(i);
                    }
                    const float expected = __bfloat162float(h_golden[i]);
                    const float actual = __bfloat162float(h_actual[i]);
                    if(std::fabs(actual - expected) > 1.0e-2f)
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
                = static_cast<double>(batch) * ctx * kLK
                  + static_cast<double>(batch) * kNHead * (kLK + kLV)
                        * sizeof(bf16);
            aka_native_perf::print_result_json("mla_decode", result, bytes_per_launch);
            if(mode == Mode::Full)
            {
                const double microseconds = result.execution_time_ms * 1000.0;
                const double bandwidth_gbs
                    = bytes_per_launch / (result.execution_time_ms * 1.0e6);
                std::cout << "Perf: " << microseconds << " us/launch | ~BW: "
                          << bandwidth_gbs << " GB/s" << std::endl;
            }
        }

        hipFree(d_q); d_q = nullptr;
        hipFree(d_o); d_o = nullptr;
        hipFree(d_kv); d_kv = nullptr;
        hipFree(d_r2t); d_r2t = nullptr;
        hipFree(d_bsl); d_bsl = nullptr;
        aka_native_perf::throw_on_hip_error(
            hipStreamDestroy(stream), "hipStreamDestroy");
        stream = nullptr;
        return check_ok ? 0 : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << "native driver failed: " << error.what() << std::endl;
        if(d_q != nullptr) hipFree(d_q);
        if(d_o != nullptr) hipFree(d_o);
        if(d_kv != nullptr) hipFree(d_kv);
        if(d_r2t != nullptr) hipFree(d_r2t);
        if(d_bsl != nullptr) hipFree(d_bsl);
        if(stream != nullptr) hipStreamDestroy(stream);
        return 1;
    }
}
