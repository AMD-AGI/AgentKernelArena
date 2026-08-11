// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Canonical HIP graph-first timing for native task harnesses. Workspace setup
// copies this header next to each protected scripts/native benchmark driver.

#ifndef AKA_NATIVE_HIP_GRAPH_BENCHMARK_HPP_
#define AKA_NATIVE_HIP_GRAPH_BENCHMARK_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace aka_native_perf
{

struct benchmark_result
{
    double      execution_time_ms = 0.0;
    std::string benchmark_method;
    std::string benchmark_fallback_reason;
    std::size_t benchmark_effective_repeats = 1;
    std::size_t benchmark_samples           = 1;
};

using replay_validator
    = std::function<std::string(hipGraphExec_t graph_exec, hipStream_t stream)>;

inline std::string hip_failure(const char* operation, hipError_t error)
{
    std::ostringstream out;
    out << operation << ": " << hipGetErrorString(error) << " ("
        << static_cast<int>(error) << ")";
    return out.str();
}

inline void throw_on_hip_error(hipError_t error, const char* operation)
{
    if(error != hipSuccess)
    {
        throw std::runtime_error(hip_failure(operation, error));
    }
}

// Ordinary HIP-event timing. The start/stop events bracket a batch so fallback
// matches the legacy method while amortizing Python/subprocess work completely.
template<class Launch>
double measure_event_batch_ms(Launch& launch,
                              hipStream_t stream,
                              std::size_t launches,
                              std::size_t warmup)
{
    launches = std::max<std::size_t>(launches, 1);
    for(std::size_t i = 0; i < warmup; ++i)
    {
        throw_on_hip_error(launch(stream), "warmup launch");
    }
    throw_on_hip_error(hipStreamSynchronize(stream), "hipStreamSynchronize(warmup)");

    hipEvent_t start = nullptr;
    hipEvent_t stop  = nullptr;
    throw_on_hip_error(hipEventCreate(&start), "hipEventCreate(start)");
    hipError_t error = hipEventCreate(&stop);
    if(error != hipSuccess)
    {
        static_cast<void>(hipEventDestroy(start));
        throw std::runtime_error(hip_failure("hipEventCreate(stop)", error));
    }

    error = hipEventRecord(start, stream);
    for(std::size_t i = 0; error == hipSuccess && i < launches; ++i)
    {
        error = launch(stream);
    }
    if(error == hipSuccess)
    {
        error = hipEventRecord(stop, stream);
    }
    if(error == hipSuccess)
    {
        error = hipEventSynchronize(stop);
    }

    float elapsed_ms = 0.0f;
    if(error == hipSuccess)
    {
        error = hipEventElapsedTime(&elapsed_ms, start, stop);
    }
    static_cast<void>(hipEventDestroy(start));
    static_cast<void>(hipEventDestroy(stop));

    if(error != hipSuccess)
    {
        throw std::runtime_error(hip_failure("HIP event benchmark", error));
    }
    if(!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f)
    {
        throw std::runtime_error("HIP event benchmark returned a non-positive time");
    }
    return static_cast<double>(elapsed_ms);
}

// The callable receives the stream to use, must enqueue exactly one logical
// operation on it, and must return hipSuccess (or its dispatch error).
// Allocation, JIT, input preparation, and temporary-storage queries must
// happen before entering this function.
template<class Launch>
benchmark_result benchmark_graph_or_events(Launch&& launch,
                                           hipStream_t stream,
                                           std::size_t warmup      = 10,
                                           std::size_t samples     = 100,
                                           double target_ms        = 1.0,
                                           std::size_t max_repeats = 1000,
                                           const replay_validator& validate = {})
{
    samples     = std::max<std::size_t>(samples, 1);
    max_repeats = std::max<std::size_t>(max_repeats, 1);

    const char* force_event = std::getenv("AKA_BENCHMARK_FORCE_EVENT");
    if(force_event != nullptr && std::string(force_event) == "1")
    {
        benchmark_result forced_result;
        const double elapsed_ms
            = measure_event_batch_ms(launch, stream, samples, warmup);
        forced_result.execution_time_ms
            = elapsed_ms / static_cast<double>(samples);
        forced_result.benchmark_method          = "cuda_event_fallback";
        forced_result.benchmark_fallback_reason = "forced_event_baseline";
        forced_result.benchmark_effective_repeats = 1;
        forced_result.benchmark_samples           = samples;
        if(validate)
        {
            const std::string validation_failure = validate(nullptr, stream);
            if(!validation_failure.empty())
            {
                throw std::runtime_error("event output validation failed: "
                                         + validation_failure);
            }
        }
        return forced_result;
    }

    for(std::size_t i = 0; i < warmup; ++i)
    {
        throw_on_hip_error(launch(stream), "warmup launch");
    }
    throw_on_hip_error(hipStreamSynchronize(stream), "hipStreamSynchronize(warmup)");

    // Calibration is not a reported sample. It only sizes the captured block
    // so each replay contains enough device work to amortize one graph launch.
    constexpr std::size_t calibration_launches = 3;
    const double calibration_ms
        = measure_event_batch_ms(launch, stream, calibration_launches, 0);
    const double per_launch_ms = calibration_ms / calibration_launches;
    std::size_t graph_repeats  = 1;
    if(std::isfinite(per_launch_ms) && per_launch_ms > 0.0 && target_ms > per_launch_ms)
    {
        graph_repeats = static_cast<std::size_t>(std::ceil(target_ms / per_launch_ms));
    }
    graph_repeats = std::max<std::size_t>(1, std::min(graph_repeats, max_repeats));

    std::string graph_failure_reason;
    hipGraph_t graph          = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    hipStream_t graph_stream  = nullptr;

    // Capture on an isolated stream. On current ROCm runtimes, a prohibited
    // API can leave an invalidated capture stream unusable even after
    // hipStreamEndCapture reports failure. Keeping the caller's stream out of
    // capture guarantees that event fallback remains available.
    hipError_t error = hipStreamCreateWithFlags(&graph_stream, hipStreamNonBlocking);
    if(error != hipSuccess)
    {
        graph_failure_reason = hip_failure("hipStreamCreateWithFlags(graph)", error);
    }
    else
    {
        error = hipStreamBeginCapture(graph_stream, hipStreamCaptureModeThreadLocal);
        if(error != hipSuccess)
        {
            graph_failure_reason = hip_failure("hipStreamBeginCapture", error);
        }
        else
        {
            hipError_t launch_error = hipSuccess;
            for(std::size_t i = 0; i < graph_repeats && launch_error == hipSuccess; ++i)
            {
                launch_error = launch(graph_stream);
            }

            // Always request capture termination after a successful begin.
            // If ROCm leaves this stream invalidated, destroying the isolated
            // stream below still restores the thread's capture state.
            const hipError_t end_error = hipStreamEndCapture(graph_stream, &graph);
            if(launch_error != hipSuccess)
            {
                graph_failure_reason = hip_failure("captured launch", launch_error);
            }
            else if(end_error != hipSuccess)
            {
                graph_failure_reason = hip_failure("hipStreamEndCapture", end_error);
            }
        }
    }

    if(graph_failure_reason.empty())
    {
        std::size_t node_count = 0;
        error = hipGraphGetNodes(graph, nullptr, &node_count);
        if(error != hipSuccess)
        {
            graph_failure_reason = hip_failure("hipGraphGetNodes", error);
        }
        else if(node_count == 0)
        {
            graph_failure_reason = "captured graph contains no device work";
        }
    }

    if(graph_failure_reason.empty())
    {
        error = hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
        if(error != hipSuccess)
        {
            graph_failure_reason = hip_failure("hipGraphInstantiate", error);
        }
    }

    // Keep lazy graph-exec initialization outside the reported sample.
    if(graph_failure_reason.empty())
    {
        error = hipGraphLaunch(graph_exec, graph_stream);
        if(error == hipSuccess)
        {
            error = hipStreamSynchronize(graph_stream);
        }
        if(error != hipSuccess)
        {
            graph_failure_reason = hip_failure("hipGraphLaunch(warmup)", error);
        }
    }

    benchmark_result result;
    std::string validation_failure;
    result.benchmark_effective_repeats = graph_repeats;
    result.benchmark_samples           = samples;

    if(graph_failure_reason.empty())
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        error = hipEventCreate(&start);
        if(error == hipSuccess)
        {
            error = hipEventCreate(&stop);
        }
        if(error == hipSuccess)
        {
            error = hipEventRecord(start, graph_stream);
        }
        for(std::size_t i = 0; error == hipSuccess && i < samples; ++i)
        {
            error = hipGraphLaunch(graph_exec, graph_stream);
        }
        if(error == hipSuccess)
        {
            error = hipEventRecord(stop, graph_stream);
        }
        if(error == hipSuccess)
        {
            error = hipEventSynchronize(stop);
        }

        float elapsed_ms = 0.0f;
        if(error == hipSuccess)
        {
            error = hipEventElapsedTime(&elapsed_ms, start, stop);
        }
        if(start != nullptr)
        {
            static_cast<void>(hipEventDestroy(start));
        }
        if(stop != nullptr)
        {
            static_cast<void>(hipEventDestroy(stop));
        }

        if(error == hipSuccess && std::isfinite(elapsed_ms) && elapsed_ms > 0.0f)
        {
            result.execution_time_ms
                = static_cast<double>(elapsed_ms)
                  / static_cast<double>(samples * graph_repeats);
            // Preserve the method name used by Python benchmarks. On ROCm this
            // is backed by HIP Graph, the CUDA-compatible runtime API surface.
            result.benchmark_method = "cuda_graph";
        }
        else if(error != hipSuccess)
        {
            graph_failure_reason = hip_failure("hipGraphLaunch(timed)", error);
        }
        else
        {
            graph_failure_reason = "graph replay returned a non-positive time";
        }
    }

    // Validate an actual replay of the graph executable that produced the
    // measured samples while it is still alive. Drivers may poison outputs in
    // this callback, replay once, and compare against an eager/reference result.
    if(result.benchmark_method == "cuda_graph" && validate)
    {
        try
        {
            validation_failure = validate(graph_exec, graph_stream);
        }
        catch(const std::exception& validation_error)
        {
            validation_failure = validation_error.what();
        }
    }

    // A timed graph failure may occur after earlier replays were enqueued.
    // Drain the isolated graph stream before destroying it so those launches
    // cannot overlap the eager fallback on the caller stream and shared
    // buffers. Preserve the original graph failure as the reported reason.
    if(result.benchmark_method.empty() && graph_stream != nullptr)
    {
        static_cast<void>(hipStreamSynchronize(graph_stream));
        static_cast<void>(hipGetLastError());
    }

    if(graph_exec != nullptr)
    {
        static_cast<void>(hipGraphExecDestroy(graph_exec));
    }
    if(graph != nullptr)
    {
        static_cast<void>(hipGraphDestroy(graph));
    }
    if(graph_stream != nullptr)
    {
        static_cast<void>(hipStreamDestroy(graph_stream));
    }

    if(!validation_failure.empty())
    {
        throw std::runtime_error("graph replay output validation failed: "
                                 + validation_failure);
    }

    if(result.benchmark_method.empty())
    {
        static_cast<void>(hipGetLastError());
        throw_on_hip_error(hipStreamSynchronize(stream),
                           "hipStreamSynchronize(before event fallback)");
        // Match the paired AKA_BENCHMARK_FORCE_EVENT baseline exactly: both
        // sides time one eager launch per requested sample. Reusing the graph
        // batching factor here would label two materially different Event
        // protocols with the same benchmark_method.
        const std::size_t fallback_launches = samples;
        const double elapsed_ms
            = measure_event_batch_ms(launch, stream, fallback_launches, 0);
        result.execution_time_ms
            = elapsed_ms / static_cast<double>(fallback_launches);
        result.benchmark_method          = "cuda_event_fallback";
        result.benchmark_fallback_reason = "cuda_graph_failed: " + graph_failure_reason;
        result.benchmark_effective_repeats = 1;
        if(validate)
        {
            validation_failure = validate(nullptr, stream);
            if(!validation_failure.empty())
            {
                throw std::runtime_error("event output validation failed: "
                                         + validation_failure);
            }
        }
    }

    return result;
}

inline std::string json_escape(const std::string& input)
{
    std::ostringstream out;
    for(const unsigned char c : input)
    {
        switch(c)
        {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if(c < 0x20)
                {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(c) << std::dec;
                }
                else
                {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

// One line per case makes parsing independent of other program diagnostics.
inline void print_result_json(const std::string& test_case_id,
                              const benchmark_result& result,
                              double workload_bytes = 0.0)
{
    std::cout << "AKA_BENCHMARK_RESULT {"
              << "\"test_case_id\":\"" << json_escape(test_case_id) << "\","
              << "\"execution_time_ms\":" << std::setprecision(12)
              << result.execution_time_ms << ","
              << "\"benchmark_method\":\"" << result.benchmark_method << "\","
              << "\"benchmark_fallback_reason\":";
    if(result.benchmark_fallback_reason.empty())
    {
        std::cout << "null";
    }
    else
    {
        std::cout << "\"" << json_escape(result.benchmark_fallback_reason) << "\"";
    }
    std::cout << ",\"benchmark_effective_repeats\":"
              << result.benchmark_effective_repeats
              << ",\"benchmark_samples\":" << result.benchmark_samples;
    if(workload_bytes > 0.0)
    {
        const double throughput_gs
            = workload_bytes / (result.execution_time_ms * 1.0e6);
        std::cout << ",\"workload_bytes\":" << workload_bytes
                  << ",\"bytes_per_second_gs\":" << throughput_gs;
    }
    std::cout << "}" << std::endl;
}

} // namespace aka_native_perf

#endif // AKA_NATIVE_HIP_GRAPH_BENCHMARK_HPP_
