// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "hip_graph_benchmark.hpp"
#include "matrix_reference.hpp"
#include "matrix_dense_inputs.hpp"

// Candidate preprocessor declarations stay in kernel_launch.hip. Neither the
// independent oracle nor benchmark identifiers share its preprocessing scope.
#include "matrix_launch.hpp"
#include <iostream>
#include <stdexcept>

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

std::uint64_t parse_seed(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == name) {
        std::size_t end = 0;
        const std::string text(argv[i + 1]);
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("Invalid unsigned input seed");
        const auto seed = std::stoull(text, &end);
        if (end != text.size()) throw std::invalid_argument("Invalid unsigned input seed");
        return seed;
    }
    return matrix_inputs::fresh_seed();
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
        const auto input_seed = parse_seed(argc, argv, "--input-seed");
        const auto replay_seed = parse_seed(argc, argv, "--replay-seed");
        matrix_inputs::fill(h_a, h_b, input_seed);
        std::vector<float> replay_a(h_a.size()), replay_b(h_b.size());
        matrix_inputs::fill(replay_a, replay_b, replay_seed);
        const auto inputs_record = matrix_inputs::record(input_seed, replay_seed, h_a, h_b, replay_a, replay_b);
        // Record reproducible draws even if a later native numerical check fails.
        std::cout << "AKA_MATRIX_INPUTS " << inputs_record << std::endl;
        const auto expected = matrix_reference::product(h_a, h_b, a_rows, a_cols, b_cols);
        const auto replay_expected = matrix_reference::product(replay_a, replay_b, a_rows, a_cols, b_cols);
        if (matrix_reference::validate(std::vector<float>(expected.begin(), expected.end()), replay_expected, b_cols).empty())
            throw std::runtime_error("Independent replay reference did not change at the original tolerance");

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
            return aka_matrix_launch(d_a, d_b, d_c, a_cols, grid, block,
                                     launch_stream);
        };

        auto mismatch_detail = [&](const char* phase, const std::vector<double>& reference) {
            double maximum = 0.;
            for (std::size_t index = 0; index < reference.size(); ++index) {
                if (!std::isfinite(h_c[index]) || !std::isfinite(reference[index]))
                    return std::string(phase) + ": nonfinite output/reference";
                maximum = std::max(maximum, std::fabs(static_cast<double>(h_c[index]) - reference[index]));
            }
            std::ostringstream out;
            out << phase << ": max_abs_error=" << std::setprecision(17) << maximum;
            return out.str();
        };
        bool validated = false;
        std::string replay_path;
        auto restore = [&](hipStream_t validation_stream) -> std::string {
            hipError_t error = hipMemcpyAsync(d_a, h_a.data(), a_bytes, hipMemcpyHostToDevice, validation_stream);
            const auto b_error = hipMemcpyAsync(d_b, h_b.data(), b_bytes, hipMemcpyHostToDevice, validation_stream);
            if (error == hipSuccess) error = b_error;
            const auto sync_error = hipStreamSynchronize(validation_stream);
            if (error == hipSuccess) error = sync_error;
            return error == hipSuccess ? std::string{} : aka_native_perf::hip_failure("matrix input restoration", error);
        };
        auto validate = [&](hipGraphExec_t graph_exec,
                            hipStream_t validation_stream) -> std::string {
            std::string failure;
            std::vector<float> read_a(h_a.size()), read_b(h_b.size());
            auto read_inputs_and_output = [&]() {
                aka_native_perf::throw_on_hip_error(hipMemcpyAsync(read_a.data(), d_a, a_bytes, hipMemcpyDeviceToHost, validation_stream), "readonly A snapshot");
                aka_native_perf::throw_on_hip_error(hipMemcpyAsync(read_b.data(), d_b, b_bytes, hipMemcpyDeviceToHost, validation_stream), "readonly B snapshot");
                aka_native_perf::throw_on_hip_error(hipMemcpyAsync(h_c.data(), d_c, c_bytes, hipMemcpyDeviceToHost, validation_stream), "full C snapshot");
                aka_native_perf::throw_on_hip_error(hipStreamSynchronize(validation_stream), "matrix validation synchronize");
            };
            try {
                // Validate the actual last measured output before any extra launch.
                read_inputs_and_output();
                if (!matrix_inputs::unchanged(read_a, h_a) || !matrix_inputs::unchanged(read_b, h_b))
                    throw std::runtime_error("Matrix readonly inputs changed during initial execution");
                failure = matrix_reference::validate(h_c, expected, b_cols);
                if (!failure.empty()) throw std::runtime_error(mismatch_detail("Initial matrix output mismatch", expected) + "; " + failure);
                aka_native_perf::throw_on_hip_error(hipMemcpyAsync(d_a, replay_a.data(), a_bytes, hipMemcpyHostToDevice, validation_stream), "independent replay A");
                aka_native_perf::throw_on_hip_error(hipMemcpyAsync(d_b, replay_b.data(), b_bytes, hipMemcpyHostToDevice, validation_stream), "independent replay B");
                aka_native_perf::throw_on_hip_error(hipMemsetAsync(d_c, 0xFF, c_bytes, validation_stream), "poison replay C");
                aka_native_perf::throw_on_hip_error(graph_exec != nullptr ? hipGraphLaunch(graph_exec, validation_stream) : launch(validation_stream), "changed-input exact replay");
                read_inputs_and_output();
                if (!matrix_inputs::unchanged(read_a, replay_a) || !matrix_inputs::unchanged(read_b, replay_b))
                    throw std::runtime_error("Matrix readonly inputs changed during replay");
                failure = matrix_reference::validate(h_c, replay_expected, b_cols);
                if (!failure.empty()) throw std::runtime_error(mismatch_detail("Independent replay matrix output mismatch", replay_expected) + "; " + failure);
            } catch (const std::exception& error) {
                failure = error.what();
            }
            const auto restoration = restore(validation_stream);
            if (!restoration.empty()) failure += (failure.empty() ? "" : "; ") + restoration;
            if (failure.empty()) {
                // Read back the restored bytes; successful copy return codes alone
                // do not establish that caller-owned initial inputs are intact.
                read_inputs_and_output();
                if (!matrix_inputs::unchanged(read_a, h_a) || !matrix_inputs::unchanged(read_b, h_b))
                    failure = "Matrix initial-input restoration mismatch";
            }
            validated = failure.empty();
            replay_path = graph_exec != nullptr ? "captured_graph" : "eager_callable";
            return failure;
        };
        auto validation_record = [&](const char* mode) {
            if (!validated) throw std::runtime_error("Matrix output/replay validation did not complete");
            std::cout << "AKA_MATRIX_VALIDATION {\"mode\":\"" << mode
                << "\",\"inputs\":" << inputs_record
                << ",\"initial_output_correctness\":\"PASS\",\"changed_input_replay_correctness\":\"PASS\","
                << "\"readonly_initial\":\"PASS\",\"readonly_replay\":\"PASS\",\"restored_initial_inputs\":\"PASS\","
                << "\"output_coverage\":\"all_output_elements\",\"replay_path\":\"" << replay_path << "\"}" << std::endl;
        };

        if(parse_uint(argc, argv, "--check-only", 0) != 0)
        {
            aka_native_perf::throw_on_hip_error(launch(stream), "initial dense correctness launch");
            const auto error = validate(nullptr, stream);
            if(!error.empty()) throw std::runtime_error(error);
            validation_record("check_only");
            std::cout << "Full reference validation passed" << std::endl;
        }
        else
        {
            const auto result = aka_native_perf::benchmark_graph_or_events(
                launch, stream, /*warmup=*/10, samples, /*target_ms=*/1.0,
                /*max_repeats=*/1000, validate);
            validation_record("performance");
            aka_native_perf::print_result_json("matrix_multiplication", result);
        }

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
