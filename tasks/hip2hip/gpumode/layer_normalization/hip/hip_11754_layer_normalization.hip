// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <cmath>
#include <cstdint>
#include <limits>

#ifndef WAVE_SIZE
#define WAVE_SIZE 64
#endif

// Round up to next multiple
static inline int round_up(int x, int mult) {
    return ((x + mult - 1) / mult) * mult;
}

// Next power of two
static inline int next_pow2_int(int x) {
    if (x <= 1) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// Accumulation type selector
template <typename T> struct AccTypeSel { using type = float; };
template <> struct AccTypeSel<double> { using type = double; };
template <> struct AccTypeSel<c10::Half> { using type = float; };
template <> struct AccTypeSel<c10::BFloat16> { using type = float; };

// Wavefront reduce sum with configurable subgroup width
template <typename T>
__device__ __forceinline__ T wave_sum_width(T v, int width) {
    // width must be a power of two and <= WAVE_SIZE
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, width);
    }
    return v;
}

// =============================
// Subgroup tiny-N kernel
// Each wavefront (64 lanes) is partitioned into subgroups of size sg (8/16/32),
// each subgroup processes one row independently using shuffle reductions of width=sg.
// No shared memory needed. Unbiased std.
// =============================

template <typename scalar_t, typename acc_t>
__global__ void layer_norm_subgroup_tinyN_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ output,
    int N,
    int M,
    acc_t epsilon,
    int subgroup)
{
    const int lanes = WAVE_SIZE;
    const int lane = threadIdx.x % lanes;
    const int wave = threadIdx.x / lanes;
    const int waves_per_block = blockDim.x / lanes;

    const int groups_per_wave = lanes / subgroup; // number of rows per wavefront processed concurrently
    const int group_id = lane / subgroup;         // which subgroup this lane belongs to [0..groups_per_wave-1]
    const int local_lane = lane % subgroup;       // lane index within subgroup [0..subgroup-1]

    // Each block processes waves_per_block * groups_per_wave rows at a time
    for (int base_row = (blockIdx.x * waves_per_block + wave) * groups_per_wave; base_row < M; base_row += gridDim.x * waves_per_block * groups_per_wave) {
        int row = base_row + group_id;
        if (row >= M) continue;

        const scalar_t* __restrict__ in_ptr  = input  + static_cast<size_t>(row) * N;
        scalar_t* __restrict__       out_ptr = output + static_cast<size_t>(row) * N;

        // Accumulate partial sum and sumsq using subgroup-strided loop
        acc_t sum = acc_t(0);
        acc_t sumsq = acc_t(0);
        for (int i = local_lane; i < N; i += subgroup) {
            acc_t v = static_cast<acc_t>(in_ptr[i]);
            sum += v;
            sumsq += v * v;
        }

        // Reduce within subgroup
        sum   = wave_sum_width(sum, subgroup);
        sumsq = wave_sum_width(sumsq, subgroup);

        // Broadcast subgroup totals from local_lane 0 of each subgroup
        acc_t sum_all   = __shfl(sum, 0, subgroup);
        acc_t sumsq_all = __shfl(sumsq, 0, subgroup);

        // Compute unbiased std and inv_denom
        acc_t mean;
        acc_t inv_denom;
        if (N > 1) {
            mean = sum_all / static_cast<acc_t>(N);
            acc_t var_unbiased = (sumsq_all - sum_all * mean) / static_cast<acc_t>(N - 1);
            if (var_unbiased < acc_t(0)) var_unbiased = acc_t(0);
            acc_t denom = sqrt(var_unbiased) + epsilon;
            inv_denom = acc_t(1) / denom;
        } else {
            // N == 1 -> std (unbiased) is NaN, propagate
            mean = sum_all; // single value
            inv_denom = acc_t(1) / (std::numeric_limits<acc_t>::quiet_NaN());
        }

        // Normalize and write back
        for (int i = local_lane; i < N; i += subgroup) {
            acc_t x = static_cast<acc_t>(in_ptr[i]);
            acc_t g = static_cast<acc_t>(gamma[i]);
            acc_t b = static_cast<acc_t>(beta[i]);
            acc_t y = g * ((x - mean) * inv_denom) + b;
            out_ptr[i] = static_cast<scalar_t>(y);
        }
    }
}

// =============================
// Fused small/medium-N kernel (cache each row in shared memory)
// One wavefront per row, multiple rows per block, unbiased std.
// =============================

template <typename scalar_t, typename acc_t>
__global__ void layer_norm_fused_smallN_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ output,
    int N,
    int M,
    acc_t epsilon)
{
    const int lanes = WAVE_SIZE;
    const int lane  = threadIdx.x % lanes;
    const int wave  = threadIdx.x / lanes; // wavefront id in block
    const int waves_per_block = blockDim.x / lanes;

    extern __shared__ unsigned char smem_raw[];
    acc_t* smem = reinterpret_cast<acc_t*>(smem_raw); // size: waves_per_block * N

    for (int base_row = blockIdx.x * waves_per_block; base_row < M; base_row += gridDim.x * waves_per_block) {
        int row = base_row + wave;
        if (row >= M) continue;

        const scalar_t* __restrict__ in_ptr  = input  + static_cast<size_t>(row) * N;
        scalar_t* __restrict__       out_ptr = output + static_cast<size_t>(row) * N;
        acc_t* row_buf = smem + static_cast<size_t>(wave) * N;

        // Load row, compute partial sums
        acc_t sum = acc_t(0);
        acc_t sumsq = acc_t(0);
        for (int i = lane; i < N; i += lanes) {
            acc_t v = static_cast<acc_t>(in_ptr[i]);
            row_buf[i] = v;
            sum += v;
            sumsq += v * v;
        }

        // Intra-wave reduction
        sum   = wave_sum_width(sum, WAVE_SIZE);
        sumsq = wave_sum_width(sumsq, WAVE_SIZE);

        // Broadcast totals within wave
        acc_t sum_all   = __shfl(sum, 0, WAVE_SIZE);
        acc_t sumsq_all = __shfl(sumsq, 0, WAVE_SIZE);

        acc_t mean;
        acc_t inv_denom;
        if (N > 1) {
            mean = sum_all / static_cast<acc_t>(N);
            acc_t var_unbiased = (sumsq_all - sum_all * mean) / static_cast<acc_t>(N - 1);
            if (var_unbiased < acc_t(0)) var_unbiased = acc_t(0);
            acc_t denom = sqrt(var_unbiased) + epsilon;
            inv_denom = acc_t(1) / denom;
        } else {
            mean = sum_all;
            inv_denom = acc_t(1) / (std::numeric_limits<acc_t>::quiet_NaN());
        }

        // Normalize and write
        for (int i = lane; i < N; i += lanes) {
            acc_t x = row_buf[i];
            acc_t g = static_cast<acc_t>(gamma[i]);
            acc_t b = static_cast<acc_t>(beta[i]);
            acc_t y = g * ((x - mean) * inv_denom) + b;
            out_ptr[i] = static_cast<scalar_t>(y);
        }
    }
}

// =============================
// Generic large-N kernel: one block per row, hybrid reduction (wave shuffles + tiny shared memory)
// =============================

template <typename scalar_t, typename acc_t>
__global__ void layer_norm_generic_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ output,
    int N,
    int M,
    acc_t epsilon)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const scalar_t* __restrict__ in_ptr  = input  + static_cast<size_t>(row) * N;
    scalar_t* __restrict__       out_ptr = output + static_cast<size_t>(row) * N;

    const int lanes = WAVE_SIZE;
    const int lane  = threadIdx.x % lanes;
    const int wave  = threadIdx.x / lanes;
    const int waves_per_block = blockDim.x / lanes;

    // Per-thread strided accumulation
    acc_t local_sum = acc_t(0);
    acc_t local_sumsq = acc_t(0);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        acc_t v = static_cast<acc_t>(in_ptr[i]);
        local_sum += v;
        local_sumsq += v * v;
    }

    // Intra-wave reductions
    local_sum   = wave_sum_width(local_sum, lanes);
    local_sumsq = wave_sum_width(local_sumsq, lanes);

    extern __shared__ unsigned char smem_raw[];
    acc_t* psum   = reinterpret_cast<acc_t*>(smem_raw);                 // waves_per_block
    acc_t* psumsq = psum + waves_per_block;                             // waves_per_block
    acc_t* scalars = psumsq + waves_per_block;                          // mean and inv_denom (2 values)

    if (lane == 0) {
        psum[wave] = local_sum;
        psumsq[wave] = local_sumsq;
    }
    __syncthreads();

    // Cross-wave reduction using first wave
    if (wave == 0) {
        acc_t tmp_sum = (lane < waves_per_block) ? psum[lane] : acc_t(0);
        acc_t tmp_sumsq = (lane < waves_per_block) ? psumsq[lane] : acc_t(0);
        tmp_sum   = wave_sum_width(tmp_sum, lanes);
        tmp_sumsq = wave_sum_width(tmp_sumsq, lanes);
        if (lane == 0) {
            if (N > 1) {
                acc_t mean = tmp_sum / static_cast<acc_t>(N);
                acc_t var_unbiased = (tmp_sumsq - tmp_sum * mean) / static_cast<acc_t>(N - 1);
                if (var_unbiased < acc_t(0)) var_unbiased = acc_t(0);
                acc_t denom = sqrt(var_unbiased) + epsilon;
                acc_t inv_denom = acc_t(1) / denom;
                scalars[0] = mean;
                scalars[1] = inv_denom;
            } else {
                scalars[0] = tmp_sum; // single value
                scalars[1] = acc_t(1) / (std::numeric_limits<acc_t>::quiet_NaN());
            }
        }
    }
    __syncthreads();

    acc_t mean = scalars[0];
    acc_t inv_denom = scalars[1];

    // Normalize and write
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        acc_t x = static_cast<acc_t>(in_ptr[i]);
        acc_t g = static_cast<acc_t>(gamma[i]);
        acc_t b = static_cast<acc_t>(beta[i]);
        acc_t y = g * ((x - mean) * inv_denom) + b;
        out_ptr[i] = static_cast<scalar_t>(y);
    }
}

// =============================
// Launcher
// =============================

torch::Tensor layer_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    double epsilon)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor on AMD ROCm");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma and beta must be CUDA/HIP tensors on AMD ROCm");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(gamma.is_contiguous() && beta.is_contiguous(), "gamma and beta must be contiguous");

    const int64_t N64 = input.size(-1);
    TORCH_CHECK(gamma.numel() == N64 && beta.numel() == N64, "gamma and beta must have shape (features,) matching the last dimension of input");

    const int64_t total = input.numel();
    TORCH_CHECK(N64 > 0 && (total % N64 == 0), "Input numel must be divisible by features and features > 0");
    const int64_t M64 = total / N64;
    TORCH_CHECK(M64 <= static_cast<int64_t>(std::numeric_limits<int>::max()), "Too many rows");

    const int N = static_cast<int>(N64);
    const int M = static_cast<int>(M64);

    auto output = torch::empty_like(input);

    // Heuristics
    int waves_per_block = 4; // default 4 wavefronts per block (256 threads)
    if (N <= 32)      waves_per_block = 6; // increase concurrency for tiny N
    else if (N <= 64) waves_per_block = 5;
    else              waves_per_block = 4;

    const int threads_default = waves_per_block * WAVE_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "layer_norm_forward", [&] {
        using scalar_t_ = scalar_t;
        using acc_t = typename AccTypeSel<scalar_t_>::type;

        // Path 1: subgroup tiny-N kernel when N <= 32
        if (N <= 32) {
            int subgroup = (N <= 8) ? 8 : (N <= 16 ? 16 : 32);
            int groups_per_wave = WAVE_SIZE / subgroup;
            int rows_per_block = waves_per_block * groups_per_wave;
            int blocks = (M + rows_per_block - 1) / rows_per_block;
            dim3 grid(blocks);
            dim3 block(waves_per_block * WAVE_SIZE);
            hipLaunchKernelGGL(
                (layer_norm_subgroup_tinyN_kernel<scalar_t_, acc_t>),
                grid, block, 0, 0,
                input.data_ptr<scalar_t_>(),
                gamma.data_ptr<scalar_t_>(),
                beta.data_ptr<scalar_t_>(),
                output.data_ptr<scalar_t_>(),
                N, M, static_cast<acc_t>(epsilon), subgroup
            );
            hipError_t err = hipGetLastError();
            TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed (subgroup tinyN): ", hipGetErrorString(err));
            return;
        }

        // Path 2: fused small/medium N with shared memory caching when feasible
        {
            int threads = threads_default;
            const int wpb = threads / WAVE_SIZE;
            size_t smem_needed = static_cast<size_t>(wpb) * static_cast<size_t>(N) * sizeof(acc_t);
            const size_t smem_limit = 48 * 1024; // conservative threshold
            bool use_fused = (N <= 2048) && (smem_needed <= smem_limit);
            if (use_fused) {
                int blocks = (M + wpb - 1) / wpb;
                dim3 grid(blocks);
                dim3 block(threads);
                hipLaunchKernelGGL(
                    (layer_norm_fused_smallN_kernel<scalar_t_, acc_t>),
                    grid, block, smem_needed, 0,
                    input.data_ptr<scalar_t_>(),
                    gamma.data_ptr<scalar_t_>(),
                    beta.data_ptr<scalar_t_>(),
                    output.data_ptr<scalar_t_>(),
                    N, M, static_cast<acc_t>(epsilon)
                );
                hipError_t err = hipGetLastError();
                TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed (fused): ", hipGetErrorString(err));
                return;
            }
        }

        // Path 3: generic large-N kernel
        {
            int tpb = next_pow2_int(N);
            if (tpb > 1024) tpb = 1024;
            tpb = round_up(tpb, WAVE_SIZE);
            if (tpb < WAVE_SIZE) tpb = WAVE_SIZE;
            int waves = tpb / WAVE_SIZE;
            size_t shmem_bytes = sizeof(acc_t) * (2 * waves + 2);
            dim3 grid(M);
            dim3 block(tpb);
            hipLaunchKernelGGL(
                (layer_norm_generic_kernel<scalar_t_, acc_t>),
                grid, block, shmem_bytes, 0,
                input.data_ptr<scalar_t_>(),
                gamma.data_ptr<scalar_t_>(),
                beta.data_ptr<scalar_t_>(),
                output.data_ptr<scalar_t_>(),
                N, M, static_cast<acc_t>(epsilon)
            );
            hipError_t err = hipGetLastError();
            TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed (generic): ", hipGetErrorString(err));
        }
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layer_norm_forward, "LayerNorm forward (HIP, ROCm, advanced optimized)");
}

