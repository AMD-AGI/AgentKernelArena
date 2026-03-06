// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <cmath>

// Helpers
#define HIP_CHECK_ERRORS 0
inline void hipCheck(const char* msg) {
#if HIP_CHECK_ERRORS
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        TORCH_CHECK(false, msg, ": ", hipGetErrorString(err));
    }
#endif
}

template <typename T>
__device__ inline T my_exp(T x) { return exp(x); }

template <>
__device__ inline float my_exp<float>(float x) { return expf(x); }

// Fused 1x1 conv + layout transform for Query
// input:  [B, C_in, H, W]
// weight: [Cq, C_in, 1, 1]
// bias:   [Cq] or nullptr
// output: [B, HW, Cq]
// Each block handles one (b, hw_tile) pair; threads iterate over output channels
template <typename scalar_t>
__global__ void conv1x1_query_to_hw_c_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out_q,
    int B, int C_in, int H, int W, int Cq)
{
    int b = blockIdx.x;
    int HW = H * W;
    int hw = blockIdx.y;
    if (b >= B || hw >= HW) return;

    // Compute spatial indices
    int h = hw / W;
    int w = hw % W;

    // Base pointers
    const int in_base = b * C_in * H * W + h * W + w; // add c * H*W inside loop

    for (int co = threadIdx.x; co < Cq; co += blockDim.x) {
        scalar_t acc = bias ? bias[co] : static_cast<scalar_t>(0);
        const int w_base = co * C_in; // weight[co, ci]
        // dot over ci
        for (int ci = 0; ci < C_in; ++ci) {
            acc += input[in_base + ci * H * W] * weight[w_base + ci];
        }
        // write to [B, HW, Cq]
        out_q[(b * HW + hw) * Cq + co] = acc;
    }
}

// Fused 1x1 conv + layout transform for Key/Value to [B, C_out, HW]
// weight: [C_out, C_in, 1, 1]
// bias:   [C_out] or nullptr
// output: [B, C_out, HW]
// Grid: (B, C_out), threads iterate over HW
template <typename scalar_t>
__global__ void conv1x1_to_c_hw_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out_cv,
    int B, int C_in, int H, int W, int C_out)
{
    int b = blockIdx.x;
    int co = blockIdx.y;
    if (b >= B || co >= C_out) return;
    int HW = H * W;

    for (int idx = threadIdx.x; idx < HW; idx += blockDim.x) {
        int h = idx / W;
        int w = idx % W;
        scalar_t acc = bias ? bias[co] : static_cast<scalar_t>(0);
        const int in_base = b * C_in * H * W + h * W + w;
        const int w_base = co * C_in;
        for (int ci = 0; ci < C_in; ++ci) {
            acc += input[in_base + ci * H * W] * weight[w_base + ci];
        }
        out_cv[(b * C_out + co) * HW + idx] = acc; // [B, C_out, HW]
    }
}

// Tiled GEMM: energy[b] = query[b] @ key[b]
// query: [HW, Cq], key: [Cq, HW], energy: [HW, HW]
// block tile sizes
#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 16
#endif
#ifndef TILE_K
#define TILE_K 16
#endif

template <typename scalar_t>
__global__ void bmm_qk_tiled_kernel(
    const scalar_t* __restrict__ query, // [B, HW, Cq]
    const scalar_t* __restrict__ key,   // [B, Cq, HW]
    scalar_t* __restrict__ energy,      // [B, HW, HW]
    int B, int HW, int Cq)
{
    int b = blockIdx.z;
    if (b >= B) return;

    int row_tile = blockIdx.y; // along HW
    int col_tile = blockIdx.x; // along HW

    int row = row_tile * TILE_M + threadIdx.y;
    int col = col_tile * TILE_N + threadIdx.x;

    extern __shared__ unsigned char smem_[];
    scalar_t* As = reinterpret_cast<scalar_t*>(smem_);
    scalar_t* Bs = As + TILE_M * TILE_K;

    scalar_t acc = static_cast<scalar_t>(0);

    for (int tk = 0; tk < (Cq + TILE_K - 1) / TILE_K; ++tk) {
        int k0 = tk * TILE_K;
        // Load A tile: [TILE_M, TILE_K] from query[b, row, k]
        if (row < HW) {
            for (int kk = threadIdx.x; kk < TILE_K; kk += blockDim.x) {
                int k = k0 + kk;
                scalar_t v = (k < Cq) ? query[(b * HW + row) * Cq + k] : static_cast<scalar_t>(0);
                if (kk < TILE_K && threadIdx.y < TILE_M)
                    As[threadIdx.y * TILE_K + kk] = v;
            }
        }
        __syncthreads();
        // Load B tile: [TILE_K, TILE_N] from key[b, k, col]
        for (int kk = 0; kk < TILE_K; ++kk) {
            int k = k0 + kk;
            scalar_t bElem = static_cast<scalar_t>(0);
            if (k < Cq && col < HW && threadIdx.y == 0) {
                bElem = key[(b * Cq + k) * HW + col];
            }
            if (threadIdx.y == 0)
                Bs[kk * TILE_N + threadIdx.x] = bElem;
            __syncthreads();
            if (row < HW && col < HW) {
                acc += As[threadIdx.y * TILE_K + kk] * Bs[kk * TILE_N + threadIdx.x];
            }
            __syncthreads();
        }
    }

    if (row < HW && col < HW) {
        energy[(b * HW + row) * HW + col] = acc;
    }
}

// Row-wise softmax without storing temporary exp: two-pass over row
// energy: [B, HW, HW]
// Each block handles one (b, row) and iterates over columns
// We perform: sum = sum_j exp(energy[row,j]); then energy[row,j] = exp(energy[row,j]) / sum

template <typename scalar_t>
__global__ void rowwise_softmax_inplace_kernel(
    scalar_t* __restrict__ energy,
    int B, int HW)
{
    int b = blockIdx.x;
    int row = blockIdx.y;
    if (b >= B || row >= HW) return;

    int row_base = (b * HW + row) * HW;

    // Compute sum of exp
    scalar_t sum = static_cast<scalar_t>(0);
    for (int j = threadIdx.x; j < HW; j += blockDim.x) {
        scalar_t v = energy[row_base + j];
        sum += my_exp<scalar_t>(v);
    }
    // Reduce within block
    __shared__ scalar_t ssum[1024];
    ssum[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    scalar_t total = ssum[0];
    __syncthreads();

    // Normalize
    for (int j = threadIdx.x; j < HW; j += blockDim.x) {
        scalar_t v = my_exp<scalar_t>(energy[row_base + j]);
        energy[row_base + j] = (total == static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) : (v / total);
    }
}

// Tiled GEMM: out = value[b] @ energy[b]
// value:  [C, HW], energy: [HW, HW], out: [C, HW]
// Grid z=b, y=tile over C rows, x=tile over HW cols

template <typename scalar_t>
__global__ void bmm_vE_tiled_kernel(
    const scalar_t* __restrict__ value,   // [B, C, HW]
    const scalar_t* __restrict__ energy,  // [B, HW, HW]
    scalar_t* __restrict__ out,           // [B, C, HW]
    int B, int C, int HW)
{
    int b = blockIdx.z;
    if (b >= B) return;
    int row_tile = blockIdx.y; // over C
    int col_tile = blockIdx.x; // over HW

    int row = row_tile * TILE_M + threadIdx.y; // channel index
    int col = col_tile * TILE_N + threadIdx.x; // output spatial index

    extern __shared__ unsigned char smem_[];
    scalar_t* As = reinterpret_cast<scalar_t*>(smem_);
    scalar_t* Bs = As + TILE_M * TILE_K; // reuse TILE_K as chunk over HW

    scalar_t acc = static_cast<scalar_t>(0);

    for (int tk = 0; tk < (HW + TILE_K - 1) / TILE_K; ++tk) {
        int k0 = tk * TILE_K;
        // Load A: value[b, row, k]
        if (row < C) {
            for (int kk = threadIdx.x; kk < TILE_K; kk += blockDim.x) {
                int k = k0 + kk;
                scalar_t v = (k < HW) ? value[(b * C + row) * HW + k] : static_cast<scalar_t>(0);
                if (kk < TILE_K && threadIdx.y < TILE_M)
                    As[threadIdx.y * TILE_K + kk] = v;
            }
        }
        __syncthreads();
        // Load B: energy[b, k, col]
        for (int kk = 0; kk < TILE_K; ++kk) {
            int k = k0 + kk;
            scalar_t e = static_cast<scalar_t>(0);
            if (k < HW && col < HW && threadIdx.y == 0) {
                e = energy[(b * HW + k) * HW + col];
            }
            if (threadIdx.y == 0)
                Bs[kk * TILE_N + threadIdx.x] = e;
            __syncthreads();
            if (row < C && col < HW) {
                acc += As[threadIdx.y * TILE_K + kk] * Bs[kk * TILE_N + threadIdx.x];
            }
            __syncthreads();
        }
    }

    if (row < C && col < HW) {
        out[(b * C + row) * HW + col] = acc;
    }
}

// Gamma 1x1 conv on flattened [B, C, HW]
// gamma_w: [C_out(=C), C_in(=C)], gamma_b: [C]
// out_g[b, co, p] = sum_ci(out_bmm[b, ci, p] * gamma_w[co, ci]) + bias[co]

template <typename scalar_t>
__global__ void gamma_conv1x1_on_flat_kernel(
    const scalar_t* __restrict__ in_cv,    // [B, C, HW]
    const scalar_t* __restrict__ w_gamma,  // [C, C]
    const scalar_t* __restrict__ b_gamma,  // [C] or nullptr
    scalar_t* __restrict__ out_cv,         // [B, C, HW]
    int B, int C, int HW)
{
    int b = blockIdx.x;
    int co = blockIdx.y;
    if (b >= B || co >= C) return;

    for (int p = threadIdx.x; p < HW; p += blockDim.x) {
        scalar_t acc = b_gamma ? b_gamma[co] : static_cast<scalar_t>(0);
        const int in_base = b * C * HW + p; // add ci*HW inside loop
        const int w_base = co * C;
        for (int ci = 0; ci < C; ++ci) {
            acc += in_cv[in_base + ci * HW] * w_gamma[w_base + ci];
        }
        out_cv[(b * C + co) * HW + p] = acc;
    }
}

// Final reshape: [B, C, HW] -> [B, C, H, W]

template <typename scalar_t>
__global__ void reshape_flat_to_nchw_kernel(
    const scalar_t* __restrict__ in_cv,
    scalar_t* __restrict__ out_nchw,
    int B, int C, int H, int W)
{
    int b = blockIdx.x;
    int c = blockIdx.y;
    if (b >= B || c >= C) return;
    int HW = H * W;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        int h = i / W;
        int w = i % W;
        out_nchw[((b * C + c) * H + h) * W + w] = in_cv[(b * C + c) * HW + i];
    }
}

// Forward launcher
// Keep function name and parameters; add k to match PyTorch API but not used internally.
torch::Tensor normal_attention_embedded_gaussian_forward(
    torch::Tensor x,
    torch::Tensor query_weight,
    torch::Tensor query_bias,
    torch::Tensor key_weight,
    torch::Tensor key_bias,
    torch::Tensor value_weight,
    torch::Tensor value_bias,
    torch::Tensor gamma_weight,
    torch::Tensor gamma_bias,
    int64_t k)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
    TORCH_CHECK(query_weight.is_cuda() && key_weight.is_cuda() && value_weight.is_cuda() && gamma_weight.is_cuda(), "weights must be on device");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int HW = H * W;

    int Cq = query_weight.size(0); // C//k
    int Ck = key_weight.size(0);
    int Cv = value_weight.size(0); // should be C
    TORCH_CHECK(Cq == Ck, "Query and Key out channels must match");
    TORCH_CHECK(Cv == C, "Value out channels must equal input channels");
    TORCH_CHECK(gamma_weight.size(0) == C && gamma_weight.size(1) == C, "Gamma weight must be [C, C]");

    auto options = x.options();

    // 1. Query conv -> [B, HW, Cq]
    auto query_hw_c = torch::empty({B, HW, Cq}, options);
    {
        dim3 grid(B, HW);
        int threads = 128;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv1x1_query_to_hw_c", ([&] {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(conv1x1_query_to_hw_c_kernel<scalar_t>), grid, dim3(threads), 0, 0,
                x.data_ptr<scalar_t>(),
                query_weight.data_ptr<scalar_t>(),
                query_bias.defined() ? query_bias.data_ptr<scalar_t>() : nullptr,
                query_hw_c.data_ptr<scalar_t>(),
                B, C, H, W, Cq);
        }));
        hipCheck("conv1x1_query_to_hw_c");
    }

    // 2. Key conv -> [B, Cq, HW]
    auto key_c_hw = torch::empty({B, Cq, HW}, options);
    {
        dim3 grid(B, Cq);
        int threads = 128;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv1x1_key_to_c_hw", ([&] {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(conv1x1_to_c_hw_kernel<scalar_t>), grid, dim3(threads), 0, 0,
                x.data_ptr<scalar_t>(),
                key_weight.data_ptr<scalar_t>(),
                key_bias.defined() ? key_bias.data_ptr<scalar_t>() : nullptr,
                key_c_hw.data_ptr<scalar_t>(),
                B, C, H, W, Cq);
        }));
        hipCheck("conv1x1_key_to_c_hw");
    }

    // 3. Value conv -> [B, C, HW]
    auto value_c_hw = torch::empty({B, C, HW}, options);
    {
        dim3 grid(B, C);
        int threads = 128;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv1x1_value_to_c_hw", ([&] {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(conv1x1_to_c_hw_kernel<scalar_t>), grid, dim3(threads), 0, 0,
                x.data_ptr<scalar_t>(),
                value_weight.data_ptr<scalar_t>(),
                value_bias.defined() ? value_bias.data_ptr<scalar_t>() : nullptr,
                value_c_hw.data_ptr<scalar_t>(),
                B, C, H, W, C);
        }));
        hipCheck("conv1x1_value_to_c_hw");
    }

    // 4. energy = bmm(query, key): [B, HW, HW] with tiled GEMM
    auto energy = torch::empty({B, HW, HW}, options);
    {
        dim3 block(TILE_N, TILE_M, 1);
        dim3 grid((HW + TILE_N - 1) / TILE_N, (HW + TILE_M - 1) / TILE_M, B);
        size_t shmem = (TILE_M * TILE_K + TILE_K * TILE_N) * torch::elementSize(x.scalar_type());
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "bmm_qk_tiled", ([&] {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(bmm_qk_tiled_kernel<scalar_t>), grid, block, shmem, 0,
                query_hw_c.data_ptr<scalar_t>(),
                key_c_hw.data_ptr<scalar_t>(),
                energy.data_ptr<scalar_t>(),
                B, HW, Cq);
        }));
        hipCheck("bmm_qk_tiled");
    }

    // 5. Row-wise exp normalize (softmax across last dim)
    {
        dim3 grid(B, HW);
        int threads = 256;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rowwise_softmax_inplace", ([&] {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rowwise_softmax_inplace_kernel<scalar_t>), grid, dim3(threads), 0, 0,
                energy.data_ptr<scalar_t>(), B, HW);
        }));
        hipCheck("rowwise_softmax_inplace");
    }

    // 6. out_bmm = bmm(value, energy): [B, C, HW]
    auto out_c_hw = torch::empty({B, C, HW}, options);
    {
        dim3 block(TILE_N, TILE_M, 1);
        dim3 grid((HW + TILE_N - 1) / TILE_N, (C + TILE_M - 1) / TILE_M, B);
        size_t shmem = (TILE_M * TILE_K + TILE_K * TILE_N) * torch::elementSize(x.scalar_type());
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "bmm_vE_tiled", ([&] {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(bmm_vE_tiled_kernel<scalar_t>), grid, block, shmem, 0,
                value_c_hw.data_ptr<scalar_t>(),
                energy.data_ptr<scalar_t>(),
                out_c_hw.data_ptr<scalar_t>(),
                B, C, HW);
        }));
        hipCheck("bmm_vE_tiled");
    }

    // 7. gamma conv1x1 on flattened [B, C, HW]
    auto out_gamma_c_hw = torch::empty({B, C, HW}, options);
    {
        dim3 grid(B, C);
        int threads = 128;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gamma_conv1x1_on_flat", ([&] {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gamma_conv1x1_on_flat_kernel<scalar_t>), grid, dim3(threads), 0, 0,
                out_c_hw.data_ptr<scalar_t>(),
                gamma_weight.data_ptr<scalar_t>(),
                gamma_bias.defined() ? gamma_bias.data_ptr<scalar_t>() : nullptr,
                out_gamma_c_hw.data_ptr<scalar_t>(),
                B, C, HW);
        }));
        hipCheck("gamma_conv1x1_on_flat");
    }

    // 8. reshape back to [B, C, H, W]
    auto out_final = torch::empty({B, C, H, W}, options);
    {
        dim3 grid(B, C);
        int threads = 128;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "reshape_flat_to_nchw", ([&] {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(reshape_flat_to_nchw_kernel<scalar_t>), grid, dim3(threads), 0, 0,
                out_gamma_c_hw.data_ptr<scalar_t>(),
                out_final.data_ptr<scalar_t>(),
                B, C, H, W);
        }));
        hipCheck("reshape_flat_to_nchw");
    }

    return out_final;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &normal_attention_embedded_gaussian_forward,
          "NormalAttention_embedded_gaussian forward (HIP)");
}

