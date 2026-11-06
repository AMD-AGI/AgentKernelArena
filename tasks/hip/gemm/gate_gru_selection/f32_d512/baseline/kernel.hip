// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// GRU-style gated selection on float32 rows, written for this task (task-original):
//   reset    = sigmoid([x1, x2] @ Wr^T + br)
//   update   = sigmoid([x1, x2] @ Wu^T + bu)
//   proposal = tanh([reset * x1, x2] @ Wp^T + bp)
//   out      = (1 - update) * x1 + update * proposal
// Two shared-memory tiled SGEMM kernels: the first computes both gates from one [x1, x2] tile,
// the second computes the proposal GEMM and applies the final blend in its epilogue.
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include <cmath>

constexpr int BM = 64;      // rows per block
constexpr int BN = 64;      // columns (model axis) per block
constexpr int BK = 16;      // reduction step over the 2*D concatenated axis
constexpr int TPB = 256;    // 16 x 16 threads, each owns a 4 x 4 output micro-tile

__device__ __forceinline__ float sigmoid_f(float v) { return 1.0f / (1.0f + expf(-v)); }

// Gates: reset and update from [x1, x2]; both weights are [D, 2D] row-major.
__global__ void gru_gates_kernel(
    const float* __restrict__ x1, const float* __restrict__ x2,
    const float* __restrict__ wr, const float* __restrict__ br,
    const float* __restrict__ wu, const float* __restrict__ bu,
    float* __restrict__ reset, float* __restrict__ update,
    int M, int D)
{
    __shared__ float As[BK][BM];
    __shared__ float Brs[BK][BN];
    __shared__ float Bus[BK][BN];

    const int K = 2 * D;
    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    float acc_r[4][4], acc_u[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) { acc_r[i][j] = 0.0f; acc_u[i][j] = 0.0f; }

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e = tid + TPB * i;
            const int row = e / BK;
            const int kk = e % BK;
            const int m = m0 + row;
            const int k = k0 + kk;
            float a = 0.0f;
            if (m < M) a = (k < D) ? x1[(long)m * D + k] : x2[(long)m * D + (k - D)];
            As[kk][row] = a;
            const int n = n0 + row;
            Brs[kk][row] = wr[(long)n * K + k];
            Bus[kk][row] = wu[(long)n * K + k];
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[4], br_[4], bu_[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) { br_[j] = Brs[kk][tx + 16 * j]; bu_[j] = Bus[kk][tx + 16 * j]; }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc_r[i][j] += a[i] * br_[j];
                    acc_u[i][j] += a[i] * bu_[j];
                }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty + 16 * i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx + 16 * j;
            reset[(long)m * D + n] = sigmoid_f(acc_r[i][j] + br[n]);
            update[(long)m * D + n] = sigmoid_f(acc_u[i][j] + bu[n]);
        }
    }
}

// Proposal GEMM over [reset * x1, x2] with the final blend fused into the epilogue.
__global__ void gru_proposal_kernel(
    const float* __restrict__ x1, const float* __restrict__ x2,
    const float* __restrict__ reset, const float* __restrict__ update,
    const float* __restrict__ wp, const float* __restrict__ bp,
    float* __restrict__ out, int M, int D)
{
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int K = 2 * D;
    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e = tid + TPB * i;
            const int row = e / BK;
            const int kk = e % BK;
            const int m = m0 + row;
            const int k = k0 + kk;
            float a = 0.0f;
            if (m < M) {
                a = (k < D) ? reset[(long)m * D + k] * x1[(long)m * D + k] : x2[(long)m * D + (k - D)];
            }
            As[kk][row] = a;
            Bs[kk][row] = wp[(long)(n0 + row) * K + k];
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty + 16 * i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx + 16 * j;
            const long idx = (long)m * D + n;
            const float p = tanhf(acc[i][j] + bp[n]);
            const float u = update[idx];
            out[idx] = (1.0f - u) * x1[idx] + u * p;
        }
    }
}

void run(const at::Tensor& x1, const at::Tensor& x2,
         const at::Tensor& reset_weight, const at::Tensor& reset_bias,
         const at::Tensor& update_weight, const at::Tensor& update_bias,
         const at::Tensor& proposal_weight, const at::Tensor& proposal_bias, at::Tensor& out)
{
    TORCH_CHECK(x1.is_contiguous() && x2.is_contiguous() && out.is_contiguous()
                && reset_weight.is_contiguous() && reset_bias.is_contiguous()
                && update_weight.is_contiguous() && update_bias.is_contiguous()
                && proposal_weight.is_contiguous() && proposal_bias.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(x1.scalar_type() == at::kFloat && x2.scalar_type() == at::kFloat
                && out.scalar_type() == at::kFloat, "float32 tensors expected");
    TORCH_CHECK(x1.dim() == 2 && x1.sizes() == x2.sizes(), "x1 and x2 must be [rows, d_model]");
    const int64_t M = x1.size(0);
    const int64_t D = x1.size(1);
    TORCH_CHECK(D % BN == 0 && D % BK == 0, "d_model must be a multiple of 64");
    TORCH_CHECK(reset_weight.size(0) == D && reset_weight.size(1) == 2 * D, "weights must be [D, 2D]");
    if (M == 0) return;

    auto reset = at::empty({M, D}, x1.options());
    auto update = at::empty({M, D}, x1.options());
    hipStream_t stream = c10::hip::getCurrentHIPStream();

    dim3 grid((unsigned)(D / BN), (unsigned)((M + BM - 1) / BM), 1);
    dim3 block(TPB);
    hipLaunchKernelGGL(
        gru_gates_kernel, grid, block, 0, stream,
        x1.data_ptr<float>(), x2.data_ptr<float>(),
        reset_weight.data_ptr<float>(), reset_bias.data_ptr<float>(),
        update_weight.data_ptr<float>(), update_bias.data_ptr<float>(),
        reset.data_ptr<float>(), update.data_ptr<float>(), (int)M, (int)D);
    hipLaunchKernelGGL(
        gru_proposal_kernel, grid, block, 0, stream,
        x1.data_ptr<float>(), x2.data_ptr<float>(),
        reset.data_ptr<float>(), update.data_ptr<float>(),
        proposal_weight.data_ptr<float>(), proposal_bias.data_ptr<float>(),
        out.data_ptr<float>(), (int)M, (int)D);
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x1, Tensor x2, Tensor reset_weight, Tensor reset_bias, Tensor update_weight, "
        "Tensor update_bias, Tensor proposal_weight, Tensor proposal_bias, Tensor(a!) out) -> ()", &run);
}
