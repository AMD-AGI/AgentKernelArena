// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Label-smoothed cross entropy with integer labels, class weights and an ignored label.
// One thread computes the loss of one row (all classes) into the caller's per-row losses;
// two reduction kernels then form the mean over the rows that are not ignored.
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include <cmath>
#include <cstdint>

constexpr int kReduceBlocks = 256;
constexpr int kReduceThreads = 256;

__global__ void cross_entropy_label_smoothing_kernel(
    const float* __restrict__ input,        // [outer, C]
    const int64_t* __restrict__ target_idx, // [outer]
    const float* __restrict__ weight,       // [C]
    float* __restrict__ out_loss,           // [outer]
    int32_t* __restrict__ out_valid,        // [outer]
    int64_t outer,
    int64_t C,
    int64_t ignore_index,
    float smooth_eps
) {
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= outer) return;

    const float* x = input + row * C;

    int64_t t_idx = target_idx[row];
    if (ignore_index >= 0 && t_idx == ignore_index) {
        out_loss[row] = 0.0f;
        out_valid[row] = 0;
        return;
    }

    // log-sum-exp
    float m = -INFINITY;
    for (int64_t c = 0; c < C; ++c) {
        float v = x[c];
        m = v > m ? v : m;
    }
    float s = 0.0f;
    for (int64_t c = 0; c < C; ++c) {
        s += expf(x[c] - m);
    }
    float lse = m + logf(s);

    // eps_nll / eps_sum formulation on the weighted log-probabilities
    float sum_lsm = 0.0f;
    float like = 0.0f;
    for (int64_t c = 0; c < C; ++c) {
        float lsm = (x[c] - lse) * weight[c];
        sum_lsm += lsm;
        if (c == t_idx) like = lsm;
    }
    float eps_sum = (smooth_eps > 0.0f) ? (smooth_eps / (float)C) : 0.0f;
    float eps_nll = 1.0f - eps_sum - smooth_eps;
    out_loss[row] = -(eps_nll * like + eps_sum * sum_lsm);
    out_valid[row] = 1;
}

// Stage 1: fixed grid of blocks, each reducing a grid-stride share of the rows.
__global__ void reduce_partials_kernel(
    const float* __restrict__ loss,
    const int32_t* __restrict__ valid,
    float* __restrict__ partial_loss,
    int32_t* __restrict__ partial_valid,
    int64_t outer
) {
    __shared__ float sloss[kReduceThreads];
    __shared__ int32_t svalid[kReduceThreads];
    float acc = 0.0f;
    int32_t cnt = 0;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < outer; i += stride) {
        acc += loss[i];
        cnt += valid[i];
    }
    sloss[threadIdx.x] = acc;
    svalid[threadIdx.x] = cnt;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sloss[threadIdx.x] += sloss[threadIdx.x + s];
            svalid[threadIdx.x] += svalid[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_loss[blockIdx.x] = sloss[0];
        partial_valid[blockIdx.x] = svalid[0];
    }
}

// Stage 2: one block combines the partials into the mean over valid rows (0 if none is valid).
__global__ void finalize_mean_kernel(
    const float* __restrict__ partial_loss,
    const int32_t* __restrict__ partial_valid,
    float* __restrict__ out,
    int nparts
) {
    __shared__ float sloss[kReduceThreads];
    __shared__ int32_t svalid[kReduceThreads];
    float acc = 0.0f;
    int32_t cnt = 0;
    for (int i = threadIdx.x; i < nparts; i += blockDim.x) {
        acc += partial_loss[i];
        cnt += partial_valid[i];
    }
    sloss[threadIdx.x] = acc;
    svalid[threadIdx.x] = cnt;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sloss[threadIdx.x] += sloss[threadIdx.x + s];
            svalid[threadIdx.x] += svalid[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = (svalid[0] > 0) ? sloss[0] / static_cast<float>(svalid[0]) : 0.0f;
    }
}

void run(
    const at::Tensor& x,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t ignore_index,
    double smooth_eps,
    at::Tensor& losses,
    at::Tensor& out
) {
    TORCH_CHECK(x.is_cuda() && target.is_cuda() && weight.is_cuda() && losses.is_cuda() && out.is_cuda(),
                "tensors must be on the device");
    TORCH_CHECK(x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat
                && losses.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat,
                "float32 tensors expected");
    TORCH_CHECK(target.scalar_type() == at::kLong, "target must be int64");
    TORCH_CHECK(x.dim() == 2 && target.dim() == 1 && target.size(0) == x.size(0), "x [rows, C], target [rows]");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1), "weight must have size C");
    TORCH_CHECK(x.is_contiguous() && target.is_contiguous() && weight.is_contiguous(),
                "contiguous tensors expected");
    TORCH_CHECK(losses.dim() == 1 && losses.size(0) == x.size(0) && losses.is_contiguous(),
                "losses must be a contiguous [rows] tensor");
    TORCH_CHECK(out.numel() == 1, "out must hold one element");

    const int64_t outer = x.size(0);
    const int64_t C = x.size(1);

    at::Tensor valid_mask = at::empty({outer}, x.options().dtype(at::kInt));
    at::Tensor partial_loss = at::empty({kReduceBlocks}, x.options());
    at::Tensor partial_valid = at::empty({kReduceBlocks}, x.options().dtype(at::kInt));

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    const int threads = 256;
    const int64_t blocks = (outer + threads - 1) / threads;
    hipLaunchKernelGGL(
        cross_entropy_label_smoothing_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        x.data_ptr<float>(),
        target.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        losses.data_ptr<float>(),
        valid_mask.data_ptr<int32_t>(),
        outer, C, ignore_index, static_cast<float>(smooth_eps));
    TORCH_CHECK(hipGetLastError() == hipSuccess, "row kernel launch failed");

    hipLaunchKernelGGL(
        reduce_partials_kernel,
        dim3(kReduceBlocks), dim3(kReduceThreads), 0, stream,
        losses.data_ptr<float>(),
        valid_mask.data_ptr<int32_t>(),
        partial_loss.data_ptr<float>(),
        partial_valid.data_ptr<int32_t>(),
        outer);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "partial reduction launch failed");

    hipLaunchKernelGGL(
        finalize_mean_kernel,
        dim3(1), dim3(kReduceThreads), 0, stream,
        partial_loss.data_ptr<float>(),
        partial_valid.data_ptr<int32_t>(),
        out.data_ptr<float>(),
        kReduceBlocks);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "finalize launch failed");
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor target, Tensor weight, int ignore_index, float smooth_eps, Tensor(a!) losses, Tensor(b!) out) -> ()", &run);
}
