// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Fused float32 MLP over vstack(x, y): out = sigmoid(relu(inp @ fc1_weight^T + fc1_bias) @ fc2_weight^T + fc2_bias).
// One thread owns one row of the stacked input and loops over the hidden units in tiles.
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include <cstdint>

constexpr int MAX_IN_FEATURES = 256;

__device__ __forceinline__ float sigmoid_f32(float x) { return 1.0f / (1.0f + ::expf(-x)); }

// x_flat: [rows_x, in_features], y_flat: [rows_y, in_features]
// fc1_weight: [hidden_size, in_features], fc1_bias: [hidden_size]
// fc2_weight: [out_features, hidden_size], fc2_bias: [out_features]
// out: [rows_x + rows_y, out_features]
template <int HTILE>
__global__ void fused_ff_tiled_kernel(const float* __restrict__ x_flat,
                                      const float* __restrict__ y_flat,
                                      const float* __restrict__ fc1_weight,
                                      const float* __restrict__ fc1_bias,
                                      const float* __restrict__ fc2_weight,
                                      const float* __restrict__ fc2_bias,
                                      float* __restrict__ out,
                                      int64_t rows_x,
                                      int64_t rows_y,
                                      int64_t in_features,
                                      int64_t hidden_size,
                                      int64_t out_features) {
    const int64_t total_rows = rows_x + rows_y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * gridDim.x;

    // Shared memory: fc1_bias [hidden_size], fc2_weight [out_features * hidden_size], fc2_bias [out_features].
    extern __shared__ unsigned char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    float* sm_b1 = smem;
    float* sm_w2 = sm_b1 + hidden_size;
    float* sm_b2 = sm_w2 + out_features * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) sm_b1[i] = fc1_bias[i];
    const int64_t w2_elems = out_features * hidden_size;
    for (int64_t i = threadIdx.x; i < w2_elems; i += blockDim.x) sm_w2[i] = fc2_weight[i];
    for (int i = threadIdx.x; i < out_features; i += blockDim.x) sm_b2[i] = fc2_bias[i];
    __syncthreads();

    for (int64_t row = tid; row < total_rows; row += nthreads) {
        const float* row_ptr = row < rows_x ? x_flat + row * in_features : y_flat + (row - rows_x) * in_features;

        float row_reg[MAX_IN_FEATURES];
        if ((in_features & 3) == 0) {
            const float4* rp4 = reinterpret_cast<const float4*>(row_ptr);
            const int n4 = in_features >> 2;
            for (int i = 0; i < n4; ++i) {
                const float4 v = rp4[i];
                const int base = i << 2;
                row_reg[base + 0] = v.x;
                row_reg[base + 1] = v.y;
                row_reg[base + 2] = v.z;
                row_reg[base + 3] = v.w;
            }
        } else {
            for (int k = 0; k < in_features; ++k) row_reg[k] = row_ptr[k];
        }

        for (int of = 0; of < out_features; ++of) {
            float out_acc = 0.0f;
            const float* w2_row = sm_w2 + of * hidden_size;

            for (int hbase = 0; hbase < hidden_size; hbase += HTILE) {
                const int tile = min(HTILE, static_cast<int>(hidden_size - hbase));
                float acc_h[HTILE];
#pragma unroll
                for (int t = 0; t < HTILE; ++t) {
                    if (t < tile) acc_h[t] = 0.0f;
                }
                for (int k = 0; k < in_features; ++k) {
                    const float xk = row_reg[k];
                    const float* w1_col = fc1_weight + hbase * in_features + k;
#pragma unroll
                    for (int t = 0; t < HTILE; ++t) {
                        if (t < tile) acc_h[t] += xk * w1_col[t * in_features];
                    }
                }
#pragma unroll
                for (int t = 0; t < HTILE; ++t) {
                    if (t < tile) {
                        acc_h[t] += sm_b1[hbase + t];
                        acc_h[t] = acc_h[t] > 0.0f ? acc_h[t] : 0.0f;
                        out_acc += acc_h[t] * w2_row[hbase + t];
                    }
                }
            }
            out_acc += sm_b2[of];
            out[row * out_features + of] = sigmoid_f32(out_acc);
        }
    }
}

void run(const at::Tensor& x, const at::Tensor& y, const at::Tensor& fc1_weight, const at::Tensor& fc1_bias,
         const at::Tensor& fc2_weight, const at::Tensor& fc2_bias, at::Tensor& out) {
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && fc1_weight.is_contiguous() && fc1_bias.is_contiguous() &&
                    fc2_weight.is_contiguous() && fc2_bias.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat &&
                    fc1_weight.scalar_type() == at::kFloat && fc1_bias.scalar_type() == at::kFloat &&
                    fc2_weight.scalar_type() == at::kFloat && fc2_bias.scalar_type() == at::kFloat &&
                    out.scalar_type() == at::kFloat,
                "float32 tensors expected");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == y.size(1), "x and y must be [rows, in_features]");

    const int64_t in_features = x.size(1);
    TORCH_CHECK(in_features <= MAX_IN_FEATURES, "in_features exceeds the register row cache");
    TORCH_CHECK(fc1_weight.dim() == 2 && fc1_weight.size(1) == in_features, "fc1_weight must be [hidden, in_features]");
    const int64_t hidden_size = fc1_weight.size(0);
    TORCH_CHECK(fc1_bias.size(0) == hidden_size, "fc1_bias size mismatch");
    TORCH_CHECK(fc2_weight.dim() == 2 && fc2_weight.size(1) == hidden_size, "fc2_weight must be [out, hidden]");
    const int64_t out_features = fc2_weight.size(0);
    TORCH_CHECK(fc2_bias.size(0) == out_features, "fc2_bias size mismatch");

    const int64_t nx = x.size(0);
    const int64_t ny = y.size(0);
    const int64_t batch = nx + ny;
    TORCH_CHECK(out.dim() == 2 && out.size(0) == batch && out.size(1) == out_features, "out must be [rows_x + rows_y, out]");
    if (batch == 0) return;

    const int threads = (batch >= 8192 ? 512 : 256);
    const int blocks = static_cast<int>((batch + threads - 1) / threads);
    const bool wide = hidden_size >= 128;
    const size_t smem_bytes = (hidden_size + out_features * hidden_size + out_features) * sizeof(float);
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    if (wide) {
        hipLaunchKernelGGL((fused_ff_tiled_kernel<32>), dim3(blocks), dim3(threads), smem_bytes, stream,
                           x.data_ptr<float>(), y.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                           fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(), fc2_bias.data_ptr<float>(),
                           out.data_ptr<float>(), nx, ny, in_features, hidden_size, out_features);
    } else {
        hipLaunchKernelGGL((fused_ff_tiled_kernel<16>), dim3(blocks), dim3(threads), smem_bytes, stream,
                           x.data_ptr<float>(), y.data_ptr<float>(), fc1_weight.data_ptr<float>(),
                           fc1_bias.data_ptr<float>(), fc2_weight.data_ptr<float>(), fc2_bias.data_ptr<float>(),
                           out.data_ptr<float>(), nx, ny, in_features, hidden_size, out_features);
    }
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor y, Tensor fc1_weight, Tensor fc1_bias, Tensor fc2_weight, Tensor fc2_bias, "
        "Tensor(a!) out) -> ()", &run);
}
