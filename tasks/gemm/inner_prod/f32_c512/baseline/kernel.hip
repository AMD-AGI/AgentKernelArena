// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// InnerProd forward: z[b,0,h,w] = sum_c (feat_img[b,0,c] * scale[c]) * feat_sound[b,c,h,w] + bias.
// One thread per output pixel; the channel dot product accumulates in double.
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
#include <cstdint>

static inline int64_t ceil_div_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

static inline dim3 make_grid_1d(int64_t n, int threads) {
    int64_t blocks = ceil_div_i64(n, threads);
    const int maxBlocks = 65535;
    if (blocks > maxBlocks) blocks = maxBlocks;
    return dim3(static_cast<unsigned int>(blocks));
}

__device__ inline void smem_load_scale(const float* __restrict__ scale, float* __restrict__ smem, int C) {
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        smem[c] = scale[c];
    }
    __syncthreads();
}

// -----------------------------
// Kernels
// -----------------------------
// forward: out[b,0,h,w] = sum_c feat_img[b,0,c] * scale[c] * feat_sound[b,c,h,w] + bias
__global__ void k_forward(
    const float* __restrict__ feat_img_bc,  // [B,C]
    const float* __restrict__ feat_sound,   // [B,C,H,W]
    const float* __restrict__ scale,        // [C]
    const float* __restrict__ bias_ptr,                      // scalar
    float* __restrict__ out,                // [B,1,H,W] (as [B,H,W])
    int B, int C, int H, int W)
{
    extern __shared__ float smem[]; // C floats
    smem_load_scale(scale, smem, C);
    const float bias = bias_ptr[0]; // Read in stream order on the GPU, including graph replay.

    const int64_t total = (int64_t)B * H * W;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)blockDim.x * gridDim.x) {
        int b  = (int)(linear / (H * (int64_t)W));
        int hw = (int)(linear % (H * (int64_t)W));
        int h  = hw / W;
        int w  = hw % W;

        const float* __restrict__ fi_b = feat_img_bc + (int64_t)b * C;            // [C]
        const float* __restrict__ fs_b = feat_sound  + (int64_t)b * C * H * W;    // [C,H,W]

        // Match the reference's float32 scaling, but avoid sequential FP32
        // reduction cancellation over 512 channels. Round the dot before bias.
        double acc = 0.0;
        const int64_t planeHW = (int64_t)H * W;
        const int64_t base_hw = (int64_t)h * W + w;

        int c = 0;
        int C4 = (C / 4) * 4;
        for (; c < C4; c += 4) {
            float fi0 = fi_b[c + 0];
            float fi1 = fi_b[c + 1];
            float fi2 = fi_b[c + 2];
            float fi3 = fi_b[c + 3];

            float sc0 = smem[c + 0];
            float sc1 = smem[c + 1];
            float sc2 = smem[c + 2];
            float sc3 = smem[c + 3];

            float fs0 = fs_b[(int64_t)(c + 0) * planeHW + base_hw];
            float fs1 = fs_b[(int64_t)(c + 1) * planeHW + base_hw];
            float fs2 = fs_b[(int64_t)(c + 2) * planeHW + base_hw];
            float fs3 = fs_b[(int64_t)(c + 3) * planeHW + base_hw];

            acc += static_cast<double>(fi0 * sc0) * static_cast<double>(fs0);
            acc += static_cast<double>(fi1 * sc1) * static_cast<double>(fs1);
            acc += static_cast<double>(fi2 * sc2) * static_cast<double>(fs2);
            acc += static_cast<double>(fi3 * sc3) * static_cast<double>(fs3);
        }
        for (; c < C; ++c) {
            float fi = fi_b[c];
            float sc = smem[c];
            float fs = fs_b[(int64_t)c * planeHW + base_hw];
            acc += static_cast<double>(fi * sc) * static_cast<double>(fs);
        }
        out[linear] = static_cast<float>(acc) + bias;
    }
}

// -----------------------------
// Host launcher
// -----------------------------
void run(const at::Tensor& feat_img,    // [B,1,C]
         const at::Tensor& feat_sound,  // [B,C,H,W]
         const at::Tensor& scale,       // [C]
         const at::Tensor& bias,        // [1]
         at::Tensor& out)               // [B,1,H,W]
{
    TORCH_CHECK(feat_img.is_contiguous() && feat_sound.is_contiguous() && scale.is_contiguous() &&
                    bias.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(feat_img.scalar_type() == at::kFloat && feat_sound.scalar_type() == at::kFloat &&
                    scale.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat &&
                    out.scalar_type() == at::kFloat,
                "float32 tensors expected");
    TORCH_CHECK(feat_img.dim() == 3 && feat_img.size(1) == 1, "feat_img must be [B,1,C]");
    TORCH_CHECK(feat_sound.dim() == 4, "feat_sound must be [B,C,H,W]");
    TORCH_CHECK(bias.numel() == 1, "bias must be shape [1]");

    const int64_t B = feat_sound.size(0);
    const int64_t C = feat_sound.size(1);
    const int64_t H = feat_sound.size(2);
    const int64_t W = feat_sound.size(3);
    TORCH_CHECK(feat_img.size(0) == B && feat_img.size(2) == C, "feat_img B and C must match feat_sound");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == C, "scale must be [C]");
    TORCH_CHECK(out.dim() == 4 && out.size(0) == B && out.size(1) == 1 && out.size(2) == H && out.size(3) == W,
                "out must be [B,1,H,W]");

    const int64_t total = B * H * W;
    if (total == 0) return;

    const int threads = 256;
    dim3 grid = make_grid_1d(total, threads);
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    size_t smem_bytes = static_cast<size_t>(C) * sizeof(float);
    hipLaunchKernelGGL(k_forward, grid, dim3(threads), smem_bytes, stream,
                       feat_img.data_ptr<float>(), feat_sound.data_ptr<float>(), scale.data_ptr<float>(),
                       bias.data_ptr<float>(), out.data_ptr<float>(), (int)B, (int)C, (int)H, (int)W);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "k_forward launch failed");
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor feat_img, Tensor feat_sound, Tensor scale, Tensor bias, Tensor(a!) out) -> ()", &run);
}
