// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <c10/core/DeviceGuard.h>
#include <string>

// -----------------------------
// Utilities
// -----------------------------
static inline int64_t ceil_div_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

static inline void check_f32_contig_hip(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be on HIP device");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
}

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
    const float  bias,                      // scalar
    float* __restrict__ out,                // [B,1,H,W] (as [B,H,W])
    int B, int C, int H, int W)
{
    extern __shared__ float smem[]; // C floats
    smem_load_scale(scale, smem, C);

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

        float acc = 0.0f;
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

            acc += (fi0 * sc0) * fs0;
            acc += (fi1 * sc1) * fs1;
            acc += (fi2 * sc2) * fs2;
            acc += (fi3 * sc3) * fs3;
        }
        for (; c < C; ++c) {
            float fi = fi_b[c];
            float sc = smem[c];
            float fs = fs_b[(int64_t)c * planeHW + base_hw];
            acc += (fi * sc) * fs;
        }
        out[linear] = acc + bias;
    }
}

// forward_nosum: out[b,c,h,w] = (feat_img[b,0,c]*scale[c]) * feat_sound[b,c,h,w] + bias
__global__ void k_forward_nosum(
    const float* __restrict__ feat_img_bc,  // [B,C]
    const float* __restrict__ feat_sound,   // [B,C,H,W]
    const float* __restrict__ scale,        // [C]
    const float  bias,                      // scalar
    float* __restrict__ out,                // [B,C,H,W]
    int B, int C, int H, int W)
{
    extern __shared__ float smem[]; // C floats
    smem_load_scale(scale, smem, C);

    const int64_t total = (int64_t)B * C * H * W;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t t = idx;
        int w = (int)(t % W); t /= W;
        int h = (int)(t % H); t /= H;
        int c = (int)(t % C); t /= C;
        int b = (int)t;

        float a = feat_img_bc[(int64_t)b * C + c] * smem[c];
        int64_t base = (((int64_t)b * C + c) * H + h) * W + w;
        float val = a * feat_sound[base] + bias;
        out[base] = val;
    }
}

// forward_pixelwise: out[b,hi,wi,hs,ws] = sum_c feats_img[b,c,hi,wi]*scale[c]*feat_sound[b,c,hs,ws] + bias
__global__ void k_forward_pixelwise(
    const float* __restrict__ feats_img,   // [B,C,HI,WI]
    const float* __restrict__ feat_sound,  // [B,C,HS,WS]
    const float* __restrict__ scale,       // [C]
    const float  bias,                     // scalar
    float* __restrict__ out,               // [B,HI,WI,HS,WS]
    int B, int C, int HI, int WI, int HS, int WS)
{
    extern __shared__ float smem[]; // C floats
    smem_load_scale(scale, smem, C);

    const int64_t total = (int64_t)B * HI * WI * HS * WS;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t t = idx;
        int ws = (int)(t % WS); t /= WS;
        int hs = (int)(t % HS); t /= HS;
        int wi = (int)(t % WI); t /= WI;
        int hi = (int)(t % HI); t /= HI;
        int b  = (int)t;

        const float* __restrict__ fi_b = feats_img  + (int64_t)b * C * HI * WI;
        const float* __restrict__ fs_b = feat_sound + (int64_t)b * C * HS * WS;

        const int64_t img_hw = (int64_t)HI * WI;
        const int64_t snd_hw = (int64_t)HS * WS;
        const int64_t img_base = (int64_t)hi * WI + wi;
        const int64_t snd_base = (int64_t)hs * WS + ws;

        float acc = 0.0f;
        int c = 0;
        int C4 = (C / 4) * 4;
        for (; c < C4; c += 4) {
            float fi0 = fi_b[(int64_t)(c + 0) * img_hw + img_base];
            float fi1 = fi_b[(int64_t)(c + 1) * img_hw + img_base];
            float fi2 = fi_b[(int64_t)(c + 2) * img_hw + img_base];
            float fi3 = fi_b[(int64_t)(c + 3) * img_hw + img_base];

            float sc0 = smem[c + 0];
            float sc1 = smem[c + 1];
            float sc2 = smem[c + 2];
            float sc3 = smem[c + 3];

            float fs0 = fs_b[(int64_t)(c + 0) * snd_hw + snd_base];
            float fs1 = fs_b[(int64_t)(c + 1) * snd_hw + snd_base];
            float fs2 = fs_b[(int64_t)(c + 2) * snd_hw + snd_base];
            float fs3 = fs_b[(int64_t)(c + 3) * snd_hw + snd_base];

            acc += (fi0 * sc0) * fs0;
            acc += (fi1 * sc1) * fs1;
            acc += (fi2 * sc2) * fs2;
            acc += (fi3 * sc3) * fs3;
        }
        for (; c < C; ++c) {
            float fi = fi_b[(int64_t)c * img_hw + img_base];
            float sc = smem[c];
            float fs = fs_b[(int64_t)c * snd_hw + snd_base];
            acc += (fi * sc) * fs;
        }
        out[idx] = acc + bias;
    }
}

// -----------------------------
// Host implementations
// -----------------------------
static torch::Tensor impl_forward(
    torch::Tensor feat_img,    // [B,1,C]
    torch::Tensor feat_sound,  // [B,C,H,W]
    torch::Tensor scale,       // [C]
    torch::Tensor bias)        // [1]
{
    check_f32_contig_hip(feat_img,  "feat_img");
    check_f32_contig_hip(feat_sound, "feat_sound");
    check_f32_contig_hip(scale,     "scale");
    check_f32_contig_hip(bias,      "bias");

    TORCH_CHECK(bias.numel() == 1, "bias must be shape [1]");
    TORCH_CHECK(feat_img.dim() == 3 && feat_img.size(1) == 1, "feat_img must be [B,1,C]");
    TORCH_CHECK(feat_sound.dim() == 4, "feat_sound must be [B,C,H,W]");

    const int64_t B = feat_sound.size(0);
    const int64_t C = feat_sound.size(1);
    const int64_t H = feat_sound.size(2);
    const int64_t W = feat_sound.size(3);

    TORCH_CHECK(feat_img.size(0) == B && feat_img.size(2) == C, "feat_img B and C must match feat_sound");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == C, "scale must be [C]");

    at::DeviceGuard guard(feat_sound.device());

    auto out = torch::empty({B, 1, H, W}, feat_sound.options());

    // Views and raw pointers
    auto feat_img_bc = feat_img.view({B, C}).contiguous();

    const float* fi = feat_img_bc.data_ptr<float>();
    const float* fs = feat_sound.data_ptr<float>();
    const float* sc = scale.data_ptr<float>();
    const float  bi = bias.data_ptr<float>()[0];
    float*       oz = out.data_ptr<float>();

    const int threads = 256;
    const int64_t total = B * H * W;
    dim3 grid = make_grid_1d(total, threads);
    hipStream_t stream = at::hip::getCurrentHIPStream();

    size_t smem_bytes = static_cast<size_t>(C) * sizeof(float);
    hipLaunchKernelGGL(k_forward, grid, dim3(threads), smem_bytes, stream,
                       fi, fs, sc, bi, oz, (int)B, (int)C, (int)H, (int)W);

    C10_HIP_CHECK(hipGetLastError());
    return out;
}

static torch::Tensor impl_forward_nosum(
    torch::Tensor feat_img,    // [B,1,C]
    torch::Tensor feat_sound,  // [B,C,H,W]
    torch::Tensor scale,       // [C]
    torch::Tensor bias)        // [1]
{
    check_f32_contig_hip(feat_img,  "feat_img");
    check_f32_contig_hip(feat_sound, "feat_sound");
    check_f32_contig_hip(scale,     "scale");
    check_f32_contig_hip(bias,      "bias");

    TORCH_CHECK(bias.numel() == 1, "bias must be shape [1]");
    TORCH_CHECK(feat_img.dim() == 3 && feat_img.size(1) == 1, "feat_img must be [B,1,C]");
    TORCH_CHECK(feat_sound.dim() == 4, "feat_sound must be [B,C,H,W]");

    const int64_t B = feat_sound.size(0);
    const int64_t C = feat_sound.size(1);
    const int64_t H = feat_sound.size(2);
    const int64_t W = feat_sound.size(3);

    TORCH_CHECK(feat_img.size(0) == B && feat_img.size(2) == C, "feat_img B and C must match feat_sound");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == C, "scale must be [C]");

    at::DeviceGuard guard(feat_sound.device());

    auto out = torch::empty_like(feat_sound);

    auto feat_img_bc = feat_img.view({B, C}).contiguous();

    const float* fi = feat_img_bc.data_ptr<float>();
    const float* fs = feat_sound.data_ptr<float>();
    const float* sc = scale.data_ptr<float>();
    const float  bi = bias.data_ptr<float>()[0];
    float*       oz = out.data_ptr<float>();

    const int threads = 256;
    const int64_t total = B * C * H * W;
    dim3 grid = make_grid_1d(total, threads);
    hipStream_t stream = at::hip::getCurrentHIPStream();

    size_t smem_bytes = static_cast<size_t>(C) * sizeof(float);
    hipLaunchKernelGGL(k_forward_nosum, grid, dim3(threads), smem_bytes, stream,
                       fi, fs, sc, bi, oz, (int)B, (int)C, (int)H, (int)W);

    C10_HIP_CHECK(hipGetLastError());
    return out;
}

static torch::Tensor impl_forward_pixelwise(
    torch::Tensor feats_img,   // [B,C,HI,WI]
    torch::Tensor feat_sound,  // [B,C,HS,WS]
    torch::Tensor scale,       // [C]
    torch::Tensor bias)        // [1]
{
    check_f32_contig_hip(feats_img,  "feats_img");
    check_f32_contig_hip(feat_sound, "feat_sound");
    check_f32_contig_hip(scale,      "scale");
    check_f32_contig_hip(bias,       "bias");

    TORCH_CHECK(bias.numel() == 1, "bias must be shape [1]");
    TORCH_CHECK(feats_img.dim() == 4, "feats_img must be [B,C,HI,WI]");
    TORCH_CHECK(feat_sound.dim() == 4, "feat_sound must be [B,C,HS,WS]");

    const int64_t B  = feats_img.size(0);
    const int64_t C  = feats_img.size(1);
    const int64_t HI = feats_img.size(2);
    const int64_t WI = feats_img.size(3);

    TORCH_CHECK(feat_sound.size(0) == B && feat_sound.size(1) == C, "B and C must match between inputs");
    const int64_t HS = feat_sound.size(2);
    const int64_t WS = feat_sound.size(3);
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == C, "scale must be [C]");

    at::DeviceGuard guard(feats_img.device());

    auto out = torch::empty({B, HI, WI, HS, WS}, feats_img.options());

    const float* fi = feats_img.data_ptr<float>();
    const float* fs = feat_sound.data_ptr<float>();
    const float* sc = scale.data_ptr<float>();
    const float  bi = bias.data_ptr<float>()[0];
    float*       oz = out.data_ptr<float>();

    const int threads = 256;
    const int64_t total = B * HI * WI * HS * WS;
    dim3 grid = make_grid_1d(total, threads);
    hipStream_t stream = at::hip::getCurrentHIPStream();

    size_t smem_bytes = static_cast<size_t>(C) * sizeof(float);
    hipLaunchKernelGGL(k_forward_pixelwise, grid, dim3(threads), smem_bytes, stream,
                       fi, fs, sc, bi, oz, (int)B, (int)C, (int)HI, (int)WI, (int)HS, (int)WS);

    C10_HIP_CHECK(hipGetLastError());
    return out;
}

// -----------------------------
// Public API: single entry to match PyTorch module_fn(feat_img, feat_sound, scale, bias, mode)
// -----------------------------
torch::Tensor forward(
    torch::Tensor feat_img,
    torch::Tensor feat_sound,
    torch::Tensor scale,
    torch::Tensor bias,
    const std::string& mode = std::string("forward"))
{
    if (mode == std::string("forward")) {
        return impl_forward(feat_img, feat_sound, scale, bias);
    } else if (mode == std::string("forward_nosum")) {
        return impl_forward_nosum(feat_img, feat_sound, scale, bias);
    } else if (mode == std::string("forward_pixelwise")) {
        // For pixelwise, the first argument is feats_img by the Python API
        return impl_forward_pixelwise(feat_img, feat_sound, scale, bias);
    } else {
        TORCH_CHECK(false, "Unknown mode: ", mode);
    }
}

// Optional: direct bindings for individual modes (not required by the test but useful)
torch::Tensor forward_nosum(
    torch::Tensor feat_img,
    torch::Tensor feat_sound,
    torch::Tensor scale,
    torch::Tensor bias) {
    return impl_forward_nosum(feat_img, feat_sound, scale, bias);
}

torch::Tensor forward_pixelwise(
    torch::Tensor feats_img,
    torch::Tensor feat_sound,
    torch::Tensor scale,
    torch::Tensor bias) {
    return impl_forward_pixelwise(feats_img, feat_sound, scale, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, py::arg("feat_img"), py::arg("feat_sound"), py::arg("scale"), py::arg("bias"), py::arg("mode") = std::string("forward"),
          "InnerProd unified forward with mode (HIP)");
    m.def("forward_nosum", &forward_nosum, "InnerProd forward_nosum (HIP)");
    m.def("forward_pixelwise", &forward_pixelwise, "InnerProd forward_pixelwise (HIP)");
}

