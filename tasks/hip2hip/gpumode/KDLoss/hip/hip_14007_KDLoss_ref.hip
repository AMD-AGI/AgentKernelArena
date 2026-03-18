// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cfloat>

// Utility: next power of two up to a cap
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

// Numerically safe xlogy for floats: x * log(y), with 0*log(0)=0 and 0*log(any)=0
__device__ __forceinline__ float xlogy_safe(float x, float y) {
    if (x == 0.0f) return 0.0f;
    return x * logf(y);
}

// Fused kernel: For each slice (n,h,w), compute log_softmax over C and accumulate KLDiv contribution into a global scalar (reduction='sum').
// Input/target shapes: [N, C, H, W] contiguous, dtype float32
__global__ void fused_kd_loss_kernel(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ out_sum, // single-element buffer on device
    int N, int C, int H, int W,
    float temp)
{
    extern __shared__ float sdata[]; // used for reductions
    float* sreduce = sdata;          // shared buffer size = blockDim.x floats

    int M = N * H * W;               // number of slices
    int m = blockIdx.x;              // one block per slice
    if (m >= M) return;

    // Derive (n, h, w) from m
    int HW = H * W;
    int n = m / HW;
    int hw = m - n * HW;
    int h = hw / W;
    int w = hw - h * W;

    // Strides for [N,C,H,W] contiguous layout
    int strideN = C * H * W;
    int strideC = H * W;
    int strideH = W;
    int strideW = 1;

    // Base offset for c=0 at (n,h,w)
    int base = n * strideN + h * strideH + w * strideW;

    // 1) Compute max over c of (input/temp) for numerical stability
    float thread_max = -INFINITY;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = input[base + c * strideC] / temp;
        thread_max = fmaxf(thread_max, v);
    }
    // Reduce max within block
    sreduce[threadIdx.x] = thread_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreduce[threadIdx.x] = fmaxf(sreduce[threadIdx.x], sreduce[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float max_val = sreduce[0];

    // 2) Compute sum_exp = sum_c exp((input/temp) - max)
    float thread_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = input[base + c * strideC] / temp;
        thread_sum += expf(v - max_val);
    }
    sreduce[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreduce[threadIdx.x] += sreduce[threadIdx.x + s];
        }
        __syncthreads();
    }
    float log_sum_exp = max_val + logf(fmaxf(sreduce[0], FLT_MIN)); // guard

    // 3) Accumulate KLDiv contribution for this slice: sum_c t*(log(t) - log_softmax)
    float thread_klsum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float t = target[base + c * strideC];
        float v = input[base + c * strideC] / temp;
        float log_soft = v - log_sum_exp; // log_softmax
        // t*log(t) with safe 0 handling
        float tlogt = xlogy_safe(t, t);
        thread_klsum += (tlogt - t * log_soft);
    }
    sreduce[threadIdx.x] = thread_klsum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreduce[threadIdx.x] += sreduce[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out_sum, sreduce[0]);
    }
}

// Launcher: matches PyTorch kd_loss_fn semantics and returns a scalar tensor
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor target,
    float temp_factor
) {
    TORCH_CHECK(input.is_cuda(), "input must be a HIP/CUDA tensor on device");
    TORCH_CHECK(target.is_cuda(), "target must be a HIP/CUDA tensor on device");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(target.scalar_type() == at::kFloat, "Only float32 supported for target");
    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have the same shape");
    TORCH_CHECK(input.dim() == 4, "Expected input/target to be 4D [N,C,H,W]");

    // Ensure contiguous for predictable strides
    input = input.contiguous();
    target = target.contiguous();

    const auto sizes = input.sizes();
    int N = static_cast<int>(sizes[0]);
    int C = static_cast<int>(sizes[1]);
    int H = static_cast<int>(sizes[2]);
    int W = static_cast<int>(sizes[3]);

    // Allocate device scalar for sum
    auto out_sum = torch::zeros({1}, input.options());

    int M = N * H * W; // blocks
    int max_threads = 256;
    int threads = next_pow2_int(C);
    if (threads > max_threads) threads = max_threads;
    if (threads < 32) threads = 32; // reasonable minimum

    size_t smem = threads * sizeof(float);

    hipLaunchKernelGGL(
        fused_kd_loss_kernel,
        dim3(M), dim3(threads), smem, 0,
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        out_sum.data_ptr<float>(),
        N, C, H, W,
        temp_factor
    );

    hipError_t err = hipDeviceSynchronize();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    // Finish loss: loss = sum * temp^2 / N
    float sum_val = out_sum.item<float>();
    float loss_val = sum_val * temp_factor * temp_factor / static_cast<float>(N);

    // Return scalar tensor on device with same dtype/options
    auto loss_tensor = torch::empty({}, input.options());
    // Write via CPU then copy or directly set by making a scalar on device
    // Use at::full to create device scalar
    loss_tensor = at::full({}, loss_val, input.options());
    return loss_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "KD Loss forward (HIP, fused)");
}

