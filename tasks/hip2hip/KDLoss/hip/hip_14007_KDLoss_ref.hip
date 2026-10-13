// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
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

// Apply the final scalar normalization on device so forward() remains
// asynchronous and graph-capturable. Keep the arithmetic order identical to
// the former host-side expression.
__global__ void finalize_kd_loss_kernel(
    float* out_sum,
    float temp_factor,
    int N)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float loss = out_sum[0];
        loss = loss * temp_factor;
        loss = loss * temp_factor;
        loss = loss / static_cast<float>(N);
        out_sum[0] = loss;
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

    // Allocate the scalar output directly on device.
    auto out_sum = torch::zeros({}, input.options());

    int M = N * H * W; // blocks
    int max_threads = 256;
    int threads = next_pow2_int(C);
    if (threads > max_threads) threads = max_threads;
    if (threads < 32) threads = 32; // reasonable minimum

    size_t smem = threads * sizeof(float);
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    hipLaunchKernelGGL(
        fused_kd_loss_kernel,
        dim3(M), dim3(threads), smem, stream,
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        out_sum.data_ptr<float>(),
        N, C, H, W,
        temp_factor
    );
    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    hipLaunchKernelGGL(
        finalize_kd_loss_kernel,
        dim3(1), dim3(1), 0, stream,
        out_sum.data_ptr<float>(),
        temp_factor,
        N
    );
    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP finalization launch failed: ", hipGetErrorString(err));

    return out_sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "KD Loss forward (HIP, fused)");
}
