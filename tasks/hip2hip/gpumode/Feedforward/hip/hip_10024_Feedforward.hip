// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <type_traits>

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + ::expf(-x));
}

__device__ __forceinline__ double sigmoid_f64(double x) {
    return 1.0 / (1.0 + ::exp(-x));
}

template <typename scalar_t>
struct AccTypeSelector {
    using type = typename std::conditional<
        std::is_same<scalar_t, double>::value, double, float>::type;
};

// A single Wave64 computes 16 rows and 64 hidden units at a time. Four
// independent MFMA accumulators provide enough instruction-level parallelism
// to cover the FP32 MFMA latency. Both operands are staged in LDS because the
// native MFMA lane layout is strided in the original row-major tensors.
//
// The LDS skew is a permutation within each K row. It makes both the
// coalesced global-to-LDS transpose and the MFMA reads use all 32 LDS banks
// (two accesses per bank is the Wave64 minimum).
using float4_mfma = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int lds_skew(int k) {
    return ((k & 1) << 4) + ((k >> 1) & 15);
}

__device__ __forceinline__ float reduce_16(float v) {
    v += __shfl_down(v, 8, 16);
    v += __shfl_down(v, 4, 16);
    v += __shfl_down(v, 2, 16);
    v += __shfl_down(v, 1, 16);
    return v;
}

__global__ __launch_bounds__(64)
void fused_ff_mfma_f32_128x2048x1(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ w1,
    const float* __restrict__ b1,
    const float* __restrict__ w2,
    const float* __restrict__ b2,
    float* __restrict__ out,
    int64_t rows_x,
    int64_t rows_y)
{
    constexpr int K = 128;
    constexpr int ROWS = 16;
    constexpr int HTILE = 64;

    // x uses a 32-float physical row so its per-K skew never overlaps data.
    // w uses a modulo-64 skew, which is a permutation of each 64-value row.
    __shared__ float sx[K * 32];
    __shared__ float sw[K * HTILE];

    const int lane = static_cast<int>(threadIdx.x);
    const int row_base = static_cast<int>(blockIdx.x) * ROWS;
    const int64_t total_rows = rows_x + rows_y;

    // Load the 16x128 input tile once; it is reused for every hidden tile.
    for (int i = lane; i < ROWS * K; i += 64) {
        const int r = i >> 7;
        const int k = i & 127;
        const int64_t global_row = static_cast<int64_t>(row_base + r);
        float value = 0.0f;
        if (global_row < total_rows) {
            value = global_row < rows_x
                ? x[global_row * K + k]
                : y[(global_row - rows_x) * K + k];
        }
        sx[k * 32 + ((r + lds_skew(k)) & 31)] = value;
    }
    __builtin_amdgcn_wave_barrier();

    const int n = lane & 15;
    const int k_lane = lane >> 4;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    #pragma unroll 1
    for (int hbase = 0; hbase < 2048; hbase += HTILE) {
        // Coalesced global loads followed by a bank-swizzled transpose.
        for (int i = lane; i < HTILE * K; i += 64) {
            const int h = i >> 7;
            const int k = i & 127;
            const int physical_h = (h + lds_skew(k)) & 63;
            sw[k * HTILE + physical_h] = w1[(hbase + h) * K + k];
        }
        __builtin_amdgcn_wave_barrier();

        float4_mfma c0 = {0.0f, 0.0f, 0.0f, 0.0f};
        float4_mfma c1 = {0.0f, 0.0f, 0.0f, 0.0f};
        float4_mfma c2 = {0.0f, 0.0f, 0.0f, 0.0f};
        float4_mfma c3 = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll 4
        for (int kb = 0; kb < K; kb += 4) {
            const int k = kb + k_lane;
            const int skew = lds_skew(k);
            const float a = sx[k * 32 + ((n + skew) & 31)];
            const int woff = k * HTILE;
            c0 = __builtin_amdgcn_mfma_f32_16x16x4f32(
                a, sw[woff + ((n      + skew) & 63)], c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x4f32(
                a, sw[woff + ((n + 16 + skew) & 63)], c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x4f32(
                a, sw[woff + ((n + 32 + skew) & 63)], c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x4f32(
                a, sw[woff + ((n + 48 + skew) & 63)], c3, 0, 0, 0);
        }

        const float bias0 = b1[hbase + n];
        const float bias1 = b1[hbase + 16 + n];
        const float bias2 = b1[hbase + 32 + n];
        const float bias3 = b1[hbase + 48 + n];
        const float weight0 = w2[hbase + n];
        const float weight1 = w2[hbase + 16 + n];
        const float weight2 = w2[hbase + 32 + n];
        const float weight3 = w2[hbase + 48 + n];

        float p0 = fmaxf(c0[0] + bias0, 0.0f) * weight0
                 + fmaxf(c1[0] + bias1, 0.0f) * weight1
                 + fmaxf(c2[0] + bias2, 0.0f) * weight2
                 + fmaxf(c3[0] + bias3, 0.0f) * weight3;
        float p1 = fmaxf(c0[1] + bias0, 0.0f) * weight0
                 + fmaxf(c1[1] + bias1, 0.0f) * weight1
                 + fmaxf(c2[1] + bias2, 0.0f) * weight2
                 + fmaxf(c3[1] + bias3, 0.0f) * weight3;
        float p2 = fmaxf(c0[2] + bias0, 0.0f) * weight0
                 + fmaxf(c1[2] + bias1, 0.0f) * weight1
                 + fmaxf(c2[2] + bias2, 0.0f) * weight2
                 + fmaxf(c3[2] + bias3, 0.0f) * weight3;
        float p3 = fmaxf(c0[3] + bias0, 0.0f) * weight0
                 + fmaxf(c1[3] + bias1, 0.0f) * weight1
                 + fmaxf(c2[3] + bias2, 0.0f) * weight2
                 + fmaxf(c3[3] + bias3, 0.0f) * weight3;

        p0 = reduce_16(p0);
        p1 = reduce_16(p1);
        p2 = reduce_16(p2);
        p3 = reduce_16(p3);
        if (n == 0) {
            sum0 += p0;
            sum1 += p1;
            sum2 += p2;
            sum3 += p3;
        }
        __builtin_amdgcn_wave_barrier();
    }

    // Each 16-lane subgroup owns four rows of the MFMA output tile.
    if (n == 0) {
        const int r = row_base + k_lane * 4;
        const float output_bias = b2[0];
        if (static_cast<int64_t>(r) < total_rows)
            out[r] = sigmoid_f32(sum0 + output_bias);
        if (static_cast<int64_t>(r + 1) < total_rows)
            out[r + 1] = sigmoid_f32(sum1 + output_bias);
        if (static_cast<int64_t>(r + 2) < total_rows)
            out[r + 2] = sigmoid_f32(sum2 + output_bias);
        if (static_cast<int64_t>(r + 3) < total_rows)
            out[r + 3] = sigmoid_f32(sum3 + output_bias);
    }
}

// Compatibility path for other shapes and floating dtypes. The benchmarked
// 128x2048x1 FP32 case is dispatched to the MFMA kernel above.
template <typename scalar_t, int HTILE>
__global__ void fused_ff_tiled_kernel(
    const scalar_t* __restrict__ x_flat,
    const scalar_t* __restrict__ y_flat,
    const scalar_t* __restrict__ fc1_weight,
    const scalar_t* __restrict__ fc1_bias,
    const scalar_t* __restrict__ fc2_weight,
    const scalar_t* __restrict__ fc2_bias,
    scalar_t* __restrict__ out,
    int64_t rows_x,
    int64_t rows_y,
    int64_t in_features,
    int64_t hidden_size,
    int64_t out_features)
{
    using acc_t = typename AccTypeSelector<scalar_t>::type;
    const int64_t total_rows = rows_x + rows_y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * gridDim.x;

    extern __shared__ unsigned char smem_raw[];
    acc_t* sm_b1 = reinterpret_cast<acc_t*>(smem_raw);
    acc_t* sm_w2 = sm_b1 + hidden_size;
    acc_t* sm_b2 = sm_w2 + out_features * hidden_size;

    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x)
        sm_b1[i] = static_cast<acc_t>(fc1_bias[i]);
    for (int64_t i = threadIdx.x; i < out_features * hidden_size; i += blockDim.x)
        sm_w2[i] = static_cast<acc_t>(fc2_weight[i]);
    for (int64_t i = threadIdx.x; i < out_features; i += blockDim.x)
        sm_b2[i] = static_cast<acc_t>(fc2_bias[i]);
    __syncthreads();

    for (int64_t row = tid; row < total_rows; row += nthreads) {
        const scalar_t* row_ptr = row < rows_x
            ? x_flat + row * in_features
            : y_flat + (row - rows_x) * in_features;

        for (int of = 0; of < out_features; ++of) {
            acc_t out_acc = static_cast<acc_t>(0);
            const acc_t* w2_row = sm_w2 + of * hidden_size;
            for (int hbase = 0; hbase < hidden_size; hbase += HTILE) {
                const int tile = min(HTILE, static_cast<int>(hidden_size - hbase));
                acc_t acc_h[HTILE];
                #pragma unroll
                for (int t = 0; t < HTILE; ++t)
                    acc_h[t] = static_cast<acc_t>(0);

                for (int k = 0; k < in_features; ++k) {
                    const acc_t xk = static_cast<acc_t>(row_ptr[k]);
                    const scalar_t* wk = fc1_weight + hbase * in_features + k;
                    #pragma unroll
                    for (int t = 0; t < HTILE; ++t) {
                        if (t < tile)
                            acc_h[t] += xk * static_cast<acc_t>(wk[t * in_features]);
                    }
                }
                #pragma unroll
                for (int t = 0; t < HTILE; ++t) {
                    if (t < tile) {
                        const acc_t v = acc_h[t] + sm_b1[hbase + t];
                        out_acc += (v > static_cast<acc_t>(0) ? v : static_cast<acc_t>(0))
                                 * w2_row[hbase + t];
                    }
                }
            }
            out_acc += sm_b2[of];
            if constexpr (std::is_same<scalar_t, double>::value)
                out[row * out_features + of] = static_cast<scalar_t>(sigmoid_f64(out_acc));
            else
                out[row * out_features + of] = static_cast<scalar_t>(sigmoid_f32(static_cast<float>(out_acc)));
        }
    }
}

torch::Tensor feedforward_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_bias)
{
    TORCH_CHECK(x.device().is_cuda() && y.device().is_cuda(),
                "Inputs must be on HIP/CUDA device (ROCm)");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x and y must have the same dtype");
    TORCH_CHECK(x.dim() == y.dim(), "x and y must have the same number of dimensions");
    TORCH_CHECK(x.size(-1) == y.size(-1),
                "Last dimension (input_size) must match between x and y");

    const int64_t in_features = x.size(-1);
    TORCH_CHECK(fc1_weight.dim() == 2 && fc1_bias.dim() == 1,
                "fc1 weight/bias shapes invalid");
    TORCH_CHECK(fc1_weight.size(1) == in_features,
                "fc1_weight second dim must equal input_size");
    const int64_t hidden_size = fc1_weight.size(0);
    TORCH_CHECK(fc1_bias.size(0) == hidden_size, "fc1_bias size mismatch");

    TORCH_CHECK(fc2_weight.dim() == 2 && fc2_bias.dim() == 1,
                "fc2 weight/bias shapes invalid");
    TORCH_CHECK(fc2_weight.size(1) == hidden_size,
                "fc2_weight second dim must equal hidden_size");
    const int64_t out_features = fc2_weight.size(0);
    TORCH_CHECK(fc2_bias.size(0) == out_features, "fc2_bias size mismatch");

    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();
    const int64_t nx = x_contig.numel() / in_features;
    const int64_t ny = y_contig.numel() / in_features;
    const int64_t batch = nx + ny;
    auto out_2d = torch::empty({batch, out_features}, x.options());
    hipStream_t stream = at::hip::getCurrentHIPStream();

    if (x.scalar_type() == at::kFloat && in_features == 128 &&
        hidden_size == 2048 && out_features == 1) {
        const int blocks = static_cast<int>((batch + 15) / 16);
        hipLaunchKernelGGL(fused_ff_mfma_f32_128x2048x1,
                           dim3(blocks), dim3(64), 0, stream,
                           x_contig.data_ptr<float>(),
                           y_contig.data_ptr<float>(),
                           fc1_weight.data_ptr<float>(),
                           fc1_bias.data_ptr<float>(),
                           fc2_weight.data_ptr<float>(),
                           fc2_bias.data_ptr<float>(),
                           out_2d.data_ptr<float>(), nx, ny);
    } else {
        const int threads = batch >= 8192 ? 512 : 256;
        const int blocks = static_cast<int>((batch + threads - 1) / threads);
        const size_t smem_elems = hidden_size + out_features * hidden_size + out_features;
        const size_t smem_bytes = smem_elems *
            (x.element_size() >= 8 ? sizeof(double) : sizeof(float));
        const bool large_tile = hidden_size >= 128;

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_ff_tiled_kernel", ([&] {
            if (large_tile) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(fused_ff_tiled_kernel<scalar_t, 32>),
                    dim3(blocks), dim3(threads), smem_bytes, stream,
                    x_contig.data_ptr<scalar_t>(), y_contig.data_ptr<scalar_t>(),
                    fc1_weight.data_ptr<scalar_t>(), fc1_bias.data_ptr<scalar_t>(),
                    fc2_weight.data_ptr<scalar_t>(), fc2_bias.data_ptr<scalar_t>(),
                    out_2d.data_ptr<scalar_t>(), nx, ny, in_features,
                    hidden_size, out_features);
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(fused_ff_tiled_kernel<scalar_t, 16>),
                    dim3(blocks), dim3(threads), smem_bytes, stream,
                    x_contig.data_ptr<scalar_t>(), y_contig.data_ptr<scalar_t>(),
                    fc1_weight.data_ptr<scalar_t>(), fc1_bias.data_ptr<scalar_t>(),
                    fc2_weight.data_ptr<scalar_t>(), fc2_bias.data_ptr<scalar_t>(),
                    out_2d.data_ptr<scalar_t>(), nx, ny, in_features,
                    hidden_size, out_features);
            }
        }));
    }

    std::vector<int64_t> out_shape;
    out_shape.reserve(x.dim());
    out_shape.push_back(x.size(0) + y.size(0));
    for (int i = 1; i < x.dim() - 1; ++i)
        out_shape.push_back(x.size(i));
    out_shape.push_back(out_features);
    return out_2d.view(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &feedforward_forward, "Feedforward forward (HIP MFMA, ROCm)");
}
