// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <type_traits>

// Sigmoid helpers per dtype
__device__ inline float sigmoid_f32(float x) {
    return 1.0f / (1.0f + ::expf(-x));
}
__device__ inline double sigmoid_f64(double x) {
    return 1.0 / (1.0 + ::exp(-x));
}

template <typename scalar_t>
struct MathOps {
    using acc_t = scalar_t;
    __device__ static inline acc_t zero() { return static_cast<acc_t>(0); }
    __device__ static inline acc_t relu(acc_t x) { return x > zero() ? x : zero(); }
    __device__ static inline acc_t sigmoid(acc_t x) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            return static_cast<acc_t>(sigmoid_f32(static_cast<float>(x)));
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            return static_cast<acc_t>(sigmoid_f64(static_cast<double>(x)));
        } else {
            // Other types: compute in float
            float xf = static_cast<float>(x);
            float yf = sigmoid_f32(xf);
            return static_cast<acc_t>(yf);
        }
    }
};

// Promote half/other types to float accumulator for stability/performance
template <typename scalar_t>
struct AccTypeSelector {
    using type = typename std::conditional<
        std::is_same<scalar_t, double>::value,
        double,
        float>::type;
};

// Fused kernel with hidden tiling and register-cached row
// x_flat: [rows_x, in_features], y_flat: [rows_y, in_features]
// fc1_weight: [hidden_size, in_features], fc1_bias: [hidden_size]
// fc2_weight: [out_features, hidden_size], fc2_bias: [out_features]
// out: [rows_x+rows_y, out_features]
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

    // Shared memory: cache fc1_bias [hidden_size], fc2_weight [out_features*hidden_size], fc2_bias [out_features]
    extern __shared__ unsigned char smem_raw[];
    acc_t* smem = reinterpret_cast<acc_t*>(smem_raw);
    acc_t* sm_b1 = smem;
    acc_t* sm_w2 = sm_b1 + hidden_size;
    acc_t* sm_b2 = sm_w2 + out_features * hidden_size;

    // Cooperative load into shared memory (convert to acc_t)
    // Note: We skip caching fc1_weight to preserve occupancy (it can be large).
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sm_b1[i] = static_cast<acc_t>(fc1_bias[i]);
    }
    const int64_t w2_elems = out_features * hidden_size;
    for (int64_t i = threadIdx.x; i < w2_elems; i += blockDim.x) {
        sm_w2[i] = static_cast<acc_t>(fc2_weight[i]);
    }
    for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
        sm_b2[i] = static_cast<acc_t>(fc2_bias[i]);
    }
    __syncthreads();

    for (int64_t row = tid; row < total_rows; row += nthreads) {
        const scalar_t* row_ptr;
        if (row < rows_x) {
            row_ptr = x_flat + row * in_features;
        } else {
            row_ptr = y_flat + (row - rows_x) * in_features;
        }

        // Cache row into registers (acc_t)
        // Vectorized path for float and in_features % 4 == 0
        // Fallback scalar for other types
        acc_t row_reg_max[256]; // supports up to 256 features comfortably; typical input_size is small (e.g., 4)
        // Limit: ensure in_features <= 256 at runtime or fall back accordingly.
        // For generality, implement loop-based fill
        if constexpr (std::is_same<scalar_t, float>::value) {
            if ((in_features & 3) == 0) {
                const float4* rp4 = reinterpret_cast<const float4*>(row_ptr);
                int n4 = in_features >> 2;
                #pragma unroll
                for (int i = 0; i < n4; ++i) {
                    float4 v = rp4[i];
                    int base = (i << 2);
                    row_reg_max[base + 0] = static_cast<acc_t>(v.x);
                    row_reg_max[base + 1] = static_cast<acc_t>(v.y);
                    row_reg_max[base + 2] = static_cast<acc_t>(v.z);
                    row_reg_max[base + 3] = static_cast<acc_t>(v.w);
                }
            } else {
                for (int k = 0; k < in_features; ++k) {
                    row_reg_max[k] = static_cast<acc_t>(row_ptr[k]);
                }
            }
        } else {
            for (int k = 0; k < in_features; ++k) {
                row_reg_max[k] = static_cast<acc_t>(row_ptr[k]);
            }
        }

        // Accumulate output features (typically 1). We'll compute generically.
        // For out_features == 1, this reduces to a single accumulator.
        for (int of = 0; of < out_features; ++of) {
            acc_t out_acc = static_cast<acc_t>(0);
            const acc_t* w2_row = sm_w2 + of * hidden_size;

            // Tile over hidden neurons
            for (int hbase = 0; hbase < hidden_size; hbase += HTILE) {
                int tile = min(HTILE, static_cast<int>(hidden_size - hbase));
                acc_t acc_h[HTILE];
                #pragma unroll
                for (int t = 0; t < HTILE; ++t) {
                    if (t < tile) acc_h[t] = static_cast<acc_t>(0);
                }
                // Accumulate dot-products for the tile
                for (int k = 0; k < in_features; ++k) {
                    acc_t xk = row_reg_max[k];
                    // fc1_weight[h, k]
                    const scalar_t* w1_col = fc1_weight + hbase * in_features + k; // base at hbase, stride by in_features per h
                    #pragma unroll
                    for (int t = 0; t < HTILE; ++t) {
                        if (t < tile) {
                            acc_h[t] += xk * static_cast<acc_t>(w1_col[t * in_features]);
                        }
                    }
                }
                // Bias + ReLU, then accumulate into out
                #pragma unroll
                for (int t = 0; t < HTILE; ++t) {
                    if (t < tile) {
                        acc_h[t] += sm_b1[hbase + t];
                        // ReLU
                        acc_h[t] = acc_h[t] > static_cast<acc_t>(0) ? acc_h[t] : static_cast<acc_t>(0);
                        // Second linear accumulation
                        out_acc += acc_h[t] * w2_row[hbase + t];
                    }
                }
            }
            // Bias + Sigmoid
            out_acc += sm_b2[of];
            // Convert back to scalar_t for storage using MathOps<scalar_t>::sigmoid on cast
            // Compute sigmoid in appropriate precision:
            if constexpr (std::is_same<scalar_t, double>::value) {
                out[row * out_features + of] = static_cast<scalar_t>(sigmoid_f64(static_cast<double>(out_acc)));
            } else {
                out[row * out_features + of] = static_cast<scalar_t>(sigmoid_f32(static_cast<float>(out_acc)));
            }
        }
    }
}

// Forward function (HIP)
torch::Tensor feedforward_forward(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_bias)
{
    TORCH_CHECK(x.device().is_cuda() && y.device().is_cuda(), "Inputs must be on HIP/CUDA device (ROCm)");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x and y must have the same dtype");
    TORCH_CHECK(x.dim() == y.dim(), "x and y must have the same number of dimensions");
    TORCH_CHECK(x.size(-1) == y.size(-1), "Last dimension (input_size) must match between x and y");

    const int64_t in_features = x.size(-1);

    TORCH_CHECK(fc1_weight.dim() == 2 && fc1_bias.dim() == 1, "fc1 weight/bias shapes invalid");
    TORCH_CHECK(fc1_weight.size(1) == in_features, "fc1_weight second dim must equal input_size");
    const int64_t hidden_size = fc1_weight.size(0);
    TORCH_CHECK(fc1_bias.size(0) == hidden_size, "fc1_bias size mismatch");

    TORCH_CHECK(fc2_weight.dim() == 2 && fc2_bias.dim() == 1, "fc2 weight/bias shapes invalid");
    TORCH_CHECK(fc2_weight.size(1) == hidden_size, "fc2_weight second dim must equal hidden_size");
    const int64_t out_features = fc2_weight.size(0);
    TORCH_CHECK(fc2_bias.size(0) == out_features, "fc2_bias size mismatch");

    // Flatten all leading dims into batch for x and y
    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();

    int64_t nx = x_contig.numel() / in_features; // rows for x
    int64_t ny = y_contig.numel() / in_features; // rows for y

    auto x_flat = x_contig.view({nx, in_features});
    auto y_flat = y_contig.view({ny, in_features});

    const int64_t batch = nx + ny;
    auto out_2d = torch::empty({batch, out_features}, x.options());

    // Kernel config autotuning
    int threads = (batch >= 8192 ? 512 : 256);
    int blocks = (batch + threads - 1) / threads;
    int HTILE = (hidden_size >= 128 ? 32 : 16);

    // Shared memory size: fc1_bias[hidden_size] + fc2_weight[out_features*hidden_size] + fc2_bias[out_features]
    size_t smem_elems = hidden_size + out_features * hidden_size + out_features;
    size_t smem_bytes = smem_elems * (x.element_size() >= 8 ? sizeof(double) : sizeof(float)); // acc_t size

    hipStream_t stream = at::hip::getCurrentHIPStream();

    // Dispatch by dtype and HTILE
    auto launch = [&](auto scalar_dummy) {
        using scalar_t_ = decltype(scalar_dummy);
        if (HTILE == 32) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(fused_ff_tiled_kernel<scalar_t_, 32>),
                dim3(blocks), dim3(threads), smem_bytes, stream,
                x_flat.data_ptr<scalar_t_>(),
                y_flat.data_ptr<scalar_t_>(),
                fc1_weight.data_ptr<scalar_t_>(),
                fc1_bias.data_ptr<scalar_t_>(),
                fc2_weight.data_ptr<scalar_t_>(),
                fc2_bias.data_ptr<scalar_t_>(),
                out_2d.data_ptr<scalar_t_>(),
                nx, ny, in_features, hidden_size, out_features);
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(fused_ff_tiled_kernel<scalar_t_, 16>),
                dim3(blocks), dim3(threads), smem_bytes, stream,
                x_flat.data_ptr<scalar_t_>(),
                y_flat.data_ptr<scalar_t_>(),
                fc1_weight.data_ptr<scalar_t_>(),
                fc1_bias.data_ptr<scalar_t_>(),
                fc2_weight.data_ptr<scalar_t_>(),
                fc2_bias.data_ptr<scalar_t_>(),
                out_2d.data_ptr<scalar_t_>(),
                nx, ny, in_features, hidden_size, out_features);
        }
    };

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_ff_tiled_kernel", ([&] {
        launch(static_cast<scalar_t>(0));
    }));

    // Reshape back to [vstack(x,y).shape[:-1], out_features]
    std::vector<int64_t> out_shape;
    out_shape.reserve(x.dim());
    out_shape.push_back(x.size(0) + y.size(0));
    for (int i = 1; i < x.dim() - 1; ++i) {
        out_shape.push_back(x.size(i));
    }
    out_shape.push_back(out_features);

    auto output = out_2d.view(out_shape);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &feedforward_forward, "Feedforward forward (HIP fused tiled, ROCm)");
}

