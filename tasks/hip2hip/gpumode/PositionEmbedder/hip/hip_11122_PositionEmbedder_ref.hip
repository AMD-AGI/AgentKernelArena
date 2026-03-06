// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Helpers to convert to/from float for mixed-precision types on device
template <typename T>
__device__ __forceinline__ float to_float(T x) { return static_cast<float>(x); }

template <>
__device__ __forceinline__ float to_float<double>(double x) { return static_cast<float>(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x) { return static_cast<T>(x); }

template <>
__device__ __forceinline__ double from_float<double>(float x) { return static_cast<double>(x); }

// 2D kernel: each block handles one (s, b) tile and all embedding_dim elements
// Caches positional embedding vector for sequence index s in shared memory
// Shapes:
//   input/output: [batch_outer, seq_length, embedding_dim] (logical)
//   embedding:    [max_seq_length, embedding_dim]

template <typename scalar_t>
__global__ void add_positional_embedding_kernel_2d(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ embedding,
    scalar_t* __restrict__ output,
    long long batch_outer,
    long long seq_length,
    long long embedding_dim)
{
    long long s = static_cast<long long>(blockIdx.x);
    long long b = static_cast<long long>(blockIdx.y);
    if (s >= seq_length || b >= batch_outer) return;

    extern __shared__ unsigned char smem_raw[];
    scalar_t* pos_sh = reinterpret_cast<scalar_t*>(smem_raw);

    // Cooperative load of positional vector for this sequence index
    for (long long e = threadIdx.x; e < embedding_dim; e += blockDim.x) {
        pos_sh[e] = embedding[s * embedding_dim + e];
    }
    __syncthreads();

    long long base = (b * seq_length + s) * embedding_dim;

    // Compute and store
    for (long long e = threadIdx.x; e < embedding_dim; e += blockDim.x) {
        float a = to_float(input[base + e]);
        float p = to_float(pos_sh[e]);
        output[base + e] = from_float<scalar_t>(a + p);
    }
}

// 1D fallback kernel (no shared memory caching), flattens everything
// Useful if batch_outer is too large for grid.y limits

template <typename scalar_t>
__global__ void add_positional_embedding_kernel_1d(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ embedding,
    scalar_t* __restrict__ output,
    long long total_elems,
    long long seq_length,
    long long embedding_dim)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    long long e = idx % embedding_dim;
    long long s = (idx / embedding_dim) % seq_length;
    float a = to_float(input[idx]);
    float p = to_float(embedding[s * embedding_dim + e]);
    output[idx] = from_float<scalar_t>(a + p);
}

// Kernel launcher
// Note: keeps original function name and parameter order
torch::Tensor position_embedder_forward(
    torch::Tensor input,         // [..., S, E]
    torch::Tensor embedding      // [max_S, E]
) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(embedding.is_contiguous(), "Embedding must be contiguous");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dims [..., S, E]");
    TORCH_CHECK(embedding.dim() == 2, "Embedding must be 2D [max_S, E]");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA/HIP device");
    TORCH_CHECK(embedding.device().is_cuda(), "Embedding must be on CUDA/HIP device");
    TORCH_CHECK(input.scalar_type() == embedding.scalar_type(), "Input and Embedding must have the same dtype");

    const int64_t seq_length = input.size(input.dim() - 2);
    const int64_t embedding_dim = input.size(input.dim() - 1);
    TORCH_CHECK(embedding.size(1) == embedding_dim, "Embedding dim must match input's last dim (embedding_dim)");
    TORCH_CHECK(embedding.size(0) >= seq_length, "Embedding rows (max_sequence_length) must be >= input seq_length");

    // Flatten leading batch dims
    int64_t leading = 1;
    for (int i = 0; i < input.dim() - 2; ++i) leading *= input.size(i);
    const int64_t batch_outer = leading;

    auto output = torch::empty_like(input);

    // Decide launch config
    const int threads = static_cast<int>(std::min<int64_t>(embedding_dim, 256));

    // Grid.y must respect device limit; HIP typically allows up to 65535 per dim
    hipDeviceProp_t prop;
    int device_id = input.get_device();
    hipGetDeviceProperties(&prop, device_id);
    const int maxGridY = prop.maxGridSize[1];

    const bool use2D = (batch_outer <= maxGridY);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "add_positional_embedding_kernel", ([&] {
        if (use2D) {
            dim3 grid(static_cast<unsigned int>(seq_length), static_cast<unsigned int>(batch_outer), 1);
            dim3 block(static_cast<unsigned int>(threads), 1, 1);
            size_t shmem_bytes = static_cast<size_t>(embedding_dim) * sizeof(scalar_t);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(add_positional_embedding_kernel_2d<scalar_t>),
                grid, block, shmem_bytes, 0,
                input.data_ptr<scalar_t>(),
                embedding.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<long long>(batch_outer),
                static_cast<long long>(seq_length),
                static_cast<long long>(embedding_dim)
            );
        } else {
            // Fallback 1D kernel
            const long long total = static_cast<long long>(batch_outer) * static_cast<long long>(seq_length) * static_cast<long long>(embedding_dim);
            const int blocks = static_cast<int>((total + threads - 1) / threads);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(add_positional_embedding_kernel_1d<scalar_t>),
                dim3(blocks), dim3(threads), 0, 0,
                input.data_ptr<scalar_t>(),
                embedding.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                total,
                static_cast<long long>(seq_length),
                static_cast<long long>(embedding_dim)
            );
        }
        hipError_t err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
    }));

    return output;
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &position_embedder_forward, "PositionEmbedder forward (HIP)");
}

