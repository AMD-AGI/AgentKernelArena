// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Tile sizes tuned for AMD wavefronts: 16x16 threads per block (256 threads)
#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 16
#endif
#ifndef TILE_K
#define TILE_K 16
#endif

// Kernel: Batched tiled GEMM computing C = A @ (B + B)
// A: [B, M, K]
// B: [B, K, N]
// C: [B, M, N]
// We fuse (B + B) by scaling loaded B tiles by 2.0f into shared memory.
__global__ void batched_gemm_bplusb_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int Batches, int M, int K, int N)
{
    int batch = blockIdx.z;
    int tile_m = blockIdx.y; // tile row index for M
    int tile_n = blockIdx.x; // tile col index for N

    int local_m = threadIdx.y; // [0, TILE_M)
    int local_n = threadIdx.x; // [0, TILE_N)

    int row = tile_m * TILE_M + local_m; // global row in M
    int col = tile_n * TILE_N + local_n; // global col in N

    // Shared memory tiles
    __shared__ float Asub[TILE_M][TILE_K];
    __shared__ float Bsub[TILE_K][TILE_N];

    // Base offsets per batch
    const float* Ab = A + (size_t)batch * M * K;
    const float* Bb = B + (size_t)batch * K * N;
    float* Cb = C + (size_t)batch * M * N;

    float acc = 0.0f;

    // Loop over K dimension in TILE_K chunks
    for (int kt = 0; kt < K; kt += TILE_K) {
        // Load A tile [row, kt + *]
        int a_col = kt + local_n; // reuse local_n for A col load into Asub[row][n]
        if (row < M && a_col < K) {
            Asub[local_m][local_n] = Ab[row * K + a_col];
        } else {
            Asub[local_m][local_n] = 0.0f;
        }

        // Load B tile [kt + *, col] and scale by 2.0f to fuse (b + b)
        int b_row = kt + local_m; // reuse local_m for B row load into Bsub[m][col]
        if (b_row < K && col < N) {
            Bsub[local_m][local_n] = 2.0f * Bb[b_row * N + col];
        } else {
            Bsub[local_m][local_n] = 0.0f;
        }

        __syncthreads();

        // Compute accumulation for the current tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += Asub[local_m][kk] * Bsub[kk][local_n];
        }

        __syncthreads();
    }

    // Write back
    if (row < M && col < N) {
        Cb[row * N + col] = acc;
    }
}

static inline std::tuple<int64_t,int64_t,int64_t,int64_t> get_bmknd(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.dim() >= 2 && b.dim() >= 2, "Inputs must have at least 2 dimensions");
    int64_t M = a.size(-2);
    int64_t K = a.size(-1);
    TORCH_CHECK(b.size(-2) == K, "Inner dimensions K must match: a[...,", K, "] vs b[...,", b.size(-2), "]");
    int64_t N = b.size(-1);

    // Compute flat batch size, require identical batch shapes
    TORCH_CHECK(a.sizes().slice(0, a.dim()-2) == b.sizes().slice(0, b.dim()-2),
                "Batch dimensions must match");
    int64_t B = 1;
    for (int64_t i = 0; i < a.dim()-2; ++i) B *= a.size(i);
    return {B,M,K,N};
}

// HIP launcher function kept as 'forward' to match expected interface
torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Only float32 supported");
    // Ensure contiguous for predictable indexing and coalesced loads
    auto a_ = a.contiguous();
    auto b_ = b.contiguous();

    auto[B,M,K,N] = get_bmknd(a_, b_);

    // View to 3D batched layout
    auto a3 = a_.view({B, M, K});
    auto b3 = b_.view({B, K, N});

    auto out = torch::empty({B, M, N}, a_.options());

    // Launch configuration
    dim3 block(TILE_N, TILE_M, 1); // x: N, y: M
    dim3 grid((unsigned)((N + TILE_N - 1) / TILE_N),
              (unsigned)((M + TILE_M - 1) / TILE_M),
              (unsigned)B);

    hipLaunchKernelGGL(batched_gemm_bplusb_kernel, grid, block, 0, 0,
        a3.data_ptr<float>(), b3.data_ptr<float>(), out.data_ptr<float>(),
        (int)B, (int)M, (int)K, (int)N);

    // Optional: synchronize to surface errors eagerly in extensions
    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    // Reshape back to original leading batch dims
    std::vector<int64_t> out_shape(a_.sizes().begin(), a_.sizes().end());
    out_shape.pop_back(); // remove K
    out_shape.back() = N; // replace MxK -> MxN (we popped K, keep M and set N)
    // Correction: rebuild properly: leading batch dims + [M, N]
    out_shape = std::vector<int64_t>(a_.sizes().begin(), a_.sizes().end()-2);
    out_shape.push_back(M);
    out_shape.push_back(N);

    return out.view(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SimpleMatmulModule HIP forward (A @ (B+B))");
}

