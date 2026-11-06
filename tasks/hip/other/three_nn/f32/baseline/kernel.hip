// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate_gpu.cu
// (via OpenMMLab mmcv ops/csrc/common/cuda/three_nn_cuda_kernel.cuh)
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void three_nn_kernel(int b, int n, int m,
                                const float *__restrict__ unknown,
                                const float *__restrict__ known,
                                float *__restrict__ dist,
                                int *__restrict__ idx) {
  // unknown: (B, N, 3)
  // known: (B, M, 3)
  // output:
  //      dist: (B, N, 3)  Euclidean distances (the original kernel stored the squares)
  //      idx: (B, N, 3)

  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= n) return;

  unknown += bs_idx * n * 3 + pt_idx * 3;
  known += bs_idx * m * 3;
  dist += bs_idx * n * 3 + pt_idx * 3;
  idx += bs_idx * n * 3 + pt_idx * 3;

  float ux = unknown[0];
  float uy = unknown[1];
  float uz = unknown[2];

  double best1 = 1e40, best2 = 1e40, best3 = 1e40;
  int besti1 = 0, besti2 = 0, besti3 = 0;
  for (int k = 0; k < m; ++k) {
    float x = known[k * 3 + 0];
    float y = known[k * 3 + 1];
    float z = known[k * 3 + 2];
    float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
    if (d < best1) {
      best3 = best2;
      besti3 = besti2;
      best2 = best1;
      besti2 = besti1;
      best1 = d;
      besti1 = k;
    } else if (d < best2) {
      best3 = best2;
      besti3 = besti2;
      best2 = d;
      besti2 = k;
    } else if (d < best3) {
      best3 = d;
      besti3 = k;
    }
  }
  // The original wrapper took the square root of the squared distances in a separate op.
  dist[0] = sqrtf(static_cast<float>(best1));
  dist[1] = sqrtf(static_cast<float>(best2));
  dist[2] = sqrtf(static_cast<float>(best3));
  idx[0] = besti1;
  idx[1] = besti2;
  idx[2] = besti3;
}

void run(const at::Tensor& unknown, const at::Tensor& known, at::Tensor& dist, at::Tensor& idx) {
  TORCH_CHECK(unknown.is_cuda() && known.is_cuda() && dist.is_cuda() && idx.is_cuda(),
              "tensors must be on the HIP device");
  TORCH_CHECK(unknown.scalar_type() == at::kFloat && known.scalar_type() == at::kFloat &&
                  dist.scalar_type() == at::kFloat, "float32 points and distances");
  TORCH_CHECK(idx.scalar_type() == at::kInt, "idx must be int32");
  TORCH_CHECK(unknown.is_contiguous() && known.is_contiguous() && dist.is_contiguous() &&
                  idx.is_contiguous(), "tensors must be contiguous");
  const int b = unknown.size(0);
  const int n = unknown.size(1);
  const int m = known.size(1);

  dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);
  dim3 threads(THREADS_PER_BLOCK);
  hipStream_t stream = c10::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(three_nn_kernel, blocks, threads, 0, stream, b, n, m,
                     unknown.data_ptr<float>(), known.data_ptr<float>(), dist.data_ptr<float>(),
                     idx.data_ptr<int>());
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor unknown, Tensor known, Tensor(a!) dist, Tensor(a!) idx) -> ()", &run);
}
