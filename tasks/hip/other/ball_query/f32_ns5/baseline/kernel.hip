// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query_gpu.cu
// (via OpenMMLab mmcv ops/csrc/common/cuda/ball_query_cuda_kernel.cuh)
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void ball_query_kernel(int b, int n, int m,
                                  float min_radius,
                                  float max_radius,
                                  int nsample,
                                  const float *__restrict__ new_xyz,
                                  const float *__restrict__ xyz,
                                  int *__restrict__ idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= m) return;

  new_xyz += bs_idx * m * 3 + pt_idx * 3;
  xyz += bs_idx * n * 3;
  idx += bs_idx * m * nsample + pt_idx * nsample;

  float max_radius2 = max_radius * max_radius;
  float min_radius2 = min_radius * min_radius;
  float new_x = new_xyz[0];
  float new_y = new_xyz[1];
  float new_z = new_xyz[2];

  int cnt = 0;
  for (int k = 0; k < n; ++k) {
    float x = xyz[k * 3 + 0];
    float y = xyz[k * 3 + 1];
    float z = xyz[k * 3 + 2];
    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
               (new_z - z) * (new_z - z);
    if (d2 == 0 || (d2 >= min_radius2 && d2 < max_radius2)) {
      if (cnt == 0) {
        for (int l = 0; l < nsample; ++l) {
          idx[l] = k;
        }
      }
      idx[cnt] = k;
      ++cnt;
      if (cnt >= nsample) break;
    }
  }
  // The original wrapper zero-initialised idx; a ball without any point stays 0.
  if (cnt == 0) {
    for (int l = 0; l < nsample; ++l) {
      idx[l] = 0;
    }
  }
}

void run(const at::Tensor& xyz, const at::Tensor& center_xyz, double min_radius, double max_radius,
         at::Tensor& idx) {
  TORCH_CHECK(xyz.is_cuda() && center_xyz.is_cuda() && idx.is_cuda(), "tensors must be on the HIP device");
  TORCH_CHECK(xyz.scalar_type() == at::kFloat && center_xyz.scalar_type() == at::kFloat, "float32 points");
  TORCH_CHECK(idx.scalar_type() == at::kInt, "idx must be int32");
  TORCH_CHECK(xyz.is_contiguous() && center_xyz.is_contiguous() && idx.is_contiguous(),
              "tensors must be contiguous");
  const int b = xyz.size(0);
  const int n = xyz.size(1);
  const int m = center_xyz.size(1);
  const int nsample = idx.size(2);

  dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);
  dim3 threads(THREADS_PER_BLOCK);
  hipStream_t stream = c10::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(ball_query_kernel, blocks, threads, 0, stream, b, n, m,
                     static_cast<float>(min_radius), static_cast<float>(max_radius), nsample,
                     center_xyz.data_ptr<float>(), xyz.data_ptr<float>(), idx.data_ptr<int>());
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor xyz, Tensor center_xyz, float min_radius, float max_radius, Tensor(a!) idx) -> ()", &run);
}
