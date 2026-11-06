// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/sampling_gpu.cu
// as vendored by OpenMMLab mmcv (Copyright (c) OpenMMLab. All rights reserved.).
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#define TOTAL_THREADS 1024

// Largest power of two not above work_size, capped at TOTAL_THREADS.
static inline int opt_n_threads(int work_size) {
  int threads = 1;
  while (threads * 2 <= work_size && threads * 2 <= TOTAL_THREADS) threads *= 2;
  return threads;
}

// Keep the larger distance; on equal distances keep the smaller point index.
__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  const bool take2 = (v2 > v1) || (v2 == v1 && i2 < i1);
  dists[idx1] = take2 ? v2 : v1;
  dists_i[idx1] = take2 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
  // dataset: (B, N, 3)
  // temp: (B, N) running minimum squared distance to the selected set
  // output:
  //      idx: (B, M)

  if (m <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  // Each thread initialises exactly the temp entries it reads and writes later.
  for (int k = tid; k < n; k += stride) temp[k] = 1e10f;

  int old = 0;
  if (threadIdx.x == 0) idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];

      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
      if (tid < 512) {
        __update(dists, dists_i, tid, tid + 512);
      }
      __syncthreads();
    }

    if (block_size >= 512) {
      if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    // All threads have read dists_i[0] before the next iteration overwrites it.
    __syncthreads();
    if (tid == 0) idxs[j] = old;
  }
}

static void furthest_point_sampling_kernel_launcher(int b, int n, int m, const float *dataset,
                                                    float *temp, int *idxs, hipStream_t stream) {
  unsigned int n_threads = opt_n_threads(n);

  switch (n_threads) {
    case 1024:
      furthest_point_sampling_kernel<1024>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 512:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_kernel<256>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_kernel<128>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_kernel<64>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_kernel<32>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_kernel<16>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_kernel<8>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_kernel<4>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_kernel<2>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_kernel<1>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
  }
}

// Entry point: idx[b, :] = furthest point sampling of points_xyz[b], starting at point 0.
void run(const at::Tensor &points_xyz, at::Tensor &idx) {
  TORCH_CHECK(points_xyz.scalar_type() == at::kFloat && idx.scalar_type() == at::kInt,
              "points_xyz must be float32 and idx int32");
  TORCH_CHECK(points_xyz.is_contiguous() && idx.is_contiguous(), "contiguous tensors expected");
  TORCH_CHECK(points_xyz.dim() == 3 && points_xyz.size(2) == 3 && idx.dim() == 2 &&
                  idx.size(0) == points_xyz.size(0),
              "points_xyz is (B, N, 3), idx is (B, M)");
  const int b = static_cast<int>(points_xyz.size(0));
  const int n = static_cast<int>(points_xyz.size(1));
  const int m = static_cast<int>(idx.size(1));
  if (b == 0 || n == 0 || m == 0) return;

  at::Tensor temp = at::empty({b, n}, points_xyz.options());
  hipStream_t stream = c10::hip::getCurrentHIPStream();
  furthest_point_sampling_kernel_launcher(b, n, m, points_xyz.data_ptr<float>(),
                                          temp.data_ptr<float>(), idx.data_ptr<int>(), stream);
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor points_xyz, Tensor(a!) idx) -> ()", &run);
}
