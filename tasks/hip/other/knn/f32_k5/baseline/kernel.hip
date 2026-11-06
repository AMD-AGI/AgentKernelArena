// Modified from https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/pointops/src/knnquery_heap
// as vendored by OpenMMLab mmcv (Copyright (c) OpenMMLab. All rights reserved.).
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define MAX_NSAMPLE 100

__device__ void swap_float(float *x, float *y) {
  float tmp = *x;
  *x = *y;
  *y = tmp;
}

__device__ void swap_int(int *x, int *y) {
  int tmp = *x;
  *x = *y;
  *y = tmp;
}

// (da, ia) orders before (db, ib): smaller distance first, smaller index on equal distances.
__device__ bool before(float da, int ia, float db, int ib) {
  return da < db || (da == db && ia < ib);
}

// Max-heap on (distance, index).
__device__ void reheap(float *dist, int *idx, int k) {
  int root = 0;
  int child = root * 2 + 1;
  while (child < k) {
    if (child + 1 < k && before(dist[child], idx[child], dist[child + 1], idx[child + 1]))
      child++;
    if (!before(dist[root], idx[root], dist[child], idx[child])) return;
    swap_float(&dist[root], &dist[child]);
    swap_int(&idx[root], &idx[child]);
    root = child;
    child = root * 2 + 1;
  }
}

__device__ void heap_sort(float *dist, int *idx, int k) {
  int i;
  for (i = k - 1; i > 0; i--) {
    swap_float(&dist[0], &dist[i]);
    swap_int(&idx[0], &idx[i]);
    reheap(dist, idx, i);
  }
}

// input: xyz (b, n, 3) new_xyz (b, m, 3)
// output: idx (b, m, nsample) dist2 (b, m, nsample), nearest first
__global__ void knn_kernel(int b, int n, int m, int nsample, const float *__restrict__ xyz,
                           const float *__restrict__ new_xyz, int *__restrict__ idx,
                           float *__restrict__ dist2) {
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= m) return;

  new_xyz += bs_idx * m * 3 + pt_idx * 3;
  xyz += bs_idx * n * 3;
  idx += bs_idx * m * nsample + pt_idx * nsample;
  dist2 += bs_idx * m * nsample + pt_idx * nsample;

  float new_x = new_xyz[0];
  float new_y = new_xyz[1];
  float new_z = new_xyz[2];

  float best_dist[MAX_NSAMPLE];
  int best_idx[MAX_NSAMPLE];
  for (int i = 0; i < nsample; i++) {
    best_dist[i] = 1e10;
    best_idx[i] = 0;
  }
  for (int i = 0; i < n; i++) {
    float x = xyz[i * 3 + 0];
    float y = xyz[i * 3 + 1];
    float z = xyz[i * 3 + 2];
    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
    if (d2 < best_dist[0]) {
      best_dist[0] = d2;
      best_idx[0] = i;
      reheap(best_dist, best_idx, nsample);
    }
  }
  heap_sort(best_dist, best_idx, nsample);
  for (int i = 0; i < nsample; i++) {
    idx[i] = best_idx[i];
    dist2[i] = best_dist[i];
  }
}

// Entry point: the nsample nearest points of xyz[b] to every row of center_xyz[b].
void run(const at::Tensor &xyz, const at::Tensor &center_xyz, at::Tensor &idx, at::Tensor &dist2) {
  TORCH_CHECK(xyz.scalar_type() == at::kFloat && center_xyz.scalar_type() == at::kFloat &&
                  dist2.scalar_type() == at::kFloat && idx.scalar_type() == at::kInt,
              "float32 coordinates/distances and int32 indices expected");
  TORCH_CHECK(xyz.is_contiguous() && center_xyz.is_contiguous() && idx.is_contiguous() &&
                  dist2.is_contiguous(),
              "contiguous tensors expected");
  TORCH_CHECK(xyz.dim() == 3 && xyz.size(2) == 3 && center_xyz.dim() == 3 && center_xyz.size(2) == 3 &&
                  idx.dim() == 3 && dist2.sizes() == idx.sizes() && idx.size(0) == xyz.size(0) &&
                  idx.size(1) == center_xyz.size(1),
              "xyz (B, N, 3), center_xyz (B, M, 3), idx and dist2 (B, M, k)");
  const int b = static_cast<int>(xyz.size(0));
  const int n = static_cast<int>(xyz.size(1));
  const int m = static_cast<int>(center_xyz.size(1));
  const int k = static_cast<int>(idx.size(2));
  TORCH_CHECK(k >= 1 && k <= MAX_NSAMPLE, "k must be in [1, 100]");
  if (b == 0 || m == 0) return;

  dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);
  hipStream_t stream = c10::hip::getCurrentHIPStream();
  knn_kernel<<<blocks, threads, 0, stream>>>(b, n, m, k, xyz.data_ptr<float>(),
                                             center_xyz.data_ptr<float>(), idx.data_ptr<int>(),
                                             dist2.data_ptr<float>());
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor xyz, Tensor center_xyz, Tensor(a!) idx, Tensor(a!) dist2) -> ()", &run);
}
