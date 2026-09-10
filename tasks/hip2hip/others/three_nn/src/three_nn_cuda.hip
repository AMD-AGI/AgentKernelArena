#include "hip/hip_runtime.h"
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// Insert one candidate into an already sorted three-element list.  Expressing
// the insertion as selects avoids repeatedly manipulating EXEC when different
// queries find a new nearest point on different loop iterations.
__device__ __forceinline__ void insert_neighbor(
    float d, int candidate_idx, float &best1, float &best2, float &best3,
    int &besti1, int &besti2, int &besti3) {
  bool take = d < best1;
  float displaced_d = take ? best1 : d;
  int displaced_i = take ? besti1 : candidate_idx;
  best1 = take ? d : best1;
  besti1 = take ? candidate_idx : besti1;

  take = displaced_d < best2;
  d = take ? best2 : displaced_d;
  candidate_idx = take ? besti2 : displaced_i;
  best2 = take ? displaced_d : best2;
  besti2 = take ? displaced_i : besti2;

  take = d < best3;
  best3 = take ? d : best3;
  besti3 = take ? candidate_idx : besti3;
}

// The wave reduction combines non-contiguous subsequences of the source
// points, so equal distances also need their original source-order tie break.
__device__ __forceinline__ void insert_neighbor_ordered(
    float d, int candidate_idx, float &best1, float &best2, float &best3,
    int &besti1, int &besti2, int &besti3) {
  bool take = (d < best1) || (d == best1 && candidate_idx < besti1);
  float displaced_d = take ? best1 : d;
  int displaced_i = take ? besti1 : candidate_idx;
  best1 = take ? d : best1;
  besti1 = take ? candidate_idx : besti1;

  take = (displaced_d < best2) ||
         (displaced_d == best2 && displaced_i < besti2);
  d = take ? best2 : displaced_d;
  candidate_idx = take ? besti2 : displaced_i;
  best2 = take ? displaced_d : best2;
  besti2 = take ? displaced_i : besti2;

  take = (d < best3) || (d == best3 && candidate_idx < besti3);
  best3 = take ? d : best3;
  besti3 = take ? candidate_idx : besti3;
}

__global__ void three_nn_kernel(int b, int n, int m,
                                const float *__restrict__ unknown,
                                const float *__restrict__ known,
                                float *__restrict__ dist2,
                                int *__restrict__ idx) {
  // unknown: (B, N, 3)
  // known: (B, M, 3)
  // output:
  //      dist2: (B, N, 3)
  //      idx: (B, N, 3)

  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= n) return;

  unknown += bs_idx * n * 3 + pt_idx * 3;
  known += bs_idx * m * 3;
  dist2 += bs_idx * n * 3 + pt_idx * 3;
  idx += bs_idx * n * 3 + pt_idx * 3;

  float ux = unknown[0];
  float uy = unknown[1];
  float uz = unknown[2];

  // Every distance is computed in float.  Infinity has the same comparison
  // behavior here as the original finite double sentinel: all finite floats
  // are accepted, while +inf and NaN are not.
  float best1 = INFINITY, best2 = INFINITY, best3 = INFINITY;
  int besti1 = 0, besti2 = 0, besti3 = 0;
  for (int k = 0; k < m; ++k) {
    float x = known[k * 3 + 0];
    float y = known[k * 3 + 1];
    float z = known[k * 3 + 2];
    float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
    insert_neighbor(d, k, best1, best2, best3, besti1, besti2, besti3);
  }
  dist2[0] = best1;
  dist2[1] = best2;
  dist2[2] = best3;
  idx[0] = besti1;
  idx[1] = besti2;
  idx[2] = besti3;
}

// When there are too few query points to fill the GPU, use one Wave64 per
// query and reduce the per-lane top threes.  Source loads are then contiguous
// across the wave instead of leaving most CUs idle for a long serial scan.
__global__ void three_nn_wave_kernel(int b, int n, int m,
                                     const float *__restrict__ unknown,
                                     const float *__restrict__ known,
                                     float *__restrict__ dist2,
                                     int *__restrict__ idx) {
  constexpr int kWaveSize = 64;
  constexpr int kWavesPerBlock = THREADS_PER_BLOCK / kWaveSize;

  int lane = threadIdx.x & (kWaveSize - 1);
  int wave = threadIdx.x / kWaveSize;
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * kWavesPerBlock + wave;
  if (bs_idx >= b || pt_idx >= n) return;

  unknown += (bs_idx * n + pt_idx) * 3;
  known += bs_idx * m * 3;

  float ux = unknown[0];
  float uy = unknown[1];
  float uz = unknown[2];

  float best1 = INFINITY, best2 = INFINITY, best3 = INFINITY;
  int besti1 = 0, besti2 = 0, besti3 = 0;
  for (int k = lane; k < m; k += kWaveSize) {
    float x = known[k * 3 + 0];
    float y = known[k * 3 + 1];
    float z = known[k * 3 + 2];
    float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) +
              (uz - z) * (uz - z);
    insert_neighbor(d, k, best1, best2, best3, besti1, besti2, besti3);
  }

  for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
    float other1 = __shfl_down(best1, offset, kWaveSize);
    float other2 = __shfl_down(best2, offset, kWaveSize);
    float other3 = __shfl_down(best3, offset, kWaveSize);
    int otheri1 = __shfl_down(besti1, offset, kWaveSize);
    int otheri2 = __shfl_down(besti2, offset, kWaveSize);
    int otheri3 = __shfl_down(besti3, offset, kWaveSize);
    if (lane < offset) {
      insert_neighbor_ordered(other1, otheri1, best1, best2, best3,
                              besti1, besti2, besti3);
      insert_neighbor_ordered(other2, otheri2, best1, best2, best3,
                              besti1, besti2, besti3);
      insert_neighbor_ordered(other3, otheri3, best1, best2, best3,
                              besti1, besti2, besti3);
    }
  }

  if (lane == 0) {
    int out = (bs_idx * n + pt_idx) * 3;
    dist2[out + 0] = best1;
    dist2[out + 1] = best2;
    dist2[out + 2] = best3;
    idx[out + 0] = besti1;
    idx[out + 1] = besti2;
    idx[out + 2] = besti3;
  }
}

void three_nn_kernel_launcher(int b, int n, int m, const float *unknown,
                              const float *known, float *dist2, int *idx,
                              hipStream_t stream) {
  // unknown: (B, N, 3)
  // known: (B, M, 3)
  // output:
  //      dist2: (B, N, 3)
  //      idx: (B, N, 3)

  hipError_t err;
  dim3 threads(THREADS_PER_BLOCK);

  if (m >= 256 && static_cast<long long>(b) * n <= 2048) {
    constexpr int kWavesPerBlock = THREADS_PER_BLOCK / 64;
    dim3 blocks(DIVUP(n, kWavesPerBlock), b);
    three_nn_wave_kernel<<<blocks, threads, 0, stream>>>(
        b, n, m, unknown, known, dist2, idx);
  } else {
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK),
                b);  // blockIdx.x(col), blockIdx.y(row)
    three_nn_kernel<<<blocks, threads, 0, stream>>>(b, n, m, unknown, known,
                                                    dist2, idx);
  }

  err = hipGetLastError();
  if (hipSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", hipGetErrorString(err));
    exit(-1);
  }
}
