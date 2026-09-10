#include "hip/hip_runtime.h"
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#define THREADS_PER_BLOCK 256
#define WAVE_SIZE 64
#define QUERIES_PER_BLOCK (THREADS_PER_BLOCK / WAVE_SIZE)
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
  int wave_idx = threadIdx.x / WAVE_SIZE;
  int lane_idx = threadIdx.x % WAVE_SIZE;
  int pt_idx = blockIdx.x * QUERIES_PER_BLOCK + wave_idx;
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
  for (int base = 0; base < n; base += WAVE_SIZE) {
    int k = base + lane_idx;
    bool is_match = false;
    if (k < n) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) +
                 (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      is_match = d2 == 0 || (d2 >= min_radius2 && d2 < max_radius2);
    }

    unsigned long long matches = __ballot(is_match);
    if (lane_idx == 0 && matches != 0) {
      unsigned long long selected = matches;
      if (cnt == 0) {
        int first = base + __builtin_ctzll(selected);
        for (int l = 0; l < nsample; ++l) {
          idx[l] = first;
        }
      }

      for (int out = cnt; out < nsample && selected != 0; ++out) {
        int match_lane = __builtin_ctzll(selected);
        idx[out] = base + match_lane;
        selected &= selected - 1;
      }
    }

    cnt += __builtin_popcountll(matches);
    if (cnt >= nsample) break;
  }
}

void ball_query_kernel_launcher(int b, int n, int m, float min_radius, float max_radius,
                                int nsample, const float *new_xyz, const float *xyz,
                                int *idx, hipStream_t stream) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)

  hipError_t err;

  dim3 blocks(DIVUP(m, QUERIES_PER_BLOCK),
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  ball_query_kernel<<<blocks, threads, 0, stream>>>(b, n, m, min_radius, max_radius,
                                                    nsample, new_xyz, xyz, idx);
  // hipDeviceSynchronize();  // for using printf in kernel function
  err = hipGetLastError();
  if (hipSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", hipGetErrorString(err));
    exit(-1);
  }
}
