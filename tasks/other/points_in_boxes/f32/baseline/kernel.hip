// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// Written by Shaoshuai Shi
// All Rights Reserved 2019.
// (via OpenMMLab mmcv ops/csrc/common/cuda/points_in_boxes_cuda_kernel.cuh)
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__device__ inline void lidar_to_local_coords(float shift_x, float shift_y,
                                             float rz, float &local_x,
                                             float &local_y) {
  float cosa = cos(-rz), sina = sin(-rz);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}

__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d,
                                        float &local_x, float &local_y) {
  // param pt: (x, y, z)
  // param box3d: (cx, cy, cz, x_size, y_size, z_size, rz) in LiDAR coordinate, cz in the
  // bottom center
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float x_size = box3d[3], y_size = box3d[4], z_size = box3d[5], rz = box3d[6];
  cz += z_size / 2.0;  // shift to the center since cz in box3d is the bottom center

  if (fabsf(z - cz) > z_size / 2.0) return 0;
  lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
  float in_flag = (local_x > -x_size / 2.0) & (local_x < x_size / 2.0) &
                  (local_y > -y_size / 2.0) & (local_y < y_size / 2.0);
  return in_flag;
}

__global__ void points_in_boxes_part_kernel(int batch_size, int boxes_num,
                                            int pts_num, const float *boxes,
                                            const float *pts,
                                            int *box_idx_of_points) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR coordinate, z is
  // the bottom center, each box DO NOT overlaps params pts: (B, npoints, 3) [x,
  // y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints), default
  // -1

  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= batch_size || pt_idx >= pts_num) return;

  boxes += bs_idx * boxes_num * 7;
  pts += bs_idx * pts_num * 3 + pt_idx * 3;
  box_idx_of_points += bs_idx * pts_num + pt_idx;

  float local_x = 0, local_y = 0;
  int cur_in_flag = 0;
  // The original wrapper pre-filled the output with -1; here the kernel writes it.
  box_idx_of_points[0] = -1;
  for (int k = 0; k < boxes_num; k++) {
    cur_in_flag = check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
    if (cur_in_flag) {
      box_idx_of_points[0] = k;
      break;
    }
  }
}

void run(const at::Tensor& points, const at::Tensor& boxes, at::Tensor& box_idx_of_points) {
  TORCH_CHECK(points.is_cuda() && boxes.is_cuda() && box_idx_of_points.is_cuda(),
              "tensors must be on the HIP device");
  TORCH_CHECK(points.scalar_type() == at::kFloat && boxes.scalar_type() == at::kFloat, "float32 inputs");
  TORCH_CHECK(box_idx_of_points.scalar_type() == at::kInt, "output must be int32");
  TORCH_CHECK(points.is_contiguous() && boxes.is_contiguous() && box_idx_of_points.is_contiguous(),
              "tensors must be contiguous");
  const int batch_size = boxes.size(0);
  const int boxes_num = boxes.size(1);
  const int pts_num = points.size(1);

  dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), batch_size);
  dim3 threads(THREADS_PER_BLOCK);
  hipStream_t stream = c10::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(points_in_boxes_part_kernel, blocks, threads, 0, stream, batch_size, boxes_num,
                     pts_num, boxes.data_ptr<float>(), points.data_ptr<float>(),
                     box_idx_of_points.data_ptr<int>());
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor points, Tensor boxes, Tensor(a!) box_idx_of_points) -> ()", &run);
}
