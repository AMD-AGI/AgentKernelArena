// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// (via OpenMMLab mmcv ops/csrc/common/cuda/roiaware_pool3d_cuda_kernel.cuh)
// Written by Shaoshuai Shi
// All Rights Reserved 2019.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define MAX_PTS_EACH_VOXEL 128  // max_pts_per_voxel of the operator (index 0 is the counter)

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

__global__ void generate_pts_mask_for_box3d(int boxes_num, int pts_num,
                                            int out_x, int out_y, int out_z,
                                            const float *rois, const float *pts,
                                            int *pts_mask) {
  // params rois: (N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z]
  // params pts_mask: (N, npoints): -1 means point does not in this box,
  // otherwise: encode (x_idxs, y_idxs, z_idxs) by binary bit
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int box_idx = blockIdx.y;
  if (pt_idx >= pts_num || box_idx >= boxes_num) return;

  pts += pt_idx * 3;
  rois += box_idx * 7;
  pts_mask += box_idx * pts_num + pt_idx;

  float local_x = 0, local_y = 0;
  int cur_in_flag = check_pt_in_box3d(pts, rois, local_x, local_y);

  pts_mask[0] = -1;
  if (cur_in_flag > 0) {
    float local_z = pts[2] - rois[2];
    float x_size = rois[3], y_size = rois[4], z_size = rois[5];

    float x_res = x_size / out_x;
    float y_res = y_size / out_y;
    float z_res = z_size / out_z;

    unsigned int x_idx = int((local_x + x_size / 2) / x_res);
    unsigned int y_idx = int((local_y + y_size / 2) / y_res);
    unsigned int z_idx = int(local_z / z_res);

    x_idx = min(max(x_idx, 0), out_x - 1);
    y_idx = min(max(y_idx, 0), out_y - 1);
    z_idx = min(max(z_idx, 0), out_z - 1);

    unsigned int idx_encoding = (x_idx << 16) + (y_idx << 8) + z_idx;
    pts_mask[0] = idx_encoding;
  }
}

__global__ void zero_voxel_counters(int num_voxels, int max_pts_each_voxel, int *pts_idx_of_voxels) {
  // The original wrapper zero-initialised pts_idx_of_voxels; only the counters need it.
  int voxel = blockIdx.x * blockDim.x + threadIdx.x;
  if (voxel >= num_voxels) return;
  pts_idx_of_voxels[voxel * max_pts_each_voxel] = 0;
}

__global__ void collect_inside_pts_for_box3d(int boxes_num, int pts_num,
                                             int max_pts_each_voxel, int out_x,
                                             int out_y, int out_z,
                                             const int *pts_mask,
                                             int *pts_idx_of_voxels) {
  // params pts_mask: (N, npoints)  0 or 1
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)

  int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (box_idx >= boxes_num) return;

  int max_num_pts = max_pts_each_voxel - 1;  // index 0 is the counter
  pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel;

  for (int k = 0; k < pts_num; k++) {
    if (pts_mask[box_idx * pts_num + k] != -1) {
      unsigned int idx_encoding = pts_mask[box_idx * pts_num + k];
      unsigned int x_idx = (idx_encoding >> 16) & 0xFF;
      unsigned int y_idx = (idx_encoding >> 8) & 0xFF;
      unsigned int z_idx = idx_encoding & 0xFF;
      unsigned int base_offset = x_idx * out_y * out_z * max_pts_each_voxel +
                                 y_idx * out_z * max_pts_each_voxel +
                                 z_idx * max_pts_each_voxel;
      unsigned int cnt = pts_idx_of_voxels[base_offset];
      if (cnt < max_num_pts) {
        pts_idx_of_voxels[base_offset + cnt + 1] = k;
        pts_idx_of_voxels[base_offset]++;
      }
    }
  }
}

__global__ void roiaware_maxpool3d(int boxes_num, int pts_num, int channels,
                                   int max_pts_each_voxel, int out_x, int out_y,
                                   int out_z, const float *pts_feature,
                                   const int *pts_idx_of_voxels,
                                   float *pooled_features, int *argmax) {
  // params pts_feature: (npoints, C)
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel),
  // index 0 is the counter params pooled_features: (N, out_x, out_y, out_z, C)
  // params argmax: (N, out_x, out_y, out_z, C)

  int box_idx = blockIdx.z;
  int channel_idx = blockIdx.y;
  int voxel_idx_flat = blockIdx.x * blockDim.x + threadIdx.x;

  int x_idx = voxel_idx_flat / (out_y * out_z);
  int y_idx = (voxel_idx_flat - x_idx * (out_y * out_z)) / out_z;
  int z_idx = voxel_idx_flat % out_z;
  if (box_idx >= boxes_num || channel_idx >= channels || x_idx >= out_x ||
      y_idx >= out_y || z_idx >= out_z)
    return;

  int offset_base = x_idx * out_y * out_z + y_idx * out_z + z_idx;
  pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel +
                       offset_base * max_pts_each_voxel;
  pooled_features += box_idx * out_x * out_y * out_z * channels +
                     offset_base * channels + channel_idx;
  argmax += box_idx * out_x * out_y * out_z * channels +
            offset_base * channels + channel_idx;

  int argmax_idx = -1;
  float max_val = -1e50;

  int total_pts = pts_idx_of_voxels[0];

  for (int k = 1; k <= total_pts; k++) {
    if (pts_feature[pts_idx_of_voxels[k] * channels + channel_idx] > max_val) {
      max_val = pts_feature[pts_idx_of_voxels[k] * channels + channel_idx];
      argmax_idx = pts_idx_of_voxels[k];
    }
  }

  // The original wrapper zero-initialised pooled_features; an empty voxel is zero-filled here.
  pooled_features[0] = (argmax_idx != -1) ? max_val : 0.0f;
  argmax[0] = argmax_idx;
}

void run(const at::Tensor& rois, const at::Tensor& pts, const at::Tensor& pts_feature,
         at::Tensor& pooled_features) {
  TORCH_CHECK(rois.is_cuda() && pts.is_cuda() && pts_feature.is_cuda() && pooled_features.is_cuda(),
              "tensors must be on the HIP device");
  TORCH_CHECK(rois.scalar_type() == at::kFloat && pts.scalar_type() == at::kFloat &&
                  pts_feature.scalar_type() == at::kFloat && pooled_features.scalar_type() == at::kFloat,
              "float32 tensors");
  TORCH_CHECK(rois.is_contiguous() && pts.is_contiguous() && pts_feature.is_contiguous() &&
                  pooled_features.is_contiguous(), "tensors must be contiguous");
  const int boxes_num = rois.size(0);
  const int pts_num = pts.size(0);
  const int channels = pts_feature.size(1);
  const int out_x = pooled_features.size(1);
  const int out_y = pooled_features.size(2);
  const int out_z = pooled_features.size(3);
  const int max_pts_each_voxel = MAX_PTS_EACH_VOXEL;
  TORCH_CHECK(out_x < 256 && out_y < 256 && out_z < 256, "voxel indices are encoded in 8 bits");

  // Scratch buffers of the original wrapper.
  at::Tensor pts_mask = at::empty({boxes_num, pts_num}, rois.options().dtype(at::kInt));
  at::Tensor pts_idx_of_voxels =
      at::empty({boxes_num, out_x, out_y, out_z, max_pts_each_voxel}, rois.options().dtype(at::kInt));
  at::Tensor argmax = at::empty({boxes_num, out_x, out_y, out_z, channels}, rois.options().dtype(at::kInt));

  hipStream_t stream = c10::hip::getCurrentHIPStream();
  dim3 threads(THREADS_PER_BLOCK);

  const int num_voxels = boxes_num * out_x * out_y * out_z;
  hipLaunchKernelGGL(zero_voxel_counters, dim3(DIVUP(num_voxels, THREADS_PER_BLOCK)), threads, 0, stream,
                     num_voxels, max_pts_each_voxel, pts_idx_of_voxels.data_ptr<int>());

  dim3 blocks_mask(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num);
  hipLaunchKernelGGL(generate_pts_mask_for_box3d, blocks_mask, threads, 0, stream, boxes_num, pts_num,
                     out_x, out_y, out_z, rois.data_ptr<float>(), pts.data_ptr<float>(),
                     pts_mask.data_ptr<int>());

  dim3 blocks_collect(DIVUP(boxes_num, THREADS_PER_BLOCK));
  hipLaunchKernelGGL(collect_inside_pts_for_box3d, blocks_collect, threads, 0, stream, boxes_num,
                     pts_num, max_pts_each_voxel, out_x, out_y, out_z, pts_mask.data_ptr<int>(),
                     pts_idx_of_voxels.data_ptr<int>());

  dim3 blocks_pool(DIVUP(out_x * out_y * out_z, THREADS_PER_BLOCK), channels, boxes_num);
  hipLaunchKernelGGL(roiaware_maxpool3d, blocks_pool, threads, 0, stream, boxes_num, pts_num, channels,
                     max_pts_each_voxel, out_x, out_y, out_z, pts_feature.data_ptr<float>(),
                     pts_idx_of_voxels.data_ptr<int>(), pooled_features.data_ptr<float>(),
                     argmax.data_ptr<int>());
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor rois, Tensor pts, Tensor pts_feature, Tensor(a!) pooled_features) -> ()", &run);
}
