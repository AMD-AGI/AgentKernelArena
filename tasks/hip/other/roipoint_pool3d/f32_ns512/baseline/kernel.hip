/*
Modified from
https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d_kernel.cu
(via OpenPCDet and OpenMMLab mmcv ops/csrc/pytorch/cuda/roipoint_pool3d_cuda.cu)
Point cloud feature pooling
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

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
  // param box3d: (cx, cy, cz, dx, dy, dz, rz) in LiDAR coordinate, cz in the
  // bottom center
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];
  cz += dz / 2.0;  // shift to the center since cz in box3d is the bottom center

  if (fabsf(z - cz) > dz / 2.0) return 0;
  lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
  float in_flag = (local_x > -dx / 2.0) & (local_x < dx / 2.0) &
                  (local_y > -dy / 2.0) & (local_y < dy / 2.0);
  return in_flag;
}

__global__ void assign_pts_to_box3d(int batch_size, int pts_num, int boxes_num, const float *xyz, const float *boxes3d, int *pts_assign){
    // params xyz: (B, N, 3)
    // params boxes3d: (B, M, 7)
    // params pts_assign: (B, N, M): idx of the corresponding box3d, -1 means background points
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = blockIdx.z;

    if (pt_idx >= pts_num || box_idx >= boxes_num || bs_idx >= batch_size){
        return;
    }
    int assign_idx = bs_idx * pts_num * boxes_num + pt_idx * boxes_num + box_idx;
    pts_assign[assign_idx] = 0;

    int box_offset = bs_idx * boxes_num * 7 + box_idx * 7;
    int pt_offset = bs_idx * pts_num * 3 + pt_idx * 3;

    float local_x = 0, local_y = 0;
    int cur_in_flag = check_pt_in_box3d(xyz + pt_offset, boxes3d + box_offset, local_x, local_y);
    pts_assign[assign_idx] = cur_in_flag;
}

__global__ void get_pooled_idx(int batch_size, int pts_num, int boxes_num, int sampled_pts_num,
                               const int *pts_assign, int *pts_idx, int *pooled_empty_flag){
    // params pts_assign: (B, N, M)
    // params pts_idx: (B, M, 512)
    // params pooled_empty_flag: (B, M)

    int boxes_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boxes_idx >= boxes_num){
        return;
    }

    int bs_idx = blockIdx.y;

    int cnt = 0;
    for (int k = 0; k < pts_num; k++){
        if (pts_assign[bs_idx * pts_num * boxes_num + k * boxes_num + boxes_idx]){
            if (cnt < sampled_pts_num){
                pts_idx[bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num + cnt] = k;
                cnt++;
            }
            else break;
        }
    }

    // The original wrapper pre-zeroed the flag; here the kernel writes it.
    pooled_empty_flag[bs_idx * boxes_num + boxes_idx] = (cnt == 0) ? 1 : 0;
    if (cnt > 0 && cnt < sampled_pts_num){
        // duplicate same points for sampling
        for (int k = cnt; k < sampled_pts_num; k++){
            int duplicate_idx = k % cnt;
            int base_offset = bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num;
            pts_idx[base_offset + k] = pts_idx[base_offset + duplicate_idx];
        }
    }
}

__global__ void roipool3d_forward(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
                                   const float *xyz, const int *pts_idx, const float *pts_feature,
                                   float *pooled_features, const int *pooled_empty_flag){
    // params xyz: (B, N, 3)
    // params pts_idx: (B, M, 512)
    // params pts_feature: (B, N, C)
    // params pooled_features: (B, M, 512, 3+C)
    // params pooled_empty_flag: (B, M)

    int sample_pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = blockIdx.z;

    if (sample_pt_idx >= sampled_pts_num || box_idx >= boxes_num || bs_idx >= batch_size){
        return;
    }

    int temp_idx = bs_idx * boxes_num * sampled_pts_num + box_idx * sampled_pts_num + sample_pt_idx;
    int dst_feature_offset = temp_idx * (3 + feature_in_len);

    if (pooled_empty_flag[bs_idx * boxes_num + box_idx]){
        // The original wrapper pre-zeroed the output; an empty box is zero-filled here.
        for (int j = 0; j < 3 + feature_in_len; j++)
            pooled_features[dst_feature_offset + j] = 0.0f;
        return;
    }

    int src_pt_idx = pts_idx[temp_idx];

    for (int j = 0; j < 3; j++)
        pooled_features[dst_feature_offset + j] = xyz[bs_idx * pts_num * 3 + src_pt_idx * 3 + j];

    int src_feature_offset = bs_idx * pts_num * feature_in_len + src_pt_idx * feature_in_len;
    for (int j = 0; j < feature_in_len; j++)
        pooled_features[dst_feature_offset + 3 + j] = pts_feature[src_feature_offset + j];
}

void run(const at::Tensor& points, const at::Tensor& point_features, const at::Tensor& boxes3d,
         at::Tensor& pooled_features, at::Tensor& pooled_empty_flag) {
  TORCH_CHECK(points.is_cuda() && point_features.is_cuda() && boxes3d.is_cuda() &&
                  pooled_features.is_cuda() && pooled_empty_flag.is_cuda(),
              "tensors must be on the HIP device");
  TORCH_CHECK(points.scalar_type() == at::kFloat && point_features.scalar_type() == at::kFloat &&
                  boxes3d.scalar_type() == at::kFloat && pooled_features.scalar_type() == at::kFloat,
              "float32 tensors");
  TORCH_CHECK(pooled_empty_flag.scalar_type() == at::kInt, "pooled_empty_flag must be int32");
  TORCH_CHECK(points.is_contiguous() && point_features.is_contiguous() && boxes3d.is_contiguous() &&
                  pooled_features.is_contiguous() && pooled_empty_flag.is_contiguous(),
              "tensors must be contiguous");
  const int batch_size = points.size(0);
  const int pts_num = points.size(1);
  const int boxes_num = boxes3d.size(1);
  const int feature_in_len = point_features.size(2);
  const int sampled_pts_num = pooled_features.size(2);

  // Scratch buffers of the original wrapper.
  at::Tensor pts_assign = at::empty({batch_size, pts_num, boxes_num}, points.options().dtype(at::kInt));
  at::Tensor pts_idx = at::empty({batch_size, boxes_num, sampled_pts_num}, points.options().dtype(at::kInt));

  hipStream_t stream = c10::hip::getCurrentHIPStream();
  dim3 threads(THREADS_PER_BLOCK);

  dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);
  hipLaunchKernelGGL(assign_pts_to_box3d, blocks, threads, 0, stream, batch_size, pts_num, boxes_num,
                     points.data_ptr<float>(), boxes3d.data_ptr<float>(), pts_assign.data_ptr<int>());

  dim3 blocks2(DIVUP(boxes_num, THREADS_PER_BLOCK), batch_size);
  hipLaunchKernelGGL(get_pooled_idx, blocks2, threads, 0, stream, batch_size, pts_num, boxes_num,
                     sampled_pts_num, pts_assign.data_ptr<int>(), pts_idx.data_ptr<int>(),
                     pooled_empty_flag.data_ptr<int>());

  dim3 blocks_pool(DIVUP(sampled_pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);
  hipLaunchKernelGGL(roipool3d_forward, blocks_pool, threads, 0, stream, batch_size, pts_num, boxes_num,
                     feature_in_len, sampled_pts_num, points.data_ptr<float>(), pts_idx.data_ptr<int>(),
                     point_features.data_ptr<float>(), pooled_features.data_ptr<float>(),
                     pooled_empty_flag.data_ptr<int>());
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor points, Tensor point_features, Tensor boxes3d, Tensor(a!) pooled_features, "
        "Tensor(a!) pooled_empty_flag) -> ()", &run);
}
