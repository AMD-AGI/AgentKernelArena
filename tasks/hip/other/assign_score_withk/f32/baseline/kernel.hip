// Modified from https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/paconv_lib/src/gpu
// (via OpenMMLab mmcv ops/csrc/common/cuda/assign_score_withk_cuda_kernel.cuh)
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// input: points(B,N0,M,O), centers(B,N0,M,O), scores(B,N1,K,M), knn_idx(B,N1,K)
// output: fout(B,O,N1,K)
// algo: fout(b,o,n,k) = sum_m s(b,n,k,m) * (p(b,i(k),m,o) - p(b,i(0),m,o)),  i(k) = knn_idx(b,n,k)
//       (neighbor kernel input: the first neighbor is the center; out-of-range indices contribute 0)
__global__ void assign_score_withk_forward_kernel(const int B, const int N0, const int N1,
                                                  const int M, const int K, const int O,
                                                  const float* points,
                                                  const float* centers,
                                                  const float* scores,
                                                  const int64_t* knn_idx,
                                                  float* output) {

    // ----- parallel loop for B, N1, K and O ---------
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)B*N1*K*O) return;
    int b = (int)(i / (O * N1 * K));
    int o = (int)(i % (O * N1 * K) / (N1 * K));
    int n = (int)(i % (N1 * K) / K);
    int k = (int)(i % K);
    int cn = (int) knn_idx[b*K*N1 + n*K + 0]; //The first neighbor is the center point
    int kn = (int) knn_idx[b*K*N1 + n*K + k];
    float acc = 0.0f;
    if (kn < N0 && kn >= 0) { // if index overflows, it is out of the neighborhood range
        // ------- loop for M ----------
        for (int m = 0; m < M; m++) {
            acc += points[b*N0*M*O + kn*M*O + m*O + o] * scores[b*N1*K*M + n*K*M + k*M + m]
                - centers[b*N0*M*O + cn*M*O + m*O + o] * scores[b*N1*K*M + n*K*M + k*M + m];
        }
    }
    // The original accumulated with atomicAdd into a zero-initialised output.
    output[b*N1*O*K + o*N1*K + n*K + k] = acc;
}

void run(const at::Tensor& scores, const at::Tensor& point_features, const at::Tensor& center_features,
         const at::Tensor& knn_idx, at::Tensor& out) {
    TORCH_CHECK(scores.is_cuda() && point_features.is_cuda() && center_features.is_cuda() &&
                knn_idx.is_cuda() && out.is_cuda(), "tensors must be on the HIP device");
    TORCH_CHECK(scores.scalar_type() == at::kFloat && point_features.scalar_type() == at::kFloat &&
                center_features.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat,
                "float32 features");
    TORCH_CHECK(knn_idx.scalar_type() == at::kLong, "knn_idx must be int64");
    TORCH_CHECK(scores.is_contiguous() && point_features.is_contiguous() && center_features.is_contiguous() &&
                knn_idx.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    const int B = point_features.size(0);
    const int N0 = point_features.size(1);
    const int M = point_features.size(2);
    const int O = point_features.size(3);
    const int N1 = scores.size(1);
    const int K = scores.size(2);

    dim3 blocks(DIVUP((long)B*O*N1*K, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(assign_score_withk_forward_kernel, blocks, threads, 0, stream,
                       B, N0, N1, M, K, O, point_features.data_ptr<float>(),
                       center_features.data_ptr<float>(), scores.data_ptr<float>(),
                       knn_idx.data_ptr<int64_t>(), out.data_ptr<float>());
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor scores, Tensor point_features, Tensor center_features, Tensor knn_idx, Tensor(a!) out) -> ()", &run);
}
