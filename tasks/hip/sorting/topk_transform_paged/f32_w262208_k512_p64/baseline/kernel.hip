/**
 * \file kernel.hip
 * \brief TopK kernel for DeepSeek v4: the paged top-k v2 transform on ROCm.
 *
 * Vendored from SGLang (sgl-project/sglang, Apache-2.0),
 * python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh, the kernel behind
 * sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2 (DeepSeek-V4 decode on ROCm).
 * The device code below is upstream's with its `USE_ROCM` branches selected: the CUDA-only
 * thread-block-cluster kernels, the ragged (prefill) kernel and the routing-plan kernel (a no-op
 * on ROCm) are removed. The host dispatcher `TopKKernel::transform_paged` is rewritten from the
 * tvm-ffi TensorView interface to ATen tensors and registered as the arena `run` op; it keeps
 * upstream's checks and its ROCm launch choice (one 1024-thread block per row, level by the
 * score width).
 *
 * Upstream credits:
 * Adapted from
 * 1:
 *   https://github.com/vllm-project/vllm/blob/a8c6ee9b787d273916206a29b77feebadb80c368/csrc/persistent_topk.cuh
 * 2:
 *   https://github.com/flashinfer-ai/flashinfer/blob/c2b4db2b1a84448d802f0e6ac445243312bd6a4c/include/flashinfer/topk.cuh
 * DarkSharpness never took a detailed look at these 2 implementation, but his claude code did.
 * So we add credit to the reference implementations.
 */
// The logical-warp prefix sum uses HIP's __shfl_up_sync.
#ifndef HIP_ENABLE_WARP_SYNC_BUILTINS
#define HIP_ENABLE_WARP_SYNC_BUILTINS 1
#endif

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>

#include "topk_impl.cuh"

namespace sglang {

namespace impl = device::topk;
using impl::TopKProblem;

enum class TopKMode {
  INDICES,     ///< raw selected indices into `out`; `page_table` unused
  PAGE_TABLE,  ///< page-table-transformed indices into `out`
};

using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Streaming = impl::TopKStreaming;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384

#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)

constexpr uint32_t kClusterFloor = 65536;

/// Metadata tensor rows (each 8 B / 2 int32). Row 0 is the global plan result;
/// rows 1..N are the (batch_id, seq_len) of items routed to the cluster pool.
struct alignas(8) GlobalMetadata {
  uint32_t cluster_threshold;
  uint32_t num_cluster_items;  // N = number of items routed to the cluster pool
};
struct alignas(8) PlanItem {
  uint32_t batch_id;
  uint32_t seq_len;
};
static_assert(sizeof(GlobalMetadata) == 2 * sizeof(int32_t) && sizeof(PlanItem) == sizeof(GlobalMetadata));

struct TopKPagedParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  const PlanItem* __restrict__ metadata;  // [0]=GlobalMetadata, [1+i]=PlanItem
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t topk;
  uint32_t page_bits;
  uint32_t cluster_floor;  // seq_len > this routes to the cluster path (batch-aware, host-set)

  SGL_DEVICE const GlobalMetadata& global() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  SGL_DEVICE uint32_t cluster_threshold() const {
    return global().cluster_threshold;
  }
  SGL_DEVICE const PlanItem& item(uint32_t i) const {
    return metadata[1 + i];
  }
  SGL_DEVICE int32_t* get_output_ptr(uint32_t batch_id) const {
    return page_indices + batch_id * static_cast<int64_t>(topk);
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id, uint32_t seq_len) const {
    const auto k = static_cast<int64_t>(topk);
    return TopKProblem{
        .in = scores + batch_id * score_stride,
        .out = page_indices + batch_id * k,
        .page_table = page_table + batch_id * page_table_stride,
        .topk = topk,
        .seq_len = seq_len,
        .page_bits = page_bits,
    };
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id) const {
    return this->problem(batch_id, static_cast<uint32_t>(seq_lens[batch_id]));
  }
};


template <typename F>
SGL_DEVICE void for_each_item(uint32_t topk, const F& f) {
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
#pragma unroll
  for (uint32_t i = 0; i < kNumElems; ++i) {
    if (const auto tx = i * kBlockSize + threadIdx.x; tx < topk) {
      __builtin_assume(tx < kMaxTopK);
      f(tx, i);
    }
  }
}

template <bool kPDL, TopKMode kMode>
SGL_DEVICE void trivial_transform(const TopKProblem& problem) {
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t) {
    const auto idx = tx < problem.seq_len ? static_cast<int32_t>(tx) : -1;
    if constexpr (kMode == TopKMode::INDICES) {
      problem.emit(tx, idx);
    } else {
      problem.transform_output(tx, idx);
    }
  });
}

SGL_DEVICE void problem_transform(TopKProblem& problem, int32_t* output_ptr) {
  static_assert(kMaxTopK % kBlockSize == 0);
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
  int32_t source_index[kNumElems];
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { source_index[i] = problem.out[tx]; });
  problem.out = output_ptr;
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { problem.transform_output(tx, source_index[i]); });
}

/**
 * \brief Main kernel for the short items and epilogue of long items.
 * \tparam kPDL whether to use PDL to synchronize with the cluster kernel (if any)
 * \tparam kLevel:
 * - Level 0: max_seq_len <= 8192           -> trivial + register<2>
 * - Level 1: max_seq_len <= 16384          -> trivial + register<4>
 * - Level 2: max_seq_len <= cluster_floor  -> trivial + register<4> + streaming
 * - Level 3: max_seq_len > cluster_floor   -> + epilogue process of cluster path
 */
template <bool kPDL, int kLevel, TopKMode kMode>
TOPK_KERNEL void topk_main_kernel(const __grid_constant__ TopKPagedParams params) {
  device::enable_smem_spilling();
  auto problem = params.problem(blockIdx.x);
  constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();
  constexpr bool kHandleCluster = (kLevel == 3);
  // Only the cluster path consumes the cluster kernel's output, so only it waits
  // on that kernel (kPDLFinal). Every other path waits at most on the indexer
  // (kPDLEarly) and must not be held on an SM slot until the long-running
  // persistent pool retires -- that would serialize the short items behind it.
  constexpr bool kPDLEarly = kPDL && !kHandleCluster;
  constexpr bool kPDLFinal = kPDL && kHandleCluster;
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (problem.seq_len <= problem.topk) return trivial_transform<kPDLEarly, kMode>(problem);

  constexpr bool kNeedStaging = kMode != TopKMode::INDICES;
  __shared__ int32_t s_topk_indices[kNeedStaging ? kMaxTopK : 1];
  if constexpr (kNeedStaging) problem.out = s_topk_indices;

  // non-trivial path: dispatch based on level and seq_len
  const auto cluster_threshold = kHandleCluster ? params.cluster_threshold() : kU32Max;
  if constexpr (kLevel == 0) {
    __builtin_assume(problem.seq_len <= kReg2MaxSeqLen);
    Register2::forward<kPDL>(problem, &smem);
  } else if constexpr (kLevel == 1) {
    __builtin_assume(problem.seq_len <= kReg4MaxSeqLen);
    Register4::forward<kPDL>(problem, &smem);  // max_seq_len <= 16384 guarantees seq <= 16384
  } else {
    static_assert(kLevel == 2 || kLevel == 3, "we only support level = 0,1,2,3 now");
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<kPDLEarly>(problem, &smem);
    } else if (problem.seq_len <= cluster_threshold) {
      Streaming::forward<kPDLEarly>(problem, &smem);
    } else {
      // Cluster path: the pool already selected into our output row; the only
      // work left is the epilogue, so this is the one path that waits for it.
      problem.out = params.get_output_ptr(blockIdx.x);
      device::PDLWaitPrimary<kPDLFinal>();
    }
  }

  device::PDLTriggerSecondary<kPDL>();
  if constexpr (kNeedStaging) {
    __syncthreads();
    problem_transform(problem, params.get_output_ptr(blockIdx.x));
  }
}

}  // namespace sglang

// --- Host: TopKKernel::transform_paged (ROCm), as the arena run op ----------------------------

void run(const at::Tensor& scores, const at::Tensor& seq_lens, const at::Tensor& page_tables, int64_t page_size,
         const at::Tensor& metadata, at::Tensor& out_page_indices) {
  using namespace sglang;
  TORCH_CHECK(scores.is_cuda() && scores.dim() == 2 && scores.scalar_type() == at::kFloat && scores.stride(1) == 1,
              "scores must be a float32 [batch, max_seq_len] GPU tensor with unit column stride");
  const auto B = scores.size(0);
  const auto dev = scores.device();
  TORCH_CHECK(seq_lens.device() == dev && seq_lens.dim() == 1 && seq_lens.size(0) == B &&
                  seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous(),
              "seq_lens must be int32 [batch], contiguous, on the scores device");
  TORCH_CHECK(page_tables.device() == dev && page_tables.dim() == 2 && page_tables.size(0) == B &&
                  page_tables.scalar_type() == at::kInt && page_tables.stride(1) == 1,
              "page_table must be int32 [batch, pages] with unit column stride, on the scores device");
  TORCH_CHECK(out_page_indices.device() == dev && out_page_indices.dim() == 2 && out_page_indices.size(0) == B &&
                  out_page_indices.scalar_type() == at::kInt && out_page_indices.is_contiguous(),
              "page_indices must be int32 [batch, topk], contiguous, on the scores device");
  TORCH_CHECK(metadata.device() == dev && metadata.dim() == 2 && metadata.size(0) == B + 1 &&
                  metadata.size(1) == 2 && metadata.scalar_type() == at::kInt && metadata.is_contiguous(),
              "invalid metadata shape: expected int32 [batch + 1, 2]");

  TORCH_CHECK(page_size > 0 && page_size <= std::numeric_limits<uint32_t>::max() && (page_size & (page_size - 1)) == 0,
              "page_size must be power of 2");
  TORCH_CHECK(scores.stride(0) % 4 == 0, "score_stride must be a multiple of 4 (16-byte vectorized load)");
  const auto topk = static_cast<uint32_t>(out_page_indices.size(1));
  TORCH_CHECK(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");

  const auto page_bits = static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(page_size)));
  const auto batch_size = static_cast<uint32_t>(B);
  const auto max_seq_len = static_cast<uint32_t>(scores.size(1));
  if (batch_size == 0) return;

  // The CUDA build fuses an 8-block cluster per row below this floor for small batches; ROCm has
  // no cluster path, so the floor is carried in the parameters but never consulted.
  constexpr uint32_t kClusterFloorSmall = 32768;
  constexpr uint32_t kSmallBatchLowFloor = 15;
  TopKPagedParams params{};
  params.scores = scores.data_ptr<float>();
  params.seq_lens = seq_lens.data_ptr<int32_t>();
  params.page_table = page_tables.data_ptr<int32_t>();
  params.page_indices = out_page_indices.data_ptr<int32_t>();
  params.metadata = reinterpret_cast<const PlanItem*>(metadata.data_ptr<int32_t>());
  params.score_stride = scores.stride(0);
  params.page_table_stride = page_tables.stride(0);
  params.topk = topk;
  params.page_bits = page_bits;
  params.cluster_floor = (batch_size <= kSmallBatchLowFloor) ? kClusterFloorSmall : kClusterFloor;

  constexpr bool kUsePDL = true;
  constexpr auto kMode = TopKMode::PAGE_TABLE;
  const hipStream_t stream = c10::hip::getCurrentHIPStream();
  if (max_seq_len <= kReg2MaxSeqLen) {
    hipLaunchKernelGGL((topk_main_kernel<kUsePDL, /*kLevel=*/0, kMode>), dim3(batch_size), dim3(kBlockSize), 0,
                       stream, params);
  } else if (max_seq_len <= kReg4MaxSeqLen) {
    hipLaunchKernelGGL((topk_main_kernel<kUsePDL, /*kLevel=*/1, kMode>), dim3(batch_size), dim3(kBlockSize), 0,
                       stream, params);
  } else {
    hipLaunchKernelGGL((topk_main_kernel<kUsePDL, /*kLevel=*/2, kMode>), dim3(batch_size), dim3(kBlockSize), 0,
                       stream, params);
  }
  const auto error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, "topk_transform_paged launch failed: ", hipGetErrorString(error));
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor scores, Tensor seq_lens, Tensor page_tables, int page_size, Tensor metadata, "
        "Tensor(a!) out_page_indices) -> ()",
        &run);
}
