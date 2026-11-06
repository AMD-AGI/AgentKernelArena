// SPDX-License-Identifier: MIT
// Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
//
// Vendored for AgentKernelArena from ROCm/aiter at commit 652f2d4b768239aefd515c929e84817681ff598d:
// the host code of csrc/cpp_itfs/pa/pa_ragged.cpp.jinja rendered for this scenario (bfloat16 query,
// cache and output, gqa_ratio 8, head_size 128, block_size 1, partition_size 256, mtp 1, no ALiBi,
// no logits soft cap, the default GOLDEN kernel) and the argument handling of
// csrc/cpp_itfs/pa/pa_ragged.py::paged_attention_ragged for SGLang's decode call (one-token pages,
// NHD cache; max_num_partitions from the batch's longest context, as vLLM's ROCm decode path sizes
// it, where SGLang passes the partition count of the model's context length). The kernels are in pa_ragged.cuh and
// pa_kernels.cuh. Modifications: the host function takes ATen tensors and launches on the current
// HIP stream; the per-partition workspace is allocated per call; the reduce kernel's npar_loops is
// selected at run time from the partition count instead of compiling one library per value.

// AITER compiles these sources with -U__HIP_NO_HALF_OPERATORS__ -U__HIP_NO_HALF_CONVERSIONS__;
// torch's extension builder defines both.
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include "pa_ragged.cuh"

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <torch/library.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kHeadSize      = 128;
constexpr int kGqaRatio      = 8;
constexpr int kBlockSize     = 1;   // SGLang's token-granular page table
constexpr int kPartitionSize = 256; // _AITER_PARTITION_SIZE_ROCM
constexpr int kMtp           = 1;
constexpr int kMaxNparLoops  = 8;   // AITER's AOT range of npar_loops

using scalar_t = __hip_bfloat16;

template <int NPAR_LOOPS>
void launch_reduce(dim3 grid, dim3 block, hipStream_t stream, scalar_t* out, float* exp_sums,
                   float* max_logits, scalar_t* tmp_out, int* kv_indptr, int max_num_partitions)
{
    // The last template argument (ENABLE_LAST_PAGE_LENS) is pa_ragged.cpp.jinja's rendering of
    // "block_size greater than 1", i.e. false for one-token pages. It is written as a literal, as AITER
    // renders it: torch's hipify rewrites triple-chevron launches into hipLaunchKernelGGL and its
    // template-argument scanner fails on a comparison operator inside the template arguments.
    static_assert(kBlockSize == 1, "ENABLE_LAST_PAGE_LENS below is rendered for block_size 1");
    paged_attention_ll4mi_reduce_kernel<scalar_t, scalar_t, kHeadSize, kHeadSize, kPartitionSize, NPAR_LOOPS,
                                        false>
        <<<grid, block, 0, stream>>>(out, exp_sums, max_logits, tmp_out, kv_indptr,
                                     /*kv_last_page_lens=*/nullptr, kBlockSize, max_num_partitions,
                                     /*fp8_out_scale_ptr=*/nullptr);
}

} // namespace

void run(const at::Tensor& query, const at::Tensor& key_cache, const at::Tensor& value_cache,
         const at::Tensor& kv_indptr, const at::Tensor& kv_indices, double scale, int64_t max_context_len,
         at::Tensor& out)
{
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key_cache.scalar_type() == at::kBFloat16 &&
                    value_cache.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
                "bfloat16 query, cache and output expected");
    TORCH_CHECK(kv_indptr.scalar_type() == at::kInt && kv_indices.scalar_type() == at::kInt,
                "int32 kv_indptr and kv_indices expected");
    TORCH_CHECK(query.dim() == 3 && key_cache.dim() == 3 && value_cache.dim() == 3,
                "query [seqs, heads, head_size] and caches [slots, kv_heads, head_size] expected");
    TORCH_CHECK(query.is_contiguous() && key_cache.is_contiguous() && value_cache.is_contiguous() &&
                    out.is_contiguous() && kv_indptr.is_contiguous() && kv_indices.is_contiguous(),
                "contiguous tensors expected");
    const int num_seqs     = static_cast<int>(query.size(0));
    const int num_heads    = static_cast<int>(query.size(1));
    const int head_size    = static_cast<int>(query.size(2));
    const int num_kv_heads = static_cast<int>(key_cache.size(1));
    TORCH_CHECK(head_size == kHeadSize && key_cache.size(2) == kHeadSize, "head_size 128 expected");
    TORCH_CHECK(num_heads == kGqaRatio * num_kv_heads, "8 query heads per KV head expected");
    TORCH_CHECK(max_context_len >= 1, "max_context_len must be positive");

    // pa_ragged.py: strides of the cache viewed as [-1, block_size=1, num_kv_heads, head_size] (NHD).
    const int q_stride        = static_cast<int>(query.stride(0));
    const int kv_block_stride = static_cast<int>(key_cache.stride(0));
    const int kv_head_stride  = static_cast<int>(key_cache.stride(1));
    const int kv_seq_stride   = static_cast<int>(key_cache.stride(0));
    const int max_num_partitions =
        static_cast<int>((max_context_len + kPartitionSize - 1) / kPartitionSize);
    const int warp_size  = 64;
    const int npar_loops = (max_num_partitions + warp_size - 1) / warp_size;
    TORCH_CHECK(npar_loops <= kMaxNparLoops, "max_context_len exceeds ", kMaxNparLoops * warp_size * kPartitionSize);

    // Workspace laid out as pa_ragged.cpp.jinja carves it: exp_sums, max_logits, tmp_out.
    const int64_t stats = static_cast<int64_t>(num_seqs) * num_heads * max_num_partitions;
    at::Tensor workspace =
        at::empty({2 * stats * static_cast<int64_t>(sizeof(float)) +
                       stats * kHeadSize * static_cast<int64_t>(sizeof(scalar_t))},
                  query.options().dtype(at::kByte));
    float* exp_sums_ptr   = reinterpret_cast<float*>(workspace.data_ptr());
    float* max_logits_ptr = exp_sums_ptr + stats;
    scalar_t* tmp_out_ptr = reinterpret_cast<scalar_t*>(max_logits_ptr + stats);

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    int* kv_indptr_ptr  = kv_indptr.data_ptr<int>();
    int* kv_indices_ptr = kv_indices.data_ptr<int>();
    scalar_t* query_ptr = reinterpret_cast<scalar_t*>(query.data_ptr());
    scalar_t* k_ptr     = reinterpret_cast<scalar_t*>(key_cache.data_ptr());
    scalar_t* v_ptr     = reinterpret_cast<scalar_t*>(value_cache.data_ptr());
    scalar_t* out_ptr   = reinterpret_cast<scalar_t*>(out.data_ptr());

    const float logits_soft_cap     = 0.f;
    const float logits_soft_cap_rcp = 0.f;
    ck_tile::ComposedAttention<0> variant;
    constexpr int NTHR = 256;
    dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
    dim3 block(NTHR);
    paged_attention_ll4mi_QKV_mfma16_kernel<0, scalar_t, scalar_t, vllm::Fp8KVCacheDataType::kAuto, kBlockSize,
                                            kHeadSize, NTHR, false, kGqaRatio, kMtp>
        <<<grid, block, 0, stream>>>(query_ptr, k_ptr, v_ptr, static_cast<float>(scale), kv_indptr_ptr,
                                     kv_indices_ptr, /*kv_last_page_lens=*/nullptr, /*alibi_slopes=*/nullptr,
                                     q_stride, kv_block_stride, kv_head_stride, kv_seq_stride, exp_sums_ptr,
                                     max_logits_ptr, tmp_out_ptr, logits_soft_cap, logits_soft_cap_rcp,
                                     /*q_scale_ptr=*/nullptr, /*k_scale_ptr=*/nullptr, /*v_scale_ptr=*/nullptr,
                                     &variant);

    dim3 reduce_grid(num_heads, num_seqs, kMtp);
    dim3 reduce_block(kHeadSize);
    switch(npar_loops)
    {
    case 1: launch_reduce<1>(reduce_grid, reduce_block, stream, out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, kv_indptr_ptr, max_num_partitions); break;
    case 2: launch_reduce<2>(reduce_grid, reduce_block, stream, out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, kv_indptr_ptr, max_num_partitions); break;
    case 3: launch_reduce<3>(reduce_grid, reduce_block, stream, out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, kv_indptr_ptr, max_num_partitions); break;
    case 4: launch_reduce<4>(reduce_grid, reduce_block, stream, out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, kv_indptr_ptr, max_num_partitions); break;
    case 5: launch_reduce<5>(reduce_grid, reduce_block, stream, out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, kv_indptr_ptr, max_num_partitions); break;
    case 6: launch_reduce<6>(reduce_grid, reduce_block, stream, out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, kv_indptr_ptr, max_num_partitions); break;
    case 7: launch_reduce<7>(reduce_grid, reduce_block, stream, out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, kv_indptr_ptr, max_num_partitions); break;
    default: launch_reduce<8>(reduce_grid, reduce_block, stream, out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, kv_indptr_ptr, max_num_partitions); break;
    }
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m)
{
    m.def("run(Tensor query, Tensor key_cache, Tensor value_cache, Tensor kv_indptr, Tensor kv_indices, "
          "float scale, int max_context_len, Tensor(a!) out) -> ()",
          &run);
}
