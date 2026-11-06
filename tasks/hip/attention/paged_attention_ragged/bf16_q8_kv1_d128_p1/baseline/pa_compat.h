// SPDX-License-Identifier: MIT
// Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// The definitions the vendored AITER paged-attention kernel takes from AITER's
// aiter_hip_common.h and dtype_fp8.cuh and from Composable Kernel's ck_tile fmha
// block_masking.hpp / variants.hpp, reduced to what this kernel uses.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <type_traits>

// aiter_hip_common.h: the wavefront size, 64 on gfx9 (the host pass also uses 64).
struct WarpSizeValue
{
    __host__ __device__ constexpr operator int() const
    {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX9__)
        return 32;
#else
        return 64;
#endif
    }
};
inline constexpr WarpSizeValue WARP_SIZE{};

// dtype_fp8.cuh (vLLM / AITER, Apache-2.0)
namespace vllm {

enum class Fp8KVCacheDataType
{
    kAuto    = 0,
    kFp8E4M3 = 1,
    kFp8E5M2 = 2,
};

} // namespace vllm

// ck_tile attention variant and mask types, as the kernel uses them: the kernel
// builds a non-masking mask, passes the softmax scale through the variant
// parameters and scales the QK logits with QueryTransform; LogitsTransform is the
// identity without a logits soft cap.
namespace ck_tile {

constexpr uint32_t CUSTOM_MASK     = 1U;
constexpr uint32_t SLIDING_WINDOW  = 2U;
constexpr uint32_t LOGITS_SOFT_CAP = 4U;
constexpr uint32_t ALIBI           = 8U;

template <bool IsMasking_>
struct SimplifiedGenericAttentionMask
{
    static constexpr bool IsMasking = IsMasking_;

    __host__ __device__ SimplifiedGenericAttentionMask(int y_total_, int x_total_)
        : y_total(y_total_), x_total(x_total_)
    {
    }

    int y_total;
    int x_total;
};

template <typename ImplMask_>
struct StandardAttentionParams
{
    using ImplMask = std::remove_cv_t<std::remove_reference_t<ImplMask_>>;

    __host__ __device__ StandardAttentionParams(const ImplMask& impl_mask_, float sm_scale_)
        : impl_mask(impl_mask_), sm_scale(sm_scale_)
    {
    }

    const ImplMask& impl_mask;
    float sm_scale;
};

template <typename ImplMask_, bool UseExp2 = false>
struct LogitsSoftCapParams
{
    using ImplMask = std::remove_cv_t<std::remove_reference_t<ImplMask_>>;

    __host__ __device__ LogitsSoftCapParams(const ImplMask& impl_mask_,
                                            float sm_scale_,
                                            float logits_soft_cap_,
                                            float logits_soft_cap_rcp_)
        : impl_mask(impl_mask_),
          sm_scale(sm_scale_),
          logits_soft_cap(logits_soft_cap_),
          logits_soft_cap_rcp(logits_soft_cap_rcp_)
    {
    }

    const ImplMask& impl_mask;
    float sm_scale;
    float logits_soft_cap;
    float logits_soft_cap_rcp;
};

// Only the plain softmax variant (no logits soft cap) is part of this task.
template <uint32_t VARIANT_CODE, bool UseExp2 = false>
struct ComposedAttention
{
    static constexpr bool use_exp2            = UseExp2;
    static constexpr bool use_logits_soft_cap = (VARIANT_CODE & LOGITS_SOFT_CAP) != 0;
    static_assert(!use_logits_soft_cap, "the logits soft cap is not part of this task");

    template <typename Params, typename T>
    __device__ __forceinline__ T QueryTransform(const Params& params, T q) const
    {
        return static_cast<T>(static_cast<float>(q) * params.sm_scale);
    }

    template <typename Params, typename T>
    __device__ __forceinline__ T LogitsTransform(const Params& /*params*/,
                                                 T logits,
                                                 uint32_t /*batch_idx*/,
                                                 uint32_t /*qo_head_idx*/,
                                                 uint32_t /*kv_head_idx*/) const
    {
        return logits;
    }
};

} // namespace ck_tile
