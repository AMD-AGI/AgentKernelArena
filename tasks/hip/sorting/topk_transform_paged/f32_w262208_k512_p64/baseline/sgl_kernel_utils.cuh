/// \file sgl_kernel_utils.cuh
/// \brief The device utilities the DeepSeek-V4 top-k kernels use, for ROCm.
///
/// Vendored from SGLang (sgl-project/sglang, Apache-2.0), python/sglang/kernels/jit/include/
/// sgl_kernel/{utils,type,vec,warp}.cuh. Only the definitions topk_impl.cuh and kernel.hip use are
/// kept, flattened into one header with their `USE_ROCM` branches selected: the fp16 conversion
/// trait, AlignedVector, the warp sum and the (no-op on ROCm) PDL helpers. `kWarpThreads` stays 32,
/// so on wave64 hardware each wave holds two logical 32-lane warps, exactly as upstream.

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef __grid_constant__
#define __grid_constant__
#endif

namespace sglang {

using fp32_t = float;
using fp16_t = __half;

namespace device {

/// \brief Macro: forced-inline device function qualifier.
#define SGL_DEVICE __forceinline__ __device__

// Maximum vector size in bytes supported by current architecture (AMD: 128-bit).
inline constexpr std::size_t kMaxVecBytes = 16;

/// \brief Number of threads per (logical) warp: always 32 in sgl_kernel.
inline constexpr auto kWarpThreads = 32u;
/// \brief Full wave active mask (64 lanes on ROCm).
inline constexpr auto kFullMask = 0xffffffffffffffffULL;

/// \brief PDL (Programmatic Dependent Launch) wait: a no-op on ROCm.
template <bool kUsePDL>
SGL_DEVICE void PDLWaitPrimary() {}

/// \brief PDL trigger of the dependent launch: a no-op on ROCm.
template <bool kUsePDL>
SGL_DEVICE void PDLTriggerSecondary() {}

/// PTX pragma that lets the compiler spill registers into shared memory: CUDA 13 only.
SGL_DEVICE void enable_smem_spilling() {}

}  // namespace device

// --- type.cuh: the float <-> fp16 conversions `device::cast` dispatches to ---

template <typename T>
struct DTypeTrait {};

template <>
struct DTypeTrait<fp32_t> {
  SGL_DEVICE static fp32_t from(const fp16_t& x) {
    return __half2float(x);
  }
};

template <>
struct DTypeTrait<fp16_t> {
  SGL_DEVICE static fp16_t from(const fp32_t& x) {
    return __float2half_rn(x);
  }
};

namespace device {

/**
 * \brief Cast a value from type `From` to type `To` on device.
 *
 * Dispatches through `DTypeTrait<To>::from()`, which uses the appropriate
 * intrinsic (e.g. `__half2float`, `__float2half_rn`).
 */
template <typename To, typename From>
SGL_DEVICE To cast(const From& value) {
  return DTypeTrait<To>::from(value);
}

// --- vec.cuh ---

namespace details {

/// \brief Maps byte-width to the corresponding unsigned integer type.
template <std::size_t N>
struct uint_trait {};

template <>
struct uint_trait<1> {
  using type = uint8_t;
};

template <>
struct uint_trait<2> {
  using type = uint16_t;
};

template <>
struct uint_trait<4> {
  using type = uint32_t;
};

template <>
struct uint_trait<8> {
  using type = uint64_t;
};

/// \brief Alias: maps `sizeof(T)` to matching unsigned int type.
template <typename T>
using sized_int = typename uint_trait<sizeof(T)>::type;

}  // namespace details

/// \brief Raw aligned storage for `N` elements of type `T`.
template <typename T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedStorage {
  T data[N];
};

/**
 * \brief Aligned vector for vectorized memory access on GPU.
 *
 * Stores `N` elements of type `T` with natural alignment so that a single
 * `load`/`store` call compiles to a wide memory transaction.
 */
template <typename T, std::size_t N>
struct AlignedVector {
 private:
  static_assert(
      (N > 0 && (N & (N - 1)) == 0) && sizeof(T) * N <= kMaxVecBytes,
      "vector size exceeds arch limit: max 16 bytes on AMD");
  using element_t = typename details::sized_int<T>;
  using storage_t = AlignedStorage<element_t, N>;

 public:
  /// \brief Vectorized load from `ptr` at the given element `offset`.
  SGL_DEVICE void load(const void* ptr, int64_t offset = 0) {
    m_storage = reinterpret_cast<const storage_t*>(ptr)[offset];
  }
  /// \brief Vectorized store to `ptr` at the given element `offset`.
  SGL_DEVICE void store(void* ptr, int64_t offset = 0) const {
    reinterpret_cast<storage_t*>(ptr)[offset] = m_storage;
  }
  /// \brief Fill all N elements with the same `value`.
  SGL_DEVICE void fill(T value) {
    const auto store_value = *reinterpret_cast<element_t*>(&value);
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      m_storage.data[i] = store_value;
    }
  }

  SGL_DEVICE auto operator[](std::size_t idx) -> T& {
    return reinterpret_cast<T*>(&m_storage)[idx];
  }
  SGL_DEVICE auto operator[](std::size_t idx) const -> T {
    return reinterpret_cast<const T*>(&m_storage)[idx];
  }
  SGL_DEVICE auto data() -> T* {
    return reinterpret_cast<T*>(&m_storage);
  }
  SGL_DEVICE auto data() const -> const T* {
    return reinterpret_cast<const T*>(&m_storage);
  }

 private:
  storage_t m_storage;
};

// --- warp.cuh: warp::reduce_sum, the HIP branch of warp::reduce<SUM> ---

namespace warp {

/**
 * \brief Warp-level sum over groups of `kNumThreads` lanes.
 *
 * On HIP: uses __shfl_xor with explicit width parameter (supports wave64 sub-groups).
 */
template <uint32_t kNumThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_sum(T value) {
  static_assert(kNumThreads >= 1 && kNumThreads <= kWarpThreads);
  static_assert((kNumThreads & (kNumThreads - 1)) == 0, "must be pow of 2");
#pragma unroll
  for (uint32_t mask = kNumThreads / 2; mask >= 1; mask >>= 1) {
    value = static_cast<T>(value + __shfl_xor(value, mask, kNumThreads));
  }
  return value;
}

}  // namespace warp

}  // namespace device

}  // namespace sglang
