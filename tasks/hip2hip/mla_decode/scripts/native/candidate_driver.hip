// Protected translation-unit boundary for the editable device implementation.
// Constants, decoder and kernel body match the original launch contract.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>

using bf16 = __hip_bfloat16;

namespace {

constexpr int NHEAD         = 128;
constexpr int BLOCK_H       = 16;
constexpr int HEAD_GROUPS   = NHEAD / BLOCK_H;  // 8
constexpr int LK            = 576;
constexpr int LV            = 512;

constexpr int WARP_SIZE     = 64;
constexpr int BLOCK_THREADS = WARP_SIZE;        // single wave per workgroup
constexpr int DIMS_PER_LANE = LV / WARP_SIZE;   // 8 (output dim partition)
constexpr int LK_PER_LANE   = LK / WARP_SIZE;   // 9 (QK dot partition)
static_assert(LV  % WARP_SIZE == 0, "LV must divide WARP_SIZE");
static_assert(LK  % WARP_SIZE == 0, "LK must divide WARP_SIZE");

}  // namespace

// ---------------------------------------------------------------------------
// fp8 e4m3fn helpers (bias = 7). Match torch.float8_e4m3fn semantics.
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ float fp8_e4m3fn_to_f32(uint8_t b) {
    if (b == 0x80 || b == 0x00) return 0.0f;
    int sign = (b & 0x80) ? -1 : 1;
    int e = (b >> 3) & 0xF;
    int m = b & 0x7;
    float val;
    if (e == 0) {
        val = static_cast<float>(m) / 8.0f / 64.0f;  // subnormal: m * 2^-9
    } else {
        float mantissa = 1.0f + static_cast<float>(m) / 8.0f;
#ifdef __HIP_DEVICE_COMPILE__
        val = mantissa * exp2f(static_cast<float>(e) - 7);
#else
        val = mantissa * std::exp2f(static_cast<float>(e) - 7);
#endif
    }
    return sign * val;
}


#include "../../source/kernel.hpp"
