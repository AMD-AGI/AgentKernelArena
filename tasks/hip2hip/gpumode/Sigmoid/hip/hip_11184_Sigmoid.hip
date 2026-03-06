// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

// Sigmoid matching PyTorch default formula (no fast-math)
__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ __forceinline__ double sigmoid_f64(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Vector type mapping
template <typename T, int Pack> struct VecT {};
template <> struct VecT<float, 4> { using type = float4; };
template <> struct VecT<double, 2> { using type = double2; };

// Load/store helpers
template <typename V>
__device__ __forceinline__ V vload(const V* __restrict__ p, int64_t idx) { return p[idx]; }

template <typename V>
__device__ __forceinline__ void vstore(V* __restrict__ p, int64_t idx, const V& v) { p[idx] = v; }

// Scalar compute helpers
template <typename T>
__device__ __forceinline__ T apply_sigmoid_scaled(T v, float a, float maxv);

template <>
__device__ __forceinline__ float apply_sigmoid_scaled<float>(float v, float a, float maxv) {
    return sigmoid_f32(a * v) * maxv;
}

template <>
__device__ __forceinline__ double apply_sigmoid_scaled<double>(double v, float a, float maxv) {
    return sigmoid_f64(static_cast<double>(a) * v) * static_cast<double>(maxv);
}

// Fused kernel: scalar head -> vectorized bulk -> scalar tail
// Pack: 4 for float, 2 for double; UNROLL processes consecutive packs per thread
// TB: threads-per-block specialization with launch_bounds

template <typename T, int Pack, int UNROLL, int TB>
__global__ __launch_bounds__(TB, 2)
void sigmoid_mul_fused_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    int64_t n,
    int64_t head,
    int64_t n_packs,
    int64_t tail,
    float a,
    float maxv)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t gstride = (int64_t)blockDim.x * gridDim.x;

    // 1) Scalar head: unroll by 4, consecutive
    for (int64_t base = tid * 4; base < head; base += gstride * 4) {
        int64_t i0 = base + 0;
        int64_t i1 = base + 1;
        int64_t i2 = base + 2;
        int64_t i3 = base + 3;
        if (i0 < head) { T v = in[i0]; out[i0] = apply_sigmoid_scaled<T>(v, a, maxv); }
        if (i1 < head) { T v = in[i1]; out[i1] = apply_sigmoid_scaled<T>(v, a, maxv); }
        if (i2 < head) { T v = in[i2]; out[i2] = apply_sigmoid_scaled<T>(v, a, maxv); }
        if (i3 < head) { T v = in[i3]; out[i3] = apply_sigmoid_scaled<T>(v, a, maxv); }
    }

    // 2) Vectorized bulk over packs
    if (n_packs > 0) {
        using V = typename VecT<T, Pack>::type;
        const T* in_bulk = in + head;
        T* out_bulk = out + head;
        const V* __restrict__ vin = reinterpret_cast<const V*>(in_bulk);
        V* __restrict__ vout = reinterpret_cast<V*>(out_bulk);

        for (int64_t p = tid * UNROLL; p < n_packs; p += gstride * UNROLL) {
            #pragma unroll
            for (int k = 0; k < UNROLL; ++k) {
                int64_t q = p + k;
                if (q >= n_packs) break;
                if constexpr (std::is_same<T, float>::value && Pack == 4) {
                    float4 vv = vload(vin, q);
                    float4 rr;
                    rr.x = sigmoid_f32(a * vv.x) * maxv;
                    rr.y = sigmoid_f32(a * vv.y) * maxv;
                    rr.z = sigmoid_f32(a * vv.z) * maxv;
                    rr.w = sigmoid_f32(a * vv.w) * maxv;
                    vstore(vout, q, rr);
                } else {
                    double2 vv = vload(vin, q);
                    double2 rr;
                    double ad = static_cast<double>(a);
                    double md = static_cast<double>(maxv);
                    rr.x = sigmoid_f64(ad * vv.x) * md;
                    rr.y = sigmoid_f64(ad * vv.y) * md;
                    vstore(vout, q, rr);
                }
            }
        }
    }

    // 3) Scalar tail: unroll by 4, consecutive
    if (tail > 0) {
        const int64_t tail_off = head + n_packs * Pack;
        for (int64_t base = tid * 4; base < tail; base += gstride * 4) {
            int64_t t0 = base + 0;
            int64_t t1 = base + 1;
            int64_t t2 = base + 2;
            int64_t t3 = base + 3;
            if (t0 < tail) { int64_t idx = tail_off + t0; T v = in[idx]; out[idx] = apply_sigmoid_scaled<T>(v, a, maxv); }
            if (t1 < tail) { int64_t idx = tail_off + t1; T v = in[idx]; out[idx] = apply_sigmoid_scaled<T>(v, a, maxv); }
            if (t2 < tail) { int64_t idx = tail_off + t2; T v = in[idx]; out[idx] = apply_sigmoid_scaled<T>(v, a, maxv); }
            if (t3 < tail) { int64_t idx = tail_off + t3; T v = in[idx]; out[idx] = apply_sigmoid_scaled<T>(v, a, maxv); }
        }
    }
}

// Scalar fallback kernel with unroll-by-4

template <typename T, int TB>
__global__ __launch_bounds__(TB, 2)
void sigmoid_mul_scalar_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    int64_t n,
    float a,
    float maxv)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t base = tid * 4; base < n; base += gstride * 4) {
        int64_t i0 = base + 0;
        int64_t i1 = base + 1;
        int64_t i2 = base + 2;
        int64_t i3 = base + 3;
        if (i0 < n) { T v = in[i0]; out[i0] = apply_sigmoid_scaled<T>(v, a, maxv); }
        if (i1 < n) { T v = in[i1]; out[i1] = apply_sigmoid_scaled<T>(v, a, maxv); }
        if (i2 < n) { T v = in[i2]; out[i2] = apply_sigmoid_scaled<T>(v, a, maxv); }
        if (i3 < n) { T v = in[i3]; out[i3] = apply_sigmoid_scaled<T>(v, a, maxv); }
    }
}

// Env helpers
static inline int env_get_int(const char* name, int defv) {
    const char* s = std::getenv(name);
    if (!s) return defv;
    int v = std::atoi(s);
    return v > 0 ? v : defv;
}

static inline int choose_threads_from_env() {
    int tpb = env_get_int("HIP_SIGMOID_TPB", 512);
    // Clamp to multiples of 64 and reasonable limits
    tpb = (tpb + 63) / 64 * 64;
    if (tpb < 64) tpb = 64;
    if (tpb > 1024) tpb = 1024;
    return tpb;
}

static inline int compute_blocks_vec(int64_t n_packs, int threads, int unroll) {
    if (n_packs <= 0) return 1;
    hipDeviceProp_t prop{}; int dev = 0; hipGetDevice(&dev); hipGetDeviceProperties(&prop, dev);
    int cu = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    int blocks_per_cu = env_get_int("HIP_SIGMOID_BLOCKS_PER_CU", 24);
    if (blocks_per_cu < 4) blocks_per_cu = 4; if (blocks_per_cu > 64) blocks_per_cu = 64;
    int64_t need = (n_packs + (int64_t)threads * unroll - 1) / ((int64_t)threads * unroll);
    int64_t cap = (int64_t)cu * blocks_per_cu;
    if (need > cap) need = cap;
    return (int)(need > 0 ? need : 1);
}

static inline int compute_blocks_scalar(int64_t n, int threads) {
    if (n <= 0) return 1;
    hipDeviceProp_t prop{}; int dev = 0; hipGetDevice(&dev); hipGetDeviceProperties(&prop, dev);
    int cu = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    int blocks_per_cu = env_get_int("HIP_SIGMOID_BLOCKS_PER_CU", 24);
    if (blocks_per_cu < 4) blocks_per_cu = 4; if (blocks_per_cu > 64) blocks_per_cu = 64;
    int64_t need = (n + (int64_t)threads * 4 - 1) / ((int64_t)threads * 4); // unroll-by-4
    int64_t cap = (int64_t)cu * blocks_per_cu;
    if (need > cap) need = cap;
    return (int)(need > 0 ? need : 1);
}

// Compute a head in [0, Pack-1] to make both in and out aligned to 16B
// Returns true if found; sets head_out (clamped to n)

template <typename T, int Pack>
static inline bool compute_dual_alignment_head(const T* in, const T* out, int64_t n, int64_t& head_out) {
    constexpr int align = 16;
    if (n <= 0) { head_out = 0; return false; }
    uintptr_t ia = reinterpret_cast<uintptr_t>(in);
    uintptr_t oa = reinterpret_cast<uintptr_t>(out);
    for (int h = 0; h < Pack && h < n; ++h) {
        uintptr_t ip = ia + (uintptr_t)h * sizeof(T);
        uintptr_t op = oa + (uintptr_t)h * sizeof(T);
        if ((ip % align == 0u) && (op % align == 0u)) { head_out = h; return true; }
    }
    head_out = 0;
    return false;
}

// TB-specialized launcher

template <typename F>
static inline void launch_with_tb(int threads, F&& fn) {
    switch (threads) {
        case 128: fn(std::integral_constant<int,128>{}); break;
        case 256: fn(std::integral_constant<int,256>{}); break;
        case 512: fn(std::integral_constant<int,512>{}); break;
        case 1024: fn(std::integral_constant<int,1024>{}); break;
        default:  fn(std::integral_constant<int,256>{}); break; // fallback
    }
}

// Forward API: forward(v, a, max)
torch::Tensor forward(torch::Tensor v, float a = 1.0f, float max = 10.0f) {
    TORCH_CHECK(v.is_cuda() || v.is_hip(), "Tensor must be on HIP (ROCm) device");
    TORCH_CHECK(v.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(v.scalar_type() == at::kFloat || v.scalar_type() == at::kDouble,
                "Only float32 and float64 tensors are supported");

    auto out = torch::empty_like(v);
    const int64_t n = v.numel();
    if (n == 0) return out;

    hipStream_t stream = c10::hip::getCurrentHIPStream();
    const int threads = choose_threads_from_env();

    if (v.scalar_type() == at::kFloat) {
        const float* in_ptr = v.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        constexpr int Pack = 4; // float4
        constexpr int UNROLL = 4;
        int64_t head = 0;
        bool can_vec = compute_dual_alignment_head<float, Pack>(in_ptr, out_ptr, n, head);
        int64_t n_packs = 0, tail = 0;
        if (can_vec) {
            int64_t eligible = n - head;
            n_packs = eligible / Pack;
            tail = eligible - n_packs * Pack;
        }

        if (can_vec && n_packs > 0) {
            int blocks = compute_blocks_vec(n_packs, threads, UNROLL);
            launch_with_tb(threads, [&](auto TB_CONST){
                constexpr int TB = decltype(TB_CONST)::value;
                hipLaunchKernelGGL((sigmoid_mul_fused_kernel<float, Pack, UNROLL, TB>),
                    dim3(blocks), dim3(TB), 0, stream,
                    in_ptr, out_ptr, n, head, n_packs, tail, a, max);
            });
        } else {
            int blocks = compute_blocks_scalar(n, threads);
            launch_with_tb(threads, [&](auto TB_CONST){
                constexpr int TB = decltype(TB_CONST)::value;
                hipLaunchKernelGGL((sigmoid_mul_scalar_kernel<float, TB>),
                    dim3(blocks), dim3(TB), 0, stream,
                    in_ptr, out_ptr, n, a, max);
            });
        }
    } else {
        const double* in_ptr = v.data_ptr<double>();
        double* out_ptr = out.data_ptr<double>();

        constexpr int Pack = 2; // double2
        constexpr int UNROLL = 4;
        int64_t head = 0;
        bool can_vec = compute_dual_alignment_head<double, Pack>(in_ptr, out_ptr, n, head);
        int64_t n_packs = 0, tail = 0;
        if (can_vec) {
            int64_t eligible = n - head;
            n_packs = eligible / Pack;
            tail = eligible - n_packs * Pack;
        }

        if (can_vec && n_packs > 0) {
            int blocks = compute_blocks_vec(n_packs, threads, UNROLL);
            launch_with_tb(threads, [&](auto TB_CONST){
                constexpr int TB = decltype(TB_CONST)::value;
                hipLaunchKernelGGL((sigmoid_mul_fused_kernel<double, Pack, UNROLL, TB>),
                    dim3(blocks), dim3(TB), 0, stream,
                    in_ptr, out_ptr, n, head, n_packs, tail, a, max);
            });
        } else {
            int blocks = compute_blocks_scalar(n, threads);
            launch_with_tb(threads, [&](auto TB_CONST){
                constexpr int TB = decltype(TB_CONST)::value;
                hipLaunchKernelGGL((sigmoid_mul_scalar_kernel<double, TB>),
                    dim3(blocks), dim3(TB), 0, stream,
                    in_ptr, out_ptr, n, a, max);
            });
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid * max forward (HIP)",
          pybind11::arg("v"), pybind11::arg("a") = 1.0f, pybind11::arg("max") = 10.0f);
}

