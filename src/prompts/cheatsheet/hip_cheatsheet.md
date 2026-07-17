# HIP Kernel Best Practices

Reference: [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/) | [AMD Instinct docs](https://instinct.docs.amd.com/)

---

## 1. Memory Access — Coalescing

AMD CDNA GPUs (MI300/MI350 series) access global memory in 64-byte cache lines. A wavefront of 64 threads fetches optimally when consecutive threads access consecutive addresses.

**Good — coalesced:**
```cpp
// Thread i reads element i: one cache line per 64-byte block
float val = a[blockDim.x * blockIdx.x + threadIdx.x];
```

**Bad — strided:**
```cpp
// Each thread jumps N elements: N separate cache-line loads
float val = a[threadIdx.x * N];
```

Rules:
- Prefer Structure-of-Arrays (SoA) over Array-of-Structures (AoS).
- Device allocations from `hipMalloc` already satisfy device-allocation alignment requirements. Use `hipMallocPitch` for pitch-aware 2D layouts, and explicit alignment attributes only for host/static objects when needed.
- Use vector loads (`float4`, `half2`, `uint4`) to widen memory transactions and reduce instruction count.

---

## 2. Occupancy and Wavefront Management

The AMD CDNA compute unit (CU) schedules up to 8 wavefronts of 64 threads. High occupancy hides memory latency.

### Controlling occupancy
```cpp
// Suggest minimum wavefronts per CU to the compiler
__attribute__((amdgpu_waves_per_eu(4, 8)))
__global__ void myKernel(...) { ... }

// Hard cap on registers to boost occupancy
__attribute__((amdgpu_flat_work_group_size(64, 256)))
__global__ void myKernel(...) { ... }
```

### Key occupancy limits
| Resource per CU | CDNA3 (MI300) limit |
|----------------|---------------------|
| Wavefronts     | 32                  |
| VGPRs          | 512 per SIMD × 4    |
| SGPRs          | 800 per SIMD × 4    |
| LDS            | 64 KB               |

Use `rocm-smi --showmeminfo` and `rocprof` (or Omniperf) to measure actual occupancy.

- Block size should be a multiple of 64 (wavefront width).
- Prefer 256 threads/block for general workloads; tune with `hipOccupancyMaxPotentialBlockSize`.

---

## 3. LDS (Local Data Share / Shared Memory)

LDS provides ~100× faster bandwidth than global memory. Each CU has 64 KB.

```cpp
__global__ void tiled_gemm(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    constexpr int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    float acc = 0.f;

    for (int t = 0; t < K / TILE; ++t) {
        As[ty][tx] = A[(blockIdx.y * TILE + ty) * K + t * TILE + tx];
        Bs[ty][tx] = B[(t * TILE + ty) * N + blockIdx.x * TILE + tx];
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[(blockIdx.y * TILE + ty) * N + blockIdx.x * TILE + tx] = acc;
}
```

**Avoid bank conflicts:** LDS has 32 banks (4-byte each). Threads within a wavefront that map to the same bank serialize. Pad shared arrays:
```cpp
__shared__ float tile[TILE][TILE + 1]; // +1 avoids 32-way conflict
```

---

## 4. Register Pressure and Spilling

Each SIMD unit has 512 VGPRs. Kernels using >64 VGPRs per thread reduce maximum wavefronts per CU from 8 to 4 (or fewer). Register spilling to scratch memory adds ~500 cycle round-trip latency.

**Check register usage:**
```bash
hipcc -O3 --save-temps --offload-arch=gfx942 kernel.cpp
# Read the .s assembly for v_readlane / s_load_dword (spill indicators)
```

**Reduce registers:**
- Break large kernels into smaller ones.
- Use `__attribute__((noinline))` on helper functions to prevent excessive inlining.
- Replace temporary arrays with reduction trees.
- Accumulate in `float` but store in `half` when precision allows.

---

## 5. Divergent Branching

Within a wavefront, divergent branches (where some threads take the if-path and others the else-path) cause both paths to execute serially with masking, doubling worst-case cost.

```cpp
// Bad: half the wavefront idles each branch
if (threadIdx.x % 2 == 0)
    doA();
else
    doB();

// Better: sort/reorder input so adjacent threads follow the same path
// Or use predicated arithmetic instead of branching:
float result = cond ? a : b;    // compiles to v_cndmask
```

- Hoist loop-invariant conditionals above the loop.
- Use `__builtin_expect` to guide branch prediction hints.
- On CDNA, `if/else` on uniform (scalar) conditions (e.g., `blockIdx.x == 0`) use the SALU and are free; only per-thread vector conditions cause divergence.

---

## 6. Atomic Operations

Global atomics stall the wavefront until the operation completes. Prefer LDS-local atomics where possible, then reduce to global at the end.

```cpp
// Pattern: per-block reduction in LDS, one global atomic per block
__shared__ int local_sum;
if (threadIdx.x == 0) local_sum = 0;
__syncthreads();

atomicAdd(&local_sum, thread_val);    // fast LDS atomic
__syncthreads();

if (threadIdx.x == 0)
    atomicAdd(global_sum, local_sum); // one global atomic per block
```

On MI300+, use `__hip_atomic_fetch_add` with `__HIP_MEMORY_SCOPE_WORKGROUP` for workgroup-scoped atomics, which route through LDS without hitting the L2.

---

## 7. Async Copies and Streams

Overlap host-device transfers with kernel execution using multiple streams:

```cpp
hipStream_t stream[2];
hipStreamCreate(&stream[0]);
hipStreamCreate(&stream[1]);

for (int i = 0; i < N; i += CHUNK) {
    int s = i / CHUNK % 2;
    hipMemcpyAsync(d_in + i, h_in + i, CHUNK * sizeof(float),
                   hipMemcpyHostToDevice, stream[s]);
    myKernel<<<grid, block, 0, stream[s]>>>(d_in + i, d_out + i, CHUNK);
    hipMemcpyAsync(h_out + i, d_out + i, CHUNK * sizeof(float),
                   hipMemcpyDeviceToHost, stream[s]);
}
hipDeviceSynchronize();
```

On MI300/MI350, the GPU has dedicated DMA engines that run concurrently with compute. Use pinned host memory (`hipHostMalloc`) for maximum transfer bandwidth.

---

## 8. CDNA-Specific Optimizations (MI300/MI350)

### Matrix cores (MFMA)
AMD CDNA3/4 has Matrix Fused Multiply-Add (MFMA) units. Use them via:
- **rocWMMA** — C++ wrappers for MFMA intrinsics
- **composable_kernel** — pre-built high-performance GEMM/convolution kernels
- Direct intrinsics: `__builtin_amdgcn_mfma_f32_16x16x4f32`

```cpp
// 16x16x4 MFMA: computes C += A * B for 16×16 tiles with depth 4
__builtin_amdgcn_mfma_f32_16x16x4f32(a_frag, b_frag, c_frag, 0, 0, 0);
```

### Unified Memory (MI300 HBM)
MI300 has a unified CPU-GPU memory pool. Use `hipMallocManaged` to exploit this:
```cpp
float *data;
hipMallocManaged(&data, N * sizeof(float));
// No explicit copies needed; prefetch to improve locality:
hipMemAdvise(data, N * sizeof(float), hipMemAdviseSetPreferredLocation, 0 /* GPU */);
hipMemPrefetchAsync(data, N * sizeof(float), 0 /* device */, stream);
```

### Infinity Fabric (xGMI) for multi-GPU
Use RCCL for collective communications; it auto-selects xGMI (NVLink equivalent) paths over PCIe when available.

---

## 9. Profiling with rocprof and Omniperf

```bash
# Basic counter collection
rocprof --stats --hip-trace my_app

# Omniperf (MI200/MI300): rich roofline and bottleneck analysis
omniperf profile --name run1 -- ./my_app
omniperf analyze --path workloads/run1/mi300a/ --list-stats
```

Key metrics to watch:
| Metric | Healthy range |
|--------|--------------|
| Wavefront occupancy | > 50% of max |
| L2 cache hit rate | > 80% for reuse-heavy kernels |
| Memory bandwidth utilization | > 70% of peak for bandwidth-bound kernels |
| VGPR usage | < 64 per thread (for 8 waves/CU) |
| LDS bank conflicts | 0 |

---

## 10. Compilation Flags

```bash
hipcc -O3 \
      --offload-arch=gfx942 \        # MI300X; use gfx950 for MI350
      -mllvm -amdgpu-function-calls=0 \  # inline device functions
      -mllvm -amdgpu-sroa=1 \        # scalar replacement of aggregates
      -ffast-math \                   # allow reassociation, approx math
      kernel.cpp -o kernel
```

- `-O3` enables loop unrolling and vectorization.
- `--offload-arch` must match the target GPU; wrong arch causes runtime failure.
- Avoid `-g` in production; it disables many optimizations and inflates binary size.

---

## 11. Quick Checklist

- [ ] Access pattern is coalesced (SoA layout, 128-byte alignment)
- [ ] Block size is a multiple of 64 (wavefront width)
- [ ] Shared memory tile avoids bank conflicts (pad by 1)
- [ ] Register count < 64 VGPRs/thread (verify with `--save-temps`)
- [ ] No divergent branches in inner loops
- [ ] Atomics use LDS-local reduction before global write
- [ ] Streams overlap compute and data transfer
- [ ] MFMA units used for matrix workloads (via rocWMMA or CK)
- [ ] Kernels profiled with Omniperf to identify actual bottleneck
