# HIP Kernel Best Practices for RDNA GPUs

Reference: [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/) | [RDNA ISA](https://gpuopen.com/amd-isa-documentation/)

---

## 1. Memory Access — Coalescing

RDNA GPUs access global memory in 64-byte cache lines. A wavefront of 32 threads fetches optimally when consecutive threads access consecutive addresses.

**Good — coalesced:**
```cpp
float val = a[blockDim.x * blockIdx.x + threadIdx.x];
```

**Bad — strided:**
```cpp
float val = a[threadIdx.x * N];
```

Rules:
- Prefer Structure-of-Arrays (SoA) over Array-of-Structures (AoS).
- Align buffers to 128 bytes (`hipMallocAligned` or `__attribute__((aligned(128)))`).
- Use vector loads (`float4`, `half2`, `uint4`) to widen memory transactions. This is critical on RDNA due to lower bandwidth (~640 GB/s GDDR6).

---

## 2. Occupancy and Wavefront Management

RDNA defaults to **Wave32** (32 threads per wavefront). Each CU can schedule up to 16 Wave32 wavefronts. High occupancy hides memory latency.

### Controlling occupancy
```cpp
__attribute__((amdgpu_waves_per_eu(4, 8)))
__global__ void myKernel(...) { ... }

__attribute__((amdgpu_flat_work_group_size(64, 256)))
__global__ void myKernel(...) { ... }
```

### Key occupancy limits (RDNA 4, gfx1201)
| Resource per CU | Limit |
|-----------------|-------|
| Wavefronts      | 16 (Wave32) |
| VGPRs           | 1536 total |
| SGPRs           | 512 total |
| LDS             | 64 KB (128 KB per WGP) |

- Block size should be a multiple of 32 (wavefront width). 64 or 128 are good starting points.
- Prefer 128–256 threads/block; tune with `hipOccupancyMaxPotentialBlockSize`.
- Target VGPR usage < 96 per thread for good occupancy.

---

## 3. LDS (Local Data Share / Shared Memory)

LDS provides ~100x faster bandwidth than global memory. Each WGP has 128 KB (64 KB per CU).

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

Each CU has 1536 VGPRs total. High register usage per thread reduces maximum wavefronts. Register spilling to scratch memory adds ~500 cycle latency.

**Check register usage:**
```bash
hipcc -O3 --save-temps --offload-arch=gfx1201 kernel.cpp
# Read the .s assembly for v_readlane / s_load_dword (spill indicators)
```

**Reduce registers:**
- Break large kernels into smaller ones.
- Use `__attribute__((noinline))` on helper functions to prevent excessive inlining.
- Replace temporary arrays with reduction trees.
- Accumulate in `float` but store in `half` when precision allows.

---

## 5. Divergent Branching

Within a Wave32, divergent branches cause both paths to execute serially with masking.

```cpp
// Bad: half the wavefront idles each branch
if (threadIdx.x % 2 == 0)
    doA();
else
    doB();

// Better: use predicated arithmetic
float result = cond ? a : b;    // compiles to v_cndmask
```

- Hoist loop-invariant conditionals above the loop.
- RDNA has a strong scalar ALU — uniform conditions (e.g., `blockIdx.x == 0`) run on the SALU for free. Only per-thread vector conditions cause divergence.

---

## 6. Atomic Operations

Global atomics stall the wavefront. Prefer LDS-local atomics, then reduce to global.

```cpp
__shared__ int local_sum;
if (threadIdx.x == 0) local_sum = 0;
__syncthreads();

atomicAdd(&local_sum, thread_val);    // fast LDS atomic
__syncthreads();

if (threadIdx.x == 0)
    atomicAdd(global_sum, local_sum); // one global atomic per block
```

Use `__hip_atomic_fetch_add` with `__HIP_MEMORY_SCOPE_WORKGROUP` for workgroup-scoped atomics.

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

Use pinned host memory (`hipHostMalloc`) for maximum PCIe transfer bandwidth.

---

## 8. RDNA-Specific Optimizations

### Wave32 advantages
- Shuffle/permute operations span 32 lanes (fewer steps for reductions)
- `__ballot()` returns a 32-bit mask
- More wavefronts fit per CU, improving latency hiding
- Cross-lane operations are faster (smaller wavefront)

### No MFMA — use WMMA or vector ALU
- **Do NOT** use `__builtin_amdgcn_mfma_*` intrinsics — they do not exist on RDNA.
- RDNA 4 supports **WMMA** (Wave Matrix Multiply-Accumulate) via `__builtin_amdgcn_wmma_*` or rocWMMA.
- For non-matrix workloads, use standard vector ALU operations. RDNA 4 has strong FP32/FP16 throughput.

### Memory bandwidth is the bottleneck
- RDNA 4 has ~640 GB/s GDDR6 (vs 5.3 TB/s HBM3 on MI300X).
- Kernels that were compute-bound on CDNA may become memory-bound on RDNA.
- Minimize global memory traffic: fuse operations, use LDS aggressively, use vector loads.

### No unified CPU-GPU memory
- Use explicit `hipMemcpy` or `hipMallocManaged` with prefetch.
- Host-device transfer goes over PCIe, not Infinity Fabric.

### Smaller Infinity Cache
- 32 MB L3 (vs 256 MB on MI300X). Do not rely on it for large working sets.
- Size tiles to fit in L2 (4 MB) or LDS (128 KB per WGP).

---

## 9. Profiling

```bash
# Basic counter collection
rocprof --stats --hip-trace my_app

# rocprofv3 counter collection
rocprofv3 --hip-trace --kernel-trace -- ./my_app
```

Key metrics to watch:
| Metric | Healthy range |
|--------|--------------|
| Wavefront occupancy | > 50% of max |
| L2 cache hit rate | > 80% for reuse-heavy kernels |
| Memory bandwidth utilization | > 70% of peak for bandwidth-bound kernels |
| VGPR usage | < 96 per thread (for good occupancy) |
| LDS bank conflicts | 0 |

---

## 10. Compilation Flags

```bash
hipcc -O3 \
      --offload-arch=gfx1201 \
      -mllvm -amdgpu-function-calls=0 \  # inline device functions
      -ffast-math \
      kernel.cpp -o kernel
```

- `--offload-arch=gfx1201` is required for RDNA 4. Do NOT use `gfx942` (MI300X) or `gfx950` (MI355X).
- `-O3` enables loop unrolling and vectorization.
- Avoid `-g` in production; it disables many optimizations.

---

## 11. Quick Checklist

- [ ] Access pattern is coalesced (SoA layout, 128-byte alignment)
- [ ] Block size is a multiple of 32 (Wave32 width)
- [ ] Shared memory tile avoids bank conflicts (pad by 1)
- [ ] Register count < 96 VGPRs/thread (verify with `--save-temps`)
- [ ] No divergent branches in inner loops
- [ ] Atomics use LDS-local reduction before global write
- [ ] Streams overlap compute and data transfer
- [ ] WMMA used for matrix workloads (not MFMA)
- [ ] Global memory traffic minimized (fuse ops, vector loads)
- [ ] Kernels profiled with rocprof/rocprofv3 to identify bottleneck
