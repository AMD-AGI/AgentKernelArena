# Triton Kernel Best Practices for RDNA GPUs

Reference: [Triton documentation](https://triton-lang.org/main/) | [Python API](https://triton-lang.org/main/python-api/triton.language.html) | [ROCm Triton](https://github.com/ROCm/triton)

---

## 1. Block / Tile Size Selection and Autotuning

On RDNA, Triton's `num_warps` multiplies by **32 threads per warp** (Wave32), not 64 as on CDNA. A kernel with `num_warps=4` dispatches 128 threads per program instance; the same value on MI300 dispatches 256 threads. Expect different occupancy trade-offs.

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=1),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    ...
```

Guidelines:
- Start with `BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 32`. `BLOCK_K = 64` may spill on RDNA due to the smaller VGPR budget per CU.
- Prefer `num_warps ∈ {2, 4, 8}`. Wave32 means even small `num_warps` values dispatch full wavefronts cleanly.
- `num_stages = 1` or `2` on RDNA. Deeper pipelines rarely help because RDNA's L1/L0 caches are smaller than CDNA's L2.
- Keep BLOCK sizes powers of 2 and divisible by 16 (required by WMMA instruction tiling).

---

## 2. Memory Access Patterns and Vectorization

RDNA 4 has ~640 GB/s GDDR6 bandwidth vs MI300X's 5.3 TB/s HBM3 — memory bandwidth is the primary bottleneck. Squeeze every byte.

```python
@triton.jit
def kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # Masked load: safe for non-power-of-2 N
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)

    tl.store(Y + offs, x, mask=mask)
```

- Use `tl.multiple_of(ptr, 16)` and `tl.max_contiguous(ptr, 16)` so Triton emits 128-bit (`dwordx4`) loads.
- Prefer `float16` / `bfloat16` for compute to double effective bandwidth; accumulate in `float32` via `tl.dot(..., out_dtype=tl.float32)`.
- Use `eviction_policy="evict_last"` for streaming data (read-once) so you don't pollute the small L2 (4 MB on gfx1201).
- Fuse elementwise ops into the same kernel whenever possible — every round-trip to GDDR6 is expensive.

---

## 3. `tl.dot` on RDNA: WMMA, not MFMA

On RDNA 4, `tl.dot` lowers to **WMMA** (Wave Matrix Multiply-Accumulate) instructions, not MFMA. The instruction tiling and dtype support differ from CDNA.

```python
# Tiled GEMM inner loop (works on both CDNA and RDNA)
a = tl.load(A + ...)                              # [BLOCK_M, BLOCK_K], float16 or bf16
b = tl.load(B + ...)                              # [BLOCK_K, BLOCK_N], float16 or bf16
acc = tl.dot(a, b, acc, out_dtype=tl.float32)     # accumulate in fp32
```

Rules and differences vs MI300:
- **Supported dtypes**: `tl.dot` on gfx1201 supports `fp16 × fp16 → fp32`, `bf16 × bf16 → fp32`, and `int8 × int8 → int32`. FP8 `tl.dot` is **not** supported on gfx1201 (MI300/MI350 only).
- **Tile shapes**: WMMA uses 16×16×16 tiles. Both inputs must have shapes divisible by 16 in all dimensions.
- **Throughput**: WMMA on a single RDNA WGP is lower than MFMA on a CDNA CU. Do not expect MI300-class matmul TFLOPS.
- **Fallback**: If a dtype combination is unsupported, Triton emits scalar FMA code silently — expect orders-of-magnitude slowdown. Verify with `MLIR_ENABLE_DUMP=1`.

Always guard with `tl.static_assert`:
```python
tl.static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be divisible by 16 for WMMA")
tl.static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be divisible by 16 for WMMA")
tl.static_assert(BLOCK_K % 16 == 0, "BLOCK_K must be divisible by 16 for WMMA")
```

---

## 4. Reductions

Wave32 reductions finish in 5 cross-lane steps (log2(32)) vs 6 on Wave64. `tl.sum` / `tl.max` / `tl.min` / `tl.argmax` compile to the optimal tree.

```python
@triton.jit
def softmax_kernel(X, Y, stride, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=-float('inf'))

    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Y + row * stride + offs, y, mask=mask)
```

- For multi-pass reductions (softmax, layer-norm), keep the entire row in registers — the LDS round-trip is wasted on RDNA.
- `tl.associative_scan` is supported but the RDNA backend may lower some primitives less efficiently than CDNA; measure before committing to scan-heavy designs.

---

## 5. `num_warps` and Occupancy

Each RDNA 4 CU supports up to 16 Wave32 wavefronts (1536 VGPRs total, 512 SGPRs). Lower `num_warps` per program → more concurrent programs per CU, better latency hiding.

Tuning heuristics:
| Problem shape | Suggested `num_warps` |
|---|---|
| Elementwise / reduction, BLOCK ≤ 1024 | 2 or 4 |
| Matmul, BLOCK_M×BLOCK_N ≤ 64×64 | 2 or 4 |
| Matmul, BLOCK_M×BLOCK_N ≤ 128×128 | 4 |
| Matmul, BLOCK_M×BLOCK_N ≥ 128×256 | 8 |
| Attention (flash-style) | 4 (kv tiling in inner loop) |

- Start at `num_warps=4`. Increase only if occupancy analysis shows you are latency-bound.
- Check VGPR usage in compiled kernel (`MLIR_ENABLE_DUMP=1` then read `.amdgcn` output). Target < 96 VGPRs per thread for good occupancy.
- RDNA has twice as many programs per CU as Wave64 CDNA at the same `num_warps` — keep BLOCK sizes modest to avoid over-subscribing the register file.

---

## 6. LDS (Shared Memory)

Triton manages LDS automatically for `tl.dot` tiles and `tl.load` with explicit reuse. RDNA 4 has **128 KB LDS per WGP** (shared between 2 CUs). Effective budget per workgroup is the same 64 KB as CDNA's per-CU LDS.

- Triton's autotuner respects the LDS budget; oversized configs are rejected with `OutOfResources`.
- For manual shared-memory patterns (e.g., persistent kernels), write explicit tile loads and keep each workgroup's LDS usage ≤ 32 KB to allow two programs per WGP.
- Avoid bank conflicts: LDS on RDNA has 32 banks (4 bytes each). Triton emits layout transforms to avoid them, but user-placed intermediate tiles (e.g., via `tl.zeros`) may still conflict for awkward shapes.

---

## 7. Register Pressure and Spilling

RDNA 4 has 1536 VGPRs per CU total; >96 VGPRs per thread cuts occupancy in half.

```bash
MLIR_ENABLE_DUMP=1 python my_kernel.py 2>&1 | grep -A2 "; NumVgprs"
```

Reduce pressure:
- Lower `BLOCK_K` (shrinks the accumulator intermediate).
- Split large fused kernels into two; use one global memory write between them if it avoids spills.
- Reuse `acc` accumulator across `tl.dot` calls — don't allocate fresh `tl.zeros` per K-iteration.
- For elementwise kernels, prefer broadcasting scalars (`tl.full((), value)`) over `tl.full([BLOCK], value)` — the latter materializes a full tile in registers.

---

## 8. AMD/ROCm Backend for RDNA

### Verify the target
```python
import triton
print(triton.runtime.driver.active.get_current_target())
# → HIPBackend(arch='gfx1201', warp_size=32)   # RDNA 4
```

If `warp_size` is 64 or `arch` is `gfx942`/`gfx950`, you are not running on RDNA. Check `HIP_VISIBLE_DEVICES` and `PYTORCH_ROCM_ARCH`.

### Triton version
- Minimum: **Triton 3.2** (first release with usable gfx1201 support).
- Recommended: **Triton 3.4+** or ROCm-triton main, which includes WMMA code-gen and `tl.dot` dtype fixes for gfx1201.

### Autotuner config space
```python
# RDNA-friendly starting configs
configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=2, num_stages=1),
]
```

### `libdevice` math
RDNA uses the same `__ocml_*` math library as CDNA. Call them via `tl.math.exp`, `tl.math.log`, etc. — unchanged from MI300 code.

### Persistent kernels
Launch overhead is higher relative to kernel time on RDNA (smaller GPU). Persistent kernels with a work-queue pattern pay off sooner than on MI300 for small problem sizes.

---

## 9. Profiling

```bash
# Basic counter collection
rocprofv3 --hip-trace --kernel-trace -- python my_kernel.py

# Show Triton autotuning results
TRITON_PRINT_AUTOTUNING=1 python my_kernel.py
```

Key metrics for RDNA Triton kernels:
| Metric | Healthy range (RDNA 4) |
|---|---|
| Wave32 occupancy | > 50% of peak (at least 8 waves/CU) |
| Memory bandwidth utilization | > 70% of 640 GB/s for bandwidth-bound kernels |
| L2 cache hit rate | > 70% (smaller 4 MB L2) |
| VGPR usage | < 96 per thread |
| LDS bank conflicts | 0 |
| WMMA instruction throughput | verify MFMA is NOT emitted |

Inspect the generated assembly:
```bash
MLIR_ENABLE_DUMP=1 AMDGCN_ENABLE_DUMP=1 python my_kernel.py 2>&1 | \
    grep -E "v_wmma|v_mfma"
# Should see v_wmma_* on gfx1201; v_mfma_* indicates wrong target or fallback
```

---

## 10. Common RDNA-vs-MI300 Gotchas

- **FP8 `tl.dot`** doesn't compile on gfx1201 — silently falls back to scalar FMA. Use fp16/bf16 on RDNA and gate FP8 paths with `tl.constexpr` flags keyed off target arch.
- **`num_warps=1` workloads**: Wave32 means a single warp is 32 threads. Existing MI300 code that assumes `num_warps=1` gives 64 threads will under-dispatch by 2x. Re-tune small block sizes.
- **Softmax / layer-norm inner reductions**: Wave32 cross-lane is faster, but there are fewer threads per warp, so multi-row SRAM layouts that relied on Wave64 broadcast need `tl.broadcast_to` adjustments.
- **Infinity Cache (L3)**: 32 MB on gfx1201 vs 256 MB on MI300X. Large working sets that fit in MI300's L3 will spill to GDDR6 on RDNA. Shrink tile sizes or re-stream.
- **Multi-GPU**: RDNA has no XGMI/Infinity Fabric — multi-GPU collectives go over PCIe. NCCL Triton overlap patterns tuned for MI300 won't map directly.

---

## 11. Quick Checklist

- [ ] Target verified: `arch='gfx1201'`, `warp_size=32`
- [ ] Triton version ≥ 3.4 (or ROCm-triton main)
- [ ] BLOCK_{M,N,K} all divisible by 16 (WMMA requirement, enforced with `tl.static_assert`)
- [ ] `num_warps` tuned for Wave32 (start at 4; do not blindly reuse MI300 values)
- [ ] `num_stages` ∈ {1, 2}
- [ ] `tl.dot` accumulates in `float32`, dtypes limited to fp16/bf16/int8 on gfx1201 (no FP8)
- [ ] VGPR usage < 96/thread (verify with `MLIR_ENABLE_DUMP=1`)
- [ ] LDS usage ≤ 32 KB per workgroup for 2-programs-per-WGP occupancy
- [ ] Memory access hinted with `tl.multiple_of` / `tl.max_contiguous` for 128-bit loads
- [ ] Streaming data uses `eviction_policy="evict_last"`
- [ ] Global memory traffic minimized (small L2 + lower GDDR6 bandwidth)
- [ ] Profiling confirms `v_wmma_*` instructions emitted (not `v_mfma_*` or scalar FMA)
- [ ] Correctness validated against PyTorch reference with `torch.testing.assert_close`
