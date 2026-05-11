# FlyDSL Kernel Best Practices

Reference: [FlyDSL GitHub](https://github.com/ROCm/FlyDSL) | [Nightly Wheels](https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/)

---

## 1. Kernel Structure and Compilation Model

FlyDSL kernels are Python functions decorated with `@flyc.kernel` that generate GPU code at build time via MLIR. A `@flyc.jit` wrapper provides the host launch entry point.

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def my_kernel(Input: fx.Tensor, Output: fx.Tensor):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    # kernel body using fx.* APIs

@flyc.jit
def launch(Input: fx.Tensor, Output: fx.Tensor, n: fx.Int32,
           stream: fx.Stream = fx.Stream(None)):
    launcher = my_kernel(Input, Output)
    launcher.launch(grid=(n, 1, 1), block=(256, 1, 1), stream=stream)
```

Guidelines:
- The `build_*_module(M, N, dtype_str)` factory pattern captures shape/dtype as compile-time constants via Python closures — use `const_expr()` and `range_constexpr()` to specialize code paths.
- Kernel functions receive `fx.Tensor` arguments; all index/arithmetic uses `fx.*` typed wrappers (`fx.Int32`, `fx.Float32`, `fx.Index`).
- Architecture is detected at build time via `get_rocm_arch()` — use this to gate architecture-specific paths (e.g., gfx950 hardware BF16 conversion).

---

## 2. Vectorized Buffer Access (Fast Path)

FlyDSL exposes ROCm buffer load/store intrinsics for maximum memory throughput.

```python
VEC_WIDTH = 8  # 8 × 16-bit = 128-bit per load

Input_buf = fx.rocdl.make_buffer_tensor(Input)
row = fx.slice(Input_buf, (bid, None))
divided = fx.logical_divide(row, fx.make_layout(VEC_WIDTH, 1))

copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
vec_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(VEC_WIDTH, 1),
                                fx.AddressSpace.Register)
vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)

# Load a vector of VEC_WIDTH elements
reg = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
fx.copy_atom_call(copy_atom, fx.slice(divided, (None, tid)), reg)
vec = fx.memref_load_vec(reg)
```

Guidelines:
- `BufferCopy128b()` → 128-bit (8 × f16 or 4 × f32) per thread per cycle. This is the widest fast path on MI300X.
- Use `logical_divide` to tile the row into VEC_WIDTH chunks, then index by `tid + tile_i * BLOCK_THREADS`.
- Fast path requires `N % (BLOCK_THREADS * VEC_WIDTH) == 0` and `elem_bits <= 16`. Fall back to scalar `BufferCopy16b()`/`BufferCopy32b()` otherwise.
- Increasing VEC_WIDTH (e.g., to 16) may improve bandwidth utilization but increases register pressure — profile to find the sweet spot.

---

## 3. Shared Memory Reductions

FlyDSL uses `SmemAllocator` for shared memory and explicit wave-level shuffle instructions.

```python
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

allocator = SmemAllocator(None, arch=arch)
red_offset = allocator._align(allocator.ptr, 16)
allocator.ptr = red_offset + RED_SLOTS * 4  # f32 slots

# Inside @flyc.kernel:
base_ptr = allocator.get_base()
s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(RED_SLOTS,))

def wave_reduce_add(x):
    w = x
    for _sh in range_constexpr(int(math.log2(WARP_SIZE))):
        off = fx.Int32(WARP_SIZE // (2 << _sh))
        peer = w.shuffle_xor(off, fx.Int32(WARP_SIZE))
        w = w.addf(peer, fastmath=fm_fast)
    return w
```

Guidelines:
- RED_SLOTS = ceil(BLOCK_THREADS / WARP_SIZE). On MI300X, WARP_SIZE = 64.
- Two-level reduction: intra-wave via `shuffle_xor`, inter-wave via shared memory.
- Always call `gpu.barrier()` between shared memory write and read phases.
- Use `arith.FastMathFlags.fast` for reduction accumulation — safe when float32 accumulation is used.
- Fuse multiple reductions (e.g., sum + sum-of-squares) into a single `block_reduce_add2` pass to halve barrier overhead.

---

## 4. Block Size and Thread Count Tuning

```python
BLOCK_THREADS = 256  # threads per block
VEC_WIDTH = 8        # elements per vectorized load
tile_cols = BLOCK_THREADS * VEC_WIDTH  # columns covered per tile
```

Guidelines:
- BLOCK_THREADS = 256 is the default. For small N (< 2048), try 128 to reduce shared memory pressure.
- For large N (> 8192), try 512 threads if register pressure allows.
- `tile_cols = BLOCK_THREADS * VEC_WIDTH` determines the fast-path granularity — ensure N is a multiple of tile_cols for vectorized access.
- Number of tiles = N / tile_cols. More tiles → more loop iterations, but each is fully vectorized.

---

## 5. Data Type Handling and Precision

```python
from flydsl.expr.numeric import Numeric, Float32, Uint32

elem_type = dtype_to_elem_type(dtype_str)  # "f16" → f16 IR type
compute_type = T.f32                        # always accumulate in f32

# Convert for computation
x_f32 = vec.to(Float32)

# Convert back for output
out = y.to(Numeric.from_ir_type(elem_type))
```

Guidelines:
- Always accumulate reductions in float32 — this is critical for numerical stability.
- For BF16 output on gfx950, use hardware conversion: `y.to(elem_dtype)`. On gfx942, software round-nearest-even is needed (bitwise pack via `Uint32`).
- Gate architecture-specific conversions with `const_expr()` to eliminate dead code at compile time.

---

## 6. Compile-Time Specialization

```python
from flydsl.expr import const_expr, range_constexpr

# Compile-time branching (dead code eliminated)
if const_expr(N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16):
    # vectorized fast path
else:
    # scalar fallback

# Compile-time loop unrolling
for tile_i in range_constexpr(num_tiles):
    ...
```

Guidelines:
- `const_expr()` evaluates at kernel build time — use for path selection based on shapes, dtypes, and architecture.
- `range_constexpr()` fully unrolls at compile time — use for tile loops, reduction tree stages, and any fixed-count iteration.
- Keep `const_expr` conditions simple (comparisons and arithmetic on Python ints/bools captured from the closure).

---

## 7. Common Optimization Patterns

1. **Two-pass fusion**: For normalization kernels, cache input in registers during the first pass (reduction), then reuse for the second pass (normalize + scale). Avoids a second global memory read.

2. **Register caching**: Store loaded vectors in a Python list (`in_local.append(vec)`) — these become register-resident across passes.

3. **Scalar fallback with masking**: For non-aligned dimensions, use `is_valid = idx < N` with `select` to mask out-of-bounds threads rather than branching.

4. **Launch configuration**: Grid = (M, 1, 1) for row-parallel kernels (one block per row). Block = (BLOCK_THREADS, 1, 1).

5. **Stream parameter**: Always accept `stream: fx.Stream = fx.Stream(None)` in the JIT wrapper for async execution compatibility.
