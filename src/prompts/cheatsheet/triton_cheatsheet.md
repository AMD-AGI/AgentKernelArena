# Triton Kernel Best Practices

Reference: [Triton documentation](https://triton-lang.org/main/) | [Python API](https://triton-lang.org/main/python-api/triton.language.html)

---

## Candidate entry point (schema v3)

Your whole solution is the directory `solution/`; it is the only editable tree. The task fixes the
call, not the implementation. Entry: `solution/kernel.py::run`. Put `@triton.jit` kernels there (helper `.py` files in `solution/` form one flat package and import relatively: `from .helpers import ...`) and launch them from `run`.

```python
def run(*, <inputs>, <outputs>) -> None: ...
```

`_aka_eval/CANDIDATE.md` in the workspace states the exact signature, shapes and dtypes.
A Triton target imports `triton` and `triton.language` only (not `triton.experimental`, `triton.tools`, `triton.testing` or `triton_kernels`) and may not launch Gluon kernels. The baseline's language does not matter.

Candidate rules (enforced by the evaluator, not a sandbox):

- `run` writes every output (and every mutable input) in place and returns `None`. Outputs are
  preallocated and poisoned before each call; other inputs are read-only.
- All computation runs in kernels you write in Triton. Host code may only allocate, take views and
  make same-dtype copies, and launch. PyTorch arithmetic, reductions, dtype conversions and library
  operators raise at run time; so do other kernel languages and production packages.
- No `eval`/`exec`, reflection, module mutation, I/O, printing, synchronization, host round trips
  (`.item()`, `.cpu()`), randomness, global PyTorch state, or imports of task, reference, baseline or
  evaluator modules. Keep no state derived from inputs across calls. Modules reached as attributes
  (`torch.ctypes`, `torch.os`, `torch.sys`) are rejected; `getattr` takes a literal attribute name.
- Attribute names `synchronize`, `item`, `cpu`, `tolist`, `numpy`, `Stream`, `Event`, `graph` and
  `CUDAGraph` are rejected anywhere in candidate code (`torch.cuda.current_stream()` is allowed), as
  are the `@` operator, `torch.matmul`/`mm`/`bmm`/`einsum`, `torch.rand*`, `torch.tensor`, `torch.nn`
  and `torch.ops`. Only `torch`, the language's own packages and a few standard modules
  (`math`, `functools`, `itertools`, `typing`, `dataclasses`, ...) import; helpers import relatively.
- Every call is timed as one CUDA-graph replay after a 1 GiB cache flush, so launches must be
  capturable on the current stream. Keep first-use compilation and any tuning deterministic and
  cached per shape.

---


## 1. Block / Tile Size Selection and Autotuning

Block sizes are the fundamental knob for Triton performance. Each program instance handles a tile of the input. Block sizes must be powers of 2 and should match the hardware's preferred access granularity.

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    ...
```

Guidelines:
- Start with `BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 32–64`.
- Use `@triton.autotune` with 4–8 configs covering the power-of-2 search space.
- Set `key` to the dimensions that change between calls so the cache stays valid.
- `num_warps` controls the number of software warps (each = 32 threads on AMD, but mapped to 64-thread wavefronts).
- For AMD/ROCm, prefer `num_warps=4` or `num_warps=8`; `num_stages` (pipeline depth) defaults to 2.

---

## 2. Memory Access Patterns and Vectorization

Triton automatically vectorizes loads/stores when it can prove alignment. Help it:

```python
@triton.jit
def kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # Masked load: safe for non-power-of-2 N
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)

    # Aligned load (no mask needed if N is a multiple of BLOCK)
    x = tl.load(X + offs)          # triggers vectorized load instruction

    tl.store(Y + offs, x, mask=mask)
```

- Pass pointers with `tl.multiple_of(ptr, alignment)` to guarantee the compiler emits wider loads:
  ```python
  X = tl.multiple_of(X, 16)   # 16-element alignment hint
  ```
- Use `eviction_policy="evict_last"` for streaming (non-reused) data to avoid polluting L2:
  ```python
  x = tl.load(X + offs, eviction_policy="evict_last")
  ```
- Prefer `tl.float16` or `tl.bfloat16` for compute to double effective bandwidth.

---

## 3. Matrix Operations with `tl.dot`

`tl.dot` maps to hardware matrix multiply units (WMMA on AMD, Tensor Cores on NVIDIA). It is the highest-throughput operation in Triton.

```python
# Tiled GEMM inner loop
a = tl.load(A + ...)                   # shape [BLOCK_M, BLOCK_K], dtype float16
b = tl.load(B + ...)                   # shape [BLOCK_K, BLOCK_N], dtype float16
acc = tl.dot(a, b, acc, out_dtype=tl.float32)   # accumulate in fp32
```

Rules:
- Both inputs to `tl.dot` must be 2-D tensors with shapes that are multiples of the matrix instruction tile (typically 16×16).
- Always accumulate in `float32` even when inputs are `float16`/`bfloat16` to avoid precision loss.
- On AMD (ROCm), `tl.dot` emits MFMA instructions when shapes match (16×16, 32×32, etc.).
- Chain multiple `tl.dot` calls in a loop — Triton will pipeline them automatically.

---

## 4. Reductions

Use `tl.sum`, `tl.max`, `tl.min` for reductions along a block axis. These generate efficient tree-reduction code.

```python
@triton.jit
def softmax_kernel(X, Y, stride, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=-float('inf'))

    # Numerically stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Y + row * stride + offs, y, mask=mask)
```

- `axis=0` reduces along the first (and for 1-D blocks, only) dimension.
- For 2-D blocks, choose the reduction axis that keeps the other dimension intact for `tl.dot`.
- Use `tl.associative_scan` for prefix sums (exclusive / inclusive scan).

---

## 5. Pointer Arithmetic and Masking

All memory in Triton is addressed via pointer + integer offset. Multi-dimensional access uses pointer arithmetic on 1-D arrays.

```python
@triton.jit
def kernel(A, stride_m, stride_n, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # 2-D pointer grid via broadcasting
    ptrs = A + rm[:, None] * stride_m + rn[None, :] * stride_n  # [BLOCK_M, BLOCK_N]
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    a = tl.load(ptrs, mask=mask, other=0.0)
```

Best practices:
- Compute masks once and reuse for load and store.
- Use `tl.constexpr` for shapes known at compile time — this enables loop unrolling and eliminates runtime branches.
- Prefer `tl.cdiv(N, BLOCK)` over `(N + BLOCK - 1) // BLOCK` for grid size calculation.

---

## 6. Compilation Hints and `tl.constexpr`

`tl.constexpr` arguments are baked into the compiled kernel (like C++ template parameters). Use them for all sizes and flags that don't change between kernel launches with the same config.

```python
@triton.jit
def kernel(X, N, BLOCK: tl.constexpr, DTYPE: tl.constexpr, USE_MASK: tl.constexpr):
    offs = tl.arange(0, BLOCK)          # BLOCK must be constexpr to unroll
    if USE_MASK:
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
    else:
        x = tl.load(X + offs)           # no branch at runtime
```

- Every unique combination of `constexpr` values produces a separate compiled binary.
- Use `tl.static_assert` to catch invalid configs at compile time:
  ```python
  tl.static_assert(BLOCK % 16 == 0, "BLOCK must be divisible by 16 for MFMA")
  ```

---

## 7. AMD/ROCm-Specific Tips

### Backend selection
```python
import triton
# Confirm ROCm backend is active
print(triton.runtime.driver.active.get_current_target())
# → HIPBackend(arch='gfx942')  (MI300X)
```

### Autotuning for AMD
Add `num_stages` to configs (AMD benefits from `num_stages=1` or `2`):
```python
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
              num_warps=8, num_stages=2)
```

### `libdevice` math
AMD's ROCm runtime provides `__ocml_*` math functions. Triton exposes them through the standard `tl.math.*` API (`tl.math.exp`, `tl.math.log`, etc.) which maps to the appropriate backend.

### Persistent kernels
For small problem sizes, launch overhead dominates. Use persistent kernels with a work queue:
```python
@triton.jit
def persistent_kernel(work_queue, ...):
    while True:
        task_id = tl.atomic_add(work_queue, 1)
        if task_id >= total_tasks:
            break
        # process task_id
```

### Flash Attention reference
The [AITER](https://github.com/ROCm/aiter) and [vLLM](https://github.com/vllm-project/vllm) repositories contain production-quality Triton flash attention kernels tuned for AMD MI300, which are excellent references.

---

## 8. Debugging Triton Kernels

### Interpret mode
Run kernels on CPU for easier debugging:
```python
os.environ["TRITON_INTERPRET"] = "1"
kernel[grid](X, N, BLOCK=64)
```
Note: interpret mode is slow and doesn't support all ops.

### `tl.device_print`
Print values from inside the kernel (ROCm 6.x+):
```python
@triton.jit
def kernel(X, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    tl.device_print("x[0] = ", x[0])  # prints from program instance 0
```

### Save intermediate IR
```bash
TRITON_PRINT_AUTOTUNING=1 python my_kernel.py   # show autotuning results
MLIR_ENABLE_DUMP=1 python my_kernel.py          # dump full MLIR/LLVM IR
```

### Profiling with rocprof
```bash
rocprof --stats --hip-trace python my_kernel.py
# Look for the triton_* kernel entries in the stats output
```

---

## 9. Common Patterns

### Softmax (row-wise)
See §4 above. Key: do max-reduction → subtract → exp → sum-reduction → divide in one pass over the row stored in SRAM.

### Layer Norm / RMS Norm
```python
# Two-pass: first compute mean/variance, then normalize
# Or: fuse into one pass with Welford's online algorithm
```

### Flash Attention
Key ideas: tile QKV to fit in SRAM, accumulate attention scores with online softmax, never materialize the full N×N attention matrix.
```python
# Outer loop over K/V blocks, inner: tl.dot(q_tile, k_tile.T) → scores
# Online softmax update: m_new = max(m_old, scores_max); l_new = l_old * exp(m_old - m_new) + sum(exp(scores - m_new))
```

### GEMM
```python
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(A + ..., mask=..., other=0.0)
    b = tl.load(B + ..., mask=..., other=0.0)
    acc = tl.dot(a, b, acc)
tl.store(C + ..., acc.to(tl.float16), mask=...)
```

---

## 10. Quick Checklist

- [ ] Block sizes are powers of 2 and divisible by 16 (for MFMA)
- [ ] `@triton.autotune` covers a range of `BLOCK_M/N/K` and `num_warps`
- [ ] All shapes / flags passed as `tl.constexpr` where possible
- [ ] `tl.dot` accumulates in `float32`
- [ ] Masks computed once and reused for both load and store
- [ ] Non-reused streaming loads use `eviction_policy="evict_last"`
- [ ] Reductions use `tl.sum` / `tl.max` (not manual loops)
- [ ] AMD target verified (`gfx942` for MI300X, `gfx950` for MI350)
- [ ] Kernel profiled with `rocprof --stats` to confirm expected throughput
- [ ] Correctness validated against a reference (PyTorch) implementation with `torch.testing.assert_close`
