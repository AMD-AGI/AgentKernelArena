# TileLang Kernel Best Practices

Reference: [TileLang GitHub](https://github.com/tile-ai/tilelang) | [Language docs](https://tilelang.com/)

TileLang kernels are Python functions that describe a tile program; the compiler lowers
them through TVM's TIR to HIP for gfx942/gfx950. You write the tiling, the memory
hierarchy and the thread mapping explicitly, and the compiler handles instruction
selection, layout inference and synchronization.

---

## 1. Kernel Structure and Compilation Model

A kernel is a plain Python function under `@tilelang.jit`. Tensor arguments are declared
with type annotations, and the body opens a launch scope with `T.Kernel`.

```python
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

@tilelang.jit(pass_configs=pass_configs)
def my_kernel(x, out, hidden_size: int, n_splits: int = 1):
    num_tokens = T.dynamic("num_tokens")

    x:   T.Tensor[[num_tokens, hidden_size], T.bfloat16]
    out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]

    with T.Kernel(num_tokens, threads=256) as i:
        ...
```

Guidelines:
- The decorated function runs **once per shape signature** to build the program; the
  returned object is cached and called with real tensors. Keep Python-level work in the
  body cheap and shape-driven.
- Non-tensor parameters (`hidden_size`, `n_splits`, tile sizes) are captured as
  **compile-time constants**. Changing one produces a new compiled kernel, so they are
  the natural knobs for autotuning — but each distinct value costs a compile.
- Use `T.dynamic("name")` only for dimensions that genuinely vary at run time (usually
  the token/batch dimension). Everything else should be static so the compiler can
  unroll and vectorize.
- `T.Kernel(*grid, threads=N)` yields the block indices. `T.Kernel(m, n, threads=...)`
  yields a tuple `(i_m, i_n)`. `threads` is the block size, not a hint.
- On ROCm, `TL_DISABLE_WARP_SPECIALIZED` and `TL_DISABLE_TMA_LOWER` are the safe
  defaults — warp specialization and TMA lowering are NVIDIA-oriented paths.

---

## 2. Memory Scopes: fragment, shared, local

TileLang makes the memory hierarchy explicit. Choosing the right scope is usually worth
more than any arithmetic change.

```python
acc    = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)  # per-thread registers
smem   = T.alloc_shared((BLOCK_M, BLOCK_K), T.bfloat16)   # LDS, block-wide
scalar = T.alloc_local((1,), T.float32)                   # single-thread register

T.clear(acc)                    # zero a buffer
T.copy(src, dst)                # layout-aware bulk copy, any scope pair
```

Guidelines:
- `alloc_fragment` is a **distributed** buffer: the compiler infers a per-thread layout
  from how you index it. This is what you want for accumulators and for data you reduce
  over.
- `alloc_shared` is LDS. Use it to stage global data once for reuse across threads, and
  to communicate between the warps of a block. MI355X has 160 KB LDS per CU; exceeding
  the per-block budget silently cuts occupancy.
- `T.copy` picks the widest legal vector width and emits the right global/LDS
  instructions. Prefer it over hand-written element loops — an explicit loop usually
  compiles to scalar accesses.
- Copy global → shared → fragment rather than global → fragment when more than one
  thread reads the same element.

---

## 3. Loop Constructs and What They Mean

The loop type controls the thread mapping; it is not a stylistic choice.

```python
for j in T.Parallel(n):              # distributed across threads, auto-vectorized
    out[j] = x[j] * scale

for k in T.serial(n_splits):         # sequential inside each thread
    acc[0] += partial[k, i]

for ko in T.Pipelined(K // BLOCK_K, num_stages=2):   # software-pipelined stages
    T.copy(A[bx, ko * BLOCK_K], A_s)
    T.gemm(A_s, B_s, acc)
```

Guidelines:
- `T.Parallel` binds the iteration space to threads and lets the compiler vectorize;
  this is the default for anything element-wise.
- `T.Pipelined(..., num_stages=N)` overlaps the global load of iteration `k+1` with the
  compute of iteration `k`. `num_stages=2` is the usual starting point; 3 helps when
  loads are long and LDS allows it, at the cost of more shared memory.
- `T.serial` is a genuine sequential loop — use it for split-k accumulation and small
  fixed reductions where parallelizing would need a cross-thread reduction anyway.
- Nested `T.Parallel(a, b)` maps a 2-D space at once and is preferable to two nested
  parallel loops.

---

## 4. Matrix Multiply and Reductions

```python
T.gemm(A_shared, B_shared, C_fragment)                 # MFMA on gfx942/gfx950
T.gemm(A_s, B_s, C_f, transpose_B=True)

T.reduce_sum(src, dst, dim=1)     # reduce along an axis into a smaller buffer
T.reduce_max(src, dst, dim=1)
```

Guidelines:
- `T.gemm` maps to MFMA. Operands normally live in shared memory and the accumulator in
  a fragment; accumulate in `float32` even for bf16/fp16 inputs.
- Tile sizes should keep the MFMA shape happy: `BLOCK_M`/`BLOCK_N` multiples of 32 (64
  and 128 are the common sweet spots) and `BLOCK_K` a multiple of 32 for 16-bit inputs.
- `T.reduce_sum(..., dim=d)` reduces a fragment along one axis. For a full block
  reduction, reduce into a fragment, stage through `alloc_shared`, then have one warp
  finish the job.
- For tiny reductions (a handful of elements per token) a `T.serial` accumulate in one
  warp beats a general reduction — the barrier traffic dominates.

---

## 5. Warp-Level Work and Divergence

```python
if T.get_thread_binding() < 32:
    # only the first warp does this phase
    ...
```

Guidelines:
- `T.get_thread_binding()` gives the linear thread index; comparing it against a warp
  boundary is the idiom for "let one warp do the serial tail".
- The wavefront is **64 lanes** on CDNA (not 32). A `< 32` guard leaves half of the
  first wavefront idle — deliberate when the tail is tiny, wasteful otherwise.
- Data written by one warp and read by another must pass through `alloc_shared`; a
  fragment is private to its thread's registers.
- Prefer few, wide phases over many narrow guarded regions: each divergent region costs
  a barrier and blocks the compiler from overlapping work.

---

## 6. Choosing `threads` and the Grid

```python
with T.Kernel(num_tokens, threads=96) as i:   # one block per token
with T.Kernel(m_tiles, n_tiles, threads=256) as (i_m, i_n):
```

Guidelines:
- `threads` must be a multiple of 64 to fill wavefronts. Values like 96 are legal and
  sometimes deliberate for a small fused kernel, but they leave a partial wavefront —
  only do it when register pressure is the binding constraint.
- 256 threads (4 wavefronts) is the default for tiled compute kernels; 512 helps
  latency-bound kernels with enough independent work and low register use.
- One block per token is right when each token's work is small and independent. Once
  tokens exceed a few thousand, tile the token dimension instead so blocks stay
  resident and launch overhead amortizes.
- Grid dimensions become the block indices in declaration order — keep the fastest
  varying dimension last for locality.

---

## 7. Split-K and Multi-Pass Fusion

```python
# pass 1: partial results into [n_splits, tokens, ...]
with T.Kernel(num_tokens, n_tiles, n_splits, threads=n_thr) as (i_n, i_t, i_s):
    ...
    out[i_s, i_n, out_idx] = partial

# pass 2: combine the splits
for i_split in T.serial(n_splits):
    acc[0] += partial[i_split, i]
```

Guidelines:
- Split-k trades extra memory traffic and a second pass for parallelism. It pays off
  only when the grid would otherwise leave CUs idle — compute the split factor from
  the SM/CU count and the grid size rather than hardcoding it.
- Avoid split-k when K is small; the combine pass then dominates.
- When several elementwise stages follow a GEMM, fuse them into the consumer kernel so
  the intermediate never reaches HBM. The big win in fused blocks is usually eliminating
  a round trip, not making any single stage faster.

---

## 8. Numerics

Guidelines:
- Accumulate in `T.float32` even when inputs and outputs are `T.bfloat16`; convert only
  at the store.
- `T.rsqrt`, `T.sigmoid`, `T.exp` are available directly; use them instead of composing
  from `T.sqrt`/division, which costs extra instructions and loses the fast path.
- Normalization epsilons and iteration counts are part of the numerical contract of the
  task — carry them through unchanged. Changing an epsilon to gain speed is a
  correctness regression, not an optimization.

---

## 9. Tuning Checklist

1. **Tile sizes** (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `tile_n`) — the highest-leverage
   knob. Sweep powers of two around the current value first.
2. **`threads`** — try 128 / 256 / 512; watch for register spills at the high end.
3. **`num_stages`** in `T.Pipelined` — 2 → 3 when loads dominate and LDS is free.
4. **Split factor** — only if the grid underfills the device.
5. **Scope changes** — move a reused global read into `alloc_shared`; move a
   short-lived shared buffer into a fragment.
6. **Fusion** — merge adjacent kernels that pass a tensor through HBM.

Measure after each change: TileLang's compile-time decisions (layout inference,
vector width, pipelining) can interact, so a change that should help sometimes does not.
