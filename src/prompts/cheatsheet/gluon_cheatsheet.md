# Gluon Kernel Best Practices (AMD CDNA4 / gfx950)

Reference: [Triton repository](https://github.com/triton-lang/triton) (`python/triton/experimental/gluon`) | [Gluon tutorials](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon)

Gluon is the lower-level sibling of Triton in the same compiler. Triton picks register
layouts, shared-memory use and instruction selection for you; Gluon makes you state them. In
exchange you control exactly how data is distributed over lanes and which instructions run.
Reach for it when the Triton compiler's layout choices leave performance on the table; a plain
elementwise or reduction kernel is rarely faster in Gluon than in Triton.

---

## Candidate entry point (schema v3)

Your whole solution is the directory `solution/`; it is the only editable tree. The task fixes the
call, not the implementation. Entry: `solution/kernel.py::run`. Put `@gluon.jit` kernels there
(helper `.py` files in `solution/` form one flat package and import relatively:
`from .helpers import ...`) and launch them from `run`.

```python
def run(*, <inputs>, <outputs>) -> None: ...
```

`_aka_eval/CANDIDATE.md` in the workspace states the exact signature, shapes and dtypes.
A Gluon target may import `triton`, `triton.language` and `triton.experimental.gluon` (with its
`language` submodule and `language.amd`), but not `triton.tools`, `triton.testing` or `triton_kernels`,
and it must launch at least one Gluon kernel per call; Triton kernels do not count. The baseline's
language does not matter.

Candidate rules (enforced by the evaluator, not a sandbox):

- `run` writes every output (and every mutable input) in place and returns `None`. Outputs are
  preallocated and poisoned before each call; other inputs are read-only.
- All computation runs in kernels you write in Gluon. Host code may only allocate, take views and
  make same-dtype copies, and launch. PyTorch arithmetic, reductions, dtype conversions and library
  operators raise at run time.
- No `eval`/`exec`, reflection, module mutation, I/O, printing, synchronization, host round trips
  (`.item()`, `.cpu()`), randomness, global PyTorch state, or imports of task, reference, baseline or
  evaluator modules. Keep no state derived from inputs across calls.
- Attribute names `synchronize`, `item`, `cpu`, `tolist`, `numpy`, `Stream`, `Event`, `graph` and
  `CUDAGraph` are rejected anywhere in candidate code (`torch.cuda.current_stream()` is allowed), as
  are the `@` operator, `torch.matmul`/`mm`/`bmm`/`einsum`, `torch.rand*`, `torch.tensor`, `torch.nn`
  and `torch.ops`.
- Every call is timed as one CUDA-graph replay after a 1 GiB cache flush, so launches must be
  capturable on the current stream. Keep first-use compilation and any tuning deterministic and
  cached per shape.

---

## 1. Kernel Structure

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def scale_kernel(x_ptr, out_ptr, n, scale, BLOCK: gl.constexpr):
    # 4 contiguous elements per lane, 64 lanes per wavefront, 4 wavefronts per program.
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4], threads_per_warp=[64], warps_per_cta=[4], order=[0])
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    gl.store(out_ptr + offs, x * scale, mask=mask)


def run(*, x, out, scale) -> None:
    n = x.numel()
    BLOCK = 1024                      # = 4 elems * 64 lanes * 4 warps (one layout tile) or a multiple
    scale_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, scale, BLOCK=BLOCK, num_warps=4)
```

Launch syntax is Triton's: `kernel[grid](args..., num_warps=..., num_stages=...)` with a tuple grid
and `gl.constexpr` meta-parameters passed by keyword. `num_warps` must equal the product of the
layout's `warps_per_cta`, and a warp is a 64-lane wavefront on AMD, so the product of
`threads_per_warp` must be 64.

Compile-time values such as layouts are declared `name: gl.constexpr = ...` inside the kernel, or
passed in as `gl.constexpr` arguments.

---

## 2. How Gluon Differs from Triton

| | Triton (`tl`) | Gluon (`gl`) |
|---|---|---|
| Decorator | `@triton.jit` | `@gluon.jit` |
| Tensor layout | inferred by the compiler | explicit on every creation (`gl.arange(..., layout=...)`) and convertible with `gl.convert_layout` |
| Shapes | `tl.arange(0, N)[:, None]` broadcasting | broadcast through `gl.SliceLayout(dim, parent)` and `gl.expand_dims` |
| Shared memory | automatic | explicit: `gl.allocate_shared_memory`, then load/store of the buffer |
| Matrix multiply | `tl.dot` | explicit MFMA layout and instruction (`gl.amd.cdna4.mfma`) |
| Pipelining | `num_stages` | you write the software pipeline |
| Math / reductions | `tl.sum`, `tl.max`, ... | `gl.sum`, `gl.max`, ... on tensors with a layout |

Many `tl` operations have a `gl` counterpart with the same name (`load`, `store`, `program_id`,
`arange`, `where`, `exp`, `sum`, `max`, `maximum`, `full`, `zeros`, `cdiv`). Read the installed
`triton.experimental.gluon.language` rather than assuming; Gluon's API is experimental and changes
between Triton versions. A Triton kernel cannot be called from a Gluon kernel, so port it fully.

---

## 3. Layouts

`gl.BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)` assigns each tensor
element to a (lane, register) pair. It is the layout for global loads/stores, elementwise math and
reductions.

- `size_per_thread`: contiguous elements one lane holds along each dim. Choose it so a lane loads
  16 bytes (8 bf16 / 4 fp32) along the contiguous dim for vectorized global access.
- `threads_per_warp`: lanes along each dim, product 64 on AMD.
- `warps_per_cta`: wavefronts along each dim, product equal to `num_warps`.
- `order`: dims from fastest-varying to slowest; the memory-contiguous dim goes first
  (`[1, 0]` for a row-major 2-D tensor).

A tensor's layout must tile its shape: if the shape is larger than one tile the layout repeats
(more registers per lane); smaller shapes replicate lanes. Prefer shapes that are multiples of the
tile so no lane holds redundant data.

2-D tiles need index vectors in sliced layouts so that broadcasting lands back in the parent:

```python
layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
rows = gl.arange(0, BM, layout=gl.SliceLayout(1, layout))
cols = gl.arange(0, BN, layout=gl.SliceLayout(0, layout))
offs = rows[:, None] * stride_m + cols[None, :]
tile = gl.load(ptr + offs, mask=(rows[:, None] < M) & (cols[None, :] < N), other=0.0)
```

Other layouts exist for specific hardware paths: `gl.amd.AMDMFMALayout` (the accumulator and
operand layout of CDNA matrix instructions), `gl.DotOperandLayout` (MFMA operands), and
`gl.SwizzledSharedLayout` / padded shared layouts for LDS buffers. Pick the MFMA instruction shape
for the operand dtype and check `gl.amd.cdna4` in the installed package for the exact constructors
and supported shapes before writing one.

---

## 4. Loads, Stores and Reductions

- `gl.load` / `gl.store` take a pointer tensor in a blocked layout, plus `mask=` and `other=`.
  Unmasked access beyond the buffer is undefined; mask the tail unless the workload guarantees
  divisibility.
- Reductions reduce along one axis of a layout: `gl.sum(x, axis=1)`, `gl.max(x, axis=1)`. The result
  has the sliced layout; use it directly for per-row scalars and re-broadcast with `[:, None]`.
  Lane-parallel reductions inside a wavefront are cheap; reductions across warps go through shared
  memory, so prefer one row per warp (or a few) when the row is short enough.
- Accumulate bf16/fp16 inputs in fp32 (`x.to(gl.float32)`), and cast to the output dtype with
  `.to(out_ptr.dtype.element_ty)` before the store.
- To change layouts between a load and a compute stage, use `gl.convert_layout(x, new_layout)`; it
  may spill through LDS, so keep it off the hot loop when you can.
- Use `gl.static_range` for compile-time unrolled loops and `range` for runtime loops.

---

## 5. Matrix Instructions on CDNA4 (gfx950)

A GEMM-like Gluon kernel is built from: global loads in a blocked layout, a store into an LDS
buffer (`gl.allocate_shared_memory` with a shared layout), a load from LDS in the dot-operand
layout, and `gl.amd.cdna4.mfma(a, b, acc)` accumulating in the MFMA layout. Overlap the next tile's
global load with the current tile's MFMA yourself; the compiler does not pipeline for you.
MI355X has 160 KB of LDS per workgroup: size buffers and stage counts to fit.

---

## 6. Common Pitfalls

- **Layout mismatch errors.** Operands of one elementwise op must share a layout; insert
  `gl.convert_layout` or create both tensors with the same layout.
- **`num_warps` and the layout disagree.** The launch's `num_warps` must equal the product of
  `warps_per_cta`; the lane product must be 64 on AMD (not 32 as on NVIDIA).
- **Layout not a `gl.constexpr`.** Layout objects must be compile-time; annotate them
  `gl.constexpr` or pass them as constexpr arguments.
- **Forgotten masks on the tail**, or masks built in the wrong sliced layout.
- **Using Triton APIs.** `tl.*` calls and `@triton.jit` helpers are not valid inside a `@gluon.jit`
  kernel, and a candidate that launches only Triton kernels fails the Gluon target.
- **Experimental API drift.** Names move between Triton releases; inspect the installed module
  (`dir(gl)`, `dir(gl.amd.cdna4)`) before relying on a signature from memory.
- **Per-call work in `run`.** Allocation, view creation and the launch are all `run` may do; no
  precomputation on tensor values, no `.item()`, no caching across calls.

---

## 7. Checklist

- [ ] `solution/kernel.py` defines `run(*, <inputs>, <outputs>)`, writes every output, returns `None`
- [ ] At least one `@gluon.jit` kernel is launched per call; no `tl` kernels, no PyTorch compute
- [ ] Every tensor has a layout whose tile matches its shape; `num_warps` matches `warps_per_cta`
- [ ] Global access is vectorized (16 bytes per lane along the contiguous dimension) and masked
- [ ] fp32 accumulation for bf16/fp16 inputs; output cast to the declared dtype
- [ ] Correct on every workload with the task's tolerance before tuning for speed
