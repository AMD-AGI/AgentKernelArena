# AMDGPU Assembly (gfx950) Kernel Best Practices

Reference: the AMD Instinct CDNA4 ISA reference guide (instruction encodings, wait states and hazards) and
the [LLVM AMDGPU usage guide](https://llvm.org/docs/AMDGPUUsage.html) (code object, kernel descriptor,
`.amdhsa_*` directives and kernel metadata).

---

## Candidate entry point (schema v3)

Your whole solution is the directory `solution/`; it is the only editable tree. The task fixes the
call, not the implementation. Entry: `solution/kernel.py::run`. Write each kernel in a `.s` file in
`solution/`, load it at import time with `aka_asm.load` and launch it from `run` with
`kernel.launch`. Helper `.py` files form one flat package, imported relatively.

```python
def run(*, <inputs>, <outputs>) -> None: ...
```

For example, for a GEMM `out = a @ b.T`:

```python
import torch
import aka_asm

GEMM = aka_asm.load("gemm.s", "gemm_kernel")          # assemble + load once, at import time


def run(*, a, b, out) -> None:
    m, k = a.shape
    n = b.shape[0]
    GEMM.launch(grid=((n + 63) // 64, (m + 63) // 64, 1), block=(256, 1, 1),
                args=[a, b, out, aka_asm.u32(m), aka_asm.u32(n), aka_asm.u32(k)],
                shared_bytes=0)
```

`_aka_eval/CANDIDATE.md` in the workspace states the exact signature, shapes and dtypes.

- `aka_asm.load(source, symbol)` assembles `source` with the image's ROCm `clang`
  (`-x assembler -target amdgcn-amd-amdhsa -mcpu=<current GPU>`), links it with `ld.lld -shared`,
  caches the code object by content and loads `symbol`. Call it at import time, not in `run`.
- `kernel.launch(grid=, block=, args=, shared_bytes=0)` launches on the current PyTorch stream.
  `grid` counts **workgroups**, not threads. `shared_bytes` is dynamic LDS on top of the kernel's static
  LDS.
- `args` are packed in order, each at its natural alignment, and the buffer is padded to 8 bytes:
  - a tensor becomes its 64-bit device pointer (8 bytes, 8-aligned);
  - scalars only through `aka_asm.u32`, `aka_asm.i32` (4 bytes), `aka_asm.u64`, `aka_asm.i64`
    (8 bytes) and `aka_asm.f32` (4 bytes);
  - a bare Python `int`/`float` is rejected, so the width is always explicit.

  The `.args` list in the kernel's `.amdgpu_metadata` must describe exactly this layout.

Candidate rules (enforced by the evaluator, not a sandbox):

- `run` writes every output (and every mutable input) in place and returns `None`. Outputs are
  preallocated and poisoned before each call; other inputs are read-only.
- All computation runs in your assembly kernels. Host code may only allocate, take views, make
  same-dtype copies and launch. PyTorch arithmetic, reductions, dtype conversions and library
  operators raise at run time. Every call must launch at least one of your kernels.
- `solution/` holds only `.py` and `.s` files. `.incbin` and `.include` are rejected: each kernel's
  whole source is in its `.s` file. No prebuilt code objects.
- Python code follows the same rules as other Python-family candidates: only `torch`, `aka_asm` and a
  few standard modules (`math`, `functools`, `itertools`, `typing`, `dataclasses`, ...) import. No
  `eval`/`exec`, reflection, module mutation, I/O, printing, synchronization, host round trips
  (`.item()`, `.cpu()`), randomness, global PyTorch state, or imports of task, reference, baseline or
  evaluator modules. Keep no state derived from inputs across calls.
- Every call is timed as one CUDA-graph replay after a 1 GiB cache flush. `launch` is capturable, and
  HIP copies the argument buffer when the launch is recorded, so build arguments freshly in `run`.

---

## 1. Code Object Layout

Each kernel needs three parts: the code, a kernel descriptor (`.amdhsa_kernel`), and an entry in the
code object metadata (`.amdgpu_metadata`). A minimal kernel that writes `out[i] = value + i` for one
64-lane workgroup, with a pointer and a `u32` argument (`args=[out, aka_asm.u32(value)]`):

```asm
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl fill
.p2align 8
.type fill,@function
fill:
  s_load_dwordx2 s[4:5], s[0:1], 0x0      // argument 0: out pointer (offset 0, 8 bytes)
  s_load_dword s6, s[0:1], 0x8            // argument 1: value (offset 8, 4 bytes)
  v_lshlrev_b32 v1, 2, v0                 // byte offset = work-item id * 4
  s_waitcnt lgkmcnt(0)
  v_add_u32 v2, s6, v0
  global_store_dword v1, v2, s[4:5]
  s_endpgm

.rodata
.p2align 6
.amdhsa_kernel fill
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 8
  .amdhsa_accum_offset 4
  .amdhsa_kernarg_size 16
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: fill
    .symbol: fill.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 4
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 8, .value_kind: by_value }
...
.end_amdgpu_metadata
```

Guidelines:
- With `.amdhsa_user_sgpr_kernarg_segment_ptr 1` the argument pointer arrives in `s[0:1]`. With
  `.amdhsa_system_sgpr_workgroup_id_x 1` (on by default) the workgroup ID X arrives in the first SGPR
  after the user SGPRs (`s2` here); Y and Z follow when enabled. The work-item ID X arrives in `v0`;
  `.amdhsa_system_vgpr_workitem_id 1`/`2` adds Y/Z in `v1`/`v2`.
- `.amdhsa_next_free_vgpr` and `.amdhsa_next_free_sgpr` must cover every register the code uses.
  gfx950 has a unified VGPR file, so `.amdhsa_accum_offset` is required: a multiple of 4, the first
  AccVGPR. Set it to `.amdhsa_next_free_vgpr` (rounded up to 4) when the kernel uses no AccVGPRs.
- Static LDS goes in `.amdhsa_group_segment_fixed_size` and `.group_segment_fixed_size`, and both must
  agree. Static plus `shared_bytes` must fit gfx950's 160 KiB of LDS per workgroup.
- `.max_flat_workgroup_size` must be at least the launched workgroup size.
- `.kernarg_segment_size` and `.amdhsa_kernarg_size` cover the packed arguments; offsets follow the
  natural-alignment packing above (a `u32` after a pointer sits at offset 8; a pointer after a `u32`
  at offset 8 + 4 rounds up to 16).
- If an argument entry has a `.name`, do not call it `y`, `n`, `yes`, `no`, `on`, `off`, `true` or
  `false`: the LLVM YAML reader in the ROCm 7.x image reads those as booleans, even quoted, and the
  assembler stops at `.end_amdgpu_metadata` with `invalid HSA metadata`.
- Check what was assembled with `/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 <file>.co` on a
  code object you build the same way in a scratch directory.

---

## 2. Memory Access

- `global_load_dwordx4 v[4:7], v1, s[4:5]` loads 16 bytes per lane from the SGPR base plus a 32-bit
  VGPR offset. 16 bytes per lane across a wave is the widest coalesced load.
- `buffer_load_dwordx4` with a buffer resource gives hardware bounds checks: out-of-range lanes return 0
  and out-of-range stores are dropped. That handles unaligned tails without branches; workloads include
  sizes that are not multiples of 16.
- Issue independent loads back to back and wait once. `s_waitcnt vmcnt(N)` waits until at most N vector
  memory operations are outstanding, so a nonzero N overlaps the next loads with compute.
- `s_waitcnt lgkmcnt(0)` covers scalar loads, LDS and GDS. Scalar loads can return out of order, so
  only `lgkmcnt(0)` is safe after several `s_load`.
- Small-M GEMMs read the weight once and are bound by memory bandwidth. MI355X HBM gives about 8 TB/s;
  scoring flushes the caches before every timed call, so plan for cold weights.

---

## 3. Arithmetic for BF16 and FP8

- `v_dot2_f32_bf16 vdst, vsrc0, vsrc1, vsrc2` multiplies two packed BF16 pairs and adds the products
  and `vsrc2` in FP32.
- MFMA on gfx950: `v_mfma_f32_16x16x32_bf16` and `v_mfma_f32_32x32x16_bf16` accumulate BF16 products in
  FP32; FP8 and MXFP4/MXFP6 variants exist (`v_mfma_f32_16x16x128_f8f6f4`, scaled forms).
- `v_cvt_pk_bf16_f32 vdst, vsrc0, vsrc1` rounds two FP32 values to BF16 with round-to-nearest-even and
  packs them.
- Keep accumulation in FP32. Correctness compares every element with a float64 reference, within a
  bound set by the baseline's own error, so an accumulation that is less accurate than the baseline's
  fails.

---

## 4. Cross-Lane Reductions and LDS

- DPP row operations reduce within rows of 16 lanes, for example `v_add_f32_dpp v1, v1, v1 row_shr:1`.
  `row_bcast:15` and `row_bcast:31` with a row mask combine rows across the wave.
- `v_readfirstlane_b32 s5, v3` moves a uniform value to an SGPR.
- Combine waves through LDS: `ds_write_b32`, then `s_waitcnt lgkmcnt(0)` and `s_barrier`, then
  `ds_read_b32`.

---

## 5. Wait States Are Your Job

The assembler inserts no hazard NOPs. The compiler normally does this, so hand-written code must.
- A DPP instruction that reads a VGPR written by the previous VALU instruction needs 2 wait states
  (`s_nop 1`).
- On gfx950, a `v_readlane_b32` or `v_readfirstlane_b32` right after the VALU write of its source VGPR
  can read the old value. Put `s_nop 4` between them.
- MFMA results have a latency before dependent reads; the CDNA4 ISA guide lists the required wait
  states per MFMA shape.
- `s_nop N` gives N + 1 wait states. Independent instructions between the two count as wait states too.

Wrong results that change from run to run, or that disappear when you add `s_nop`, usually point to a
missing wait state. A kernel that never returns is stopped by the evaluator's timeout and fails.
