# Triton Kernel Best Practices for RDNA3 (gfx1100, Radeon PRO W7900)

Use the accompanying architecture context for hardware limits. Check the
installed backend before tuning; compiler support evolves independently of
the hardware instruction set.

```python
import triton
target = triton.runtime.driver.active.get_current_target()
print(target)  # Confirm arch='gfx1100' and warp_size=32 for this configuration.
```

## Launch geometry and resource usage

On gfx1100, `num_warps` counts **wave32** waves: `num_warps=4` is 128 threads
per program, not the 256 threads it means on CDNA. Re-tune `num_warps`, tile
shape, and `num_stages` together when moving a kernel from CDNA.

For matrix kernels, `BLOCK_M`/`BLOCK_N` in {32, 64, 128} with `num_warps` in
{2, 4, 8} are reasonable seeds; expand the search as the workload warrants.
Larger tiles improve reuse while consuming more accumulator VGPRs and LDS, and
gfx1100 VGPRs are allocated in granules of 24, so occupancy drops in steps.
Fewer warps do not by themselves guarantee more resident programs.

Inspect the compiled kernel's metadata (`kernel.n_regs`, `kernel.n_spills`,
`kernel.metadata.shared`) and keep shared memory within 64 KiB per workgroup.
Choose pipeline depth from generated code and measurements, not a CDNA preset.

## Matrix lowering and numeric formats

On gfx1100, `tl.dot` with FP16 or BF16 inputs lowers to gfx11 WMMA
(`v_wmma_f32_16x16x16_f16` and the `bf16` form). The hardware has **no FP8,
FP4 or sparse WMMA**, and no MFMA. In AKA's RDNA3 runtime (Triton 3.8.0), an
FP8 `tl.dot` (e4m3fn, e4m3fnuz or e5m2) compiles by upcasting both operands to
FP16 and issuing `v_wmma_f32_16x16x16_f16`, so it runs at the FP16 rate and
saves only load bandwidth. Block-scaled `tl.dot_scaled` and MXFP4 paths are
CDNA features. Compile the exact
dtype pair, accumulator, shape, and layout, then grep the AMD assembly
(`kernel.asm["amdgcn"]`) for `v_wmma` before claiming a matrix path. See the
[Triton dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).

The gfx1100 backend accepts the MFMA tuning knobs `matrix_instr_nonkdim` and
`kpack`, but they do not change WMMA lowering; drop them from autotune configs
rather than spending trials on them.

An elementwise product of two BF16 tensors lowers on gfx1100 to
`v_dot2_bf16_bf16`, whose BF16 result is truncated; for gfx942 the same code
multiplies in FP32 and rounds to nearest. Cast the operands to `tl.float32`
before multiplying and round once at the store when the task's tolerance is
tight.

Follow the compiler's shape constraints, mask problem tails, and keep the
task's input and output dtypes and tolerances. Kernels that import AITER or depend on CDNA-only Triton
paths are outside this device's support; report them instead of rewriting the
reference.

## Memory and reductions

- Coalesce lanes' accesses. Use `tl.multiple_of` and `tl.max_contiguous` only
  when the asserted alignment and contiguity hold for every case; false hints
  can produce incorrect code.
- Mask out-of-bounds loads and stores. For reductions, use neutral values for
  masked lanes, such as negative infinity for a maximum before softmax.
- Fuse work when it saves traffic without excessive registers, synchronization,
  or recomputation. Do not change precision to manufacture bandwidth gains.
- Leave cache policy at its default until measurements justify a change. The
  W7900's 96 MB Infinity Cache can serve repeated small working sets, so time
  with the task's real shapes. See the
  [Triton load API](https://triton-lang.org/main/python-api/generated/triton.language.load.html).
- A row that fits in registers may suit a single-program reduction. Large rows
  spill; compare decompositions instead of assuming one. Preserve stable
  softmax math, including max subtraction and task-defined edge cases.
- Autotuning candidates must preserve state between trials for mutating kernels.
  Follow the existing harness's reset and timing policy.

## Profiling and measurement

In the AKA runtime, trace a separate diagnostic execution:

```bash
rocprofv3 --kernel-trace --output-format csv -- python my_kernel.py
```

Tracing reports dispatch activity, not occupancy or cache-hit counters.
Counters collected with `--pmc` depend on the operator-set amdgpu perf level;
do not change it yourself. On a W7900 at `profile_peak`, `SQ_WAVES` was
populated while `GRBM_COUNT` read zero, so check a counter against a value you
can predict before trusting it. Consult
the [rocprofv3 guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/how-to/using-rocprofv3.html)
for tracing versus counter collection.

Separate first-use JIT, autotuning, and profiling from the harness's scored
measurement. Keep baseline and candidate timing methods, allocations, state
resets, warmups, and cases equivalent. Report unsupported task dependencies or
architecture restrictions without bypassing guards or editing protected files.
