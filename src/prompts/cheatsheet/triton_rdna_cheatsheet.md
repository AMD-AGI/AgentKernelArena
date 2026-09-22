# Triton Kernel Best Practices for RDNA4 (gfx1201)

Use the accompanying architecture context for hardware limits. Check the
installed backend before tuning; compiler support evolves independently of
the hardware instruction set.

```python
import triton
target = triton.runtime.driver.active.get_current_target()
print(target)  # Confirm arch='gfx1201' and warp_size=32 for this configuration.
```

## Launch geometry and resource usage

On the wave32 configuration, `num_warps=4` represents 128 threads per program,
rather than the 256 threads of a wave64 configuration. Re-tune `num_warps`, tile
shape, and pipeline stages together when moving a kernel from CDNA.

For matrix kernels, tiles with M/N dimensions of 32 or 64 and `num_warps` of 4
or 8 are useful initial candidates; expand the search as the workload warrants.
These are not universal optima. Larger tiles may improve reuse while consuming
more accumulator registers or LDS. A reduction in `num_warps` does not by
itself guarantee more resident programs or better performance.

Inspect generated kernel metadata, including registers, spills, and shared
memory. Keep shared memory within the actual per-workgroup limit; the WGP's
aggregate LDS capacity is not the launch budget. Choose pipeline depth based
on generated code and measured performance rather than importing a CDNA preset.

## Matrix lowering and numeric formats

RDNA4 uses WMMA for supported matrix operations. Hardware supports FP8/BF8 as
well as FP16/BF16 and integer matrix types; it is incorrect to declare FP8
universally unavailable on gfx1201. See AMD's
[RDNA4 WMMA overview](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/).

`tl.dot` support depends on the exact dtype pair, accumulator, shape, layout,
and installed compiler. Compile and validate that combination, then inspect
the generated AMD assembly for the expected instructions. Unsupported cases
can fail compilation; do not assume a silent or correct scalar fallback.
Hardware INT4 support alone also does not establish support for a mixed-format
or block-scaled Triton operation.

Do not force CDNA MFMA layouts or carry over MFMA tuning controls such as
`matrix_instr_nonkdim` without checking their meaning in the installed AMD
backend. Follow the compiler's shape constraints, mask problem tails, and
retain the task's input/output dtypes and numerical tolerances. See the
[Triton dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).

## Memory and reductions

- Coalesce lanes' accesses. Use `tl.multiple_of` and `tl.max_contiguous` only
  when the asserted alignment and contiguity are true for every relevant case;
  false compiler hints can produce incorrect code.
- Mask out-of-bounds loads and stores. For reductions, use neutral values for
  masked lanes, such as negative infinity for a maximum before softmax.
- Fuse work when it saves traffic without excessive registers, synchronization,
  or recomputation. Do not change precision to manufacture bandwidth gains.
- Leave cache policy at its default until measurements justify a change. Do
  not prescribe `evict_last` for all read-once data; eviction hints are backend
  dependent and must match actual reuse. See the
  [Triton load API](https://triton-lang.org/main/python-api/generated/triton.language.load.html).
- A row that fits comfortably in registers may suit a single-program reduction.
  Large rows can spill; compare appropriate decompositions rather than assuming
  that all reductions should remain in registers. Preserve stable softmax
  math, including max subtraction and safe handling of task-defined edge cases.
- Autotuning candidates must preserve state between trials for mutating kernels.
  Follow the existing harness's reset and timing policy.

## Profiling and measurement

In the AKA runtime, trace a separate diagnostic execution:

```bash
rocprofv3 --hip-trace --kernel-trace --output-format csv -- python my_kernel.py
```

Tracing reports dispatch/API activity, not occupancy or cache-hit counters.
`rocprofv3 --list-avail` lists the installed tool's available counters; requesting
or collecting them requires separate capability checks. Consult the
[rocprofv3 guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/how-to/using-rocprofv3.html)
for the distinction and device-specific prerequisites.

Separate first-use JIT/autotuning and profiling from the harness's scored
measurement as required by its contract. Keep baseline and candidate timing
methods, allocations, state resets, warmups, and cases equivalent. Report
unsupported task dependencies or architecture restrictions without bypassing
guards or editing protected harness files.
