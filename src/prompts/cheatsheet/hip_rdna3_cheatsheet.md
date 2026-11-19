# HIP Kernel Best Practices for RDNA3 (gfx1100, Radeon PRO W7900)

Use the accompanying architecture context for device limits. These are tuning
directions to measure, not guarantees of speedup.

## Wave operations and launch geometry

Use HIP `warpSize` for lane arithmetic. gfx1100 builds default to wave32;
block sizes such as 64, 128, and 256 are starting points to benchmark, not
mandatory choices. Tail lanes must participate correctly in barriers and
collectives. Do not add `-mwavefrontsize64` unless the task's build already
uses it; wave64 changes every lane-level assumption. In a wave64 build,
device-side `warpSize` is 64 while `hipDeviceProp_t::warpSize` still reports
32, so derive host launch math from the build flags.

HIP `__ballot()` and `__activemask()` return **64-bit** values even on wave32.
Keep their declared mask types rather than inferring a 32-bit return type from
the wave size. `_sync` operations need consistent masks and participation. See
the [HIP warp-function contract](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#warp-cross-lane-functions).

`hipDeviceProp_t::multiProcessorCount` is 48 on the W7900 (WGPs), not the 96
CUs `rocminfo` reports. Use `hipOccupancyMaxActiveBlocksPerMultiprocessor` or
`hipOccupancyMaxPotentialBlockSize` for the compiled kernel, and keep per-thread
registers, registers per SIMD, and waves per CU or WGP as separate units.

## Memory, LDS, and register pressure

- Make adjacent lanes access adjacent elements where the task layout permits.
  Vector loads cut instruction count but need valid alignment, in-bounds tails,
  and spare registers; they are not automatically faster.
- Reuse data in registers or LDS when it pays. Keep static plus dynamic LDS
  within 64 KiB per workgroup. Padding or swizzling can cut bank conflicts, but
  the extra storage can reduce occupancy.
- VGPRs are allocated in granules of 24 on gfx1100, so occupancy changes in
  steps (96 VGPRs allow 16 waves per SIMD, 120 allow 12, 144 allow 10). Check
  the compiled resource report and scratch usage after changing tiles or
  unrolling; a single `v_readlane` is not evidence of spilling.
- Without `__launch_bounds__`, HIP assumes 1024-thread blocks and caps the
  kernel at 192 VGPRs. If the report shows 192 VGPRs plus scratch and the
  launch uses 256 threads or fewer, `__launch_bounds__(256)` allows up to 256
  VGPRs and can remove the spills, at lower occupancy.
- Block-local aggregation can reduce global atomic contention when it preserves
  the required semantics. Reordered floating-point sums must pass the original
  tolerances. Do not change output dtype merely to save bandwidth.

Pinned host buffers and asynchronous copies help only when transfers are inside
the declared measurement boundary. Keep that boundary identical for both
implementations.

## Matrix operations and compilation

gfx1100 matrix math uses the gfx11 WMMA builtins, for example
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)` with `a`, `b` as 16 x
`_Float16` and `c` as 8 x `float` per lane. The `bf16`, `iu8` and `iu4`
variants follow the same pattern; the f16- and bf16-output forms take an extra
OPSEL argument. Lane `l` loads row `l % 16` of A and column `l % 16` of B, so
**lanes 16-31 duplicate lanes 0-15**; result VGPR `v` of lane `l` is
`D[2*v + l/16][l % 16]`. If lanes 16-31 do not replicate the inputs, half of
`D` is wrong. AMD's [RDNA3 WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/)
has a worked example; rocWMMA also supports gfx11 if the task can use it.

There is no FP8, FP4, or sparse WMMA and no MFMA on gfx1100. Do not copy gfx12
`_w32_gfx12` intrinsics or CDNA `__builtin_amdgcn_mfma_*` calls; they fail to
compile for this target. For FP8 data, dequantize to FP16 or BF16 in the
operand load and use FP16/BF16 WMMA, within the task's tolerances.

For a standalone diagnostic kernel, compile for the actual target and look for
`v_wmma_` in the assembly:

```bash
hipcc -O3 --offload-arch=gfx1100 --save-temps -c kernel.hip -o kernel.o
```

Use the task's declared build commands for evaluation. Add fast-math, launch
bounds, or unrolling only after checking correctness and resource usage. Do not
rewrite protected build or test files to force a result.

## Profiling

Inside the AKA runtime, trace a separate diagnostic run of your application:

```bash
rocprofv3 --kernel-trace --output-format csv -- ./my_app
```

This gives dispatch durations, **not hardware counters**. Counters collected
with `--pmc` depend on the operator-set amdgpu perf level, and you must not
change it yourself. On a W7900 at `profile_peak`, `SQ_WAVES` was populated
while `GRBM_COUNT` read zero; check each counter against a value you can
predict before using it. Without trustworthy counters, classify the bottleneck
from kernel durations, dispatch counts, and the occupancy steps above. The
[rocprofv3 guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/how-to/using-rocprofv3.html)
covers tracing versus counter collection.

Profiling perturbs execution. Collect scored performance separately with the
original harness, warmups, samples, synchronization, and state resets.
