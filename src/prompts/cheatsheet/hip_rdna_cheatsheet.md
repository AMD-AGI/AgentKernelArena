# HIP Kernel Best Practices for RDNA4 (gfx1201)

Use the accompanying architecture context for device limits. These are tuning
directions to measure, not guarantees of speedup.

## Wave operations and launch geometry

Use HIP `warpSize` for lane arithmetic. The normal configuration is wave32;
block sizes such as 64, 128, and 256 are starting points to benchmark, not
mandatory choices. Tail lanes must participate correctly in barriers and
collectives.

HIP `__ballot()` and `__activemask()` return **64-bit** values even on wave32.
Keep their declared mask types instead of inferring a 32-bit C++ return type
from the number of active lanes. `_sync` operations require consistent masks
and participation. See the
[HIP warp-function contract](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#warp-cross-lane-functions).

Use `hipOccupancyMaxActiveBlocksPerMultiprocessor` or
`hipOccupancyMaxPotentialBlockSize` to explore the compiled kernel's resource
limits. Tune block size and work per thread together. Register counts reported
per thread, physical registers per SIMD, and waves per CU are different units;
do not insert them into the same occupancy formula.

## Memory, LDS, and register pressure

- Make adjacent lanes access adjacent elements where the task layout permits.
  Vector loads can reduce instruction count, but require valid alignment,
  in-bounds tails, and enough registers. They are not automatically faster.
- Reuse data in registers or LDS when useful. Include static and dynamic LDS
  in the device's per-workgroup limit. Padding or swizzling can reduce bank
  conflicts, but the extra storage can reduce occupancy.
- Inspect compiler resource metadata and scratch usage when changing tile
  sizes or unrolling. An `s_load_dword` or `v_readlane` instruction alone is
  not evidence of spilling. Compare the full compiled resource report.
- Uniform control flow can reduce divergence; scalar instructions still have
  cost. Do not assume predicated arithmetic always beats a branch.
- Block-local aggregation can reduce global atomic contention when it preserves
  the required semantics. Reordered floating-point sums must pass the original
  tolerances. Do not change output dtype merely to save bandwidth.

Pinned host buffers and asynchronous copies can help transfer-heavy workloads.
They do not improve a device-only score unless those transfers are inside the
declared measurement boundary. Preserve that boundary for both implementations.

## Matrix operations and compilation

Use WMMA-capable implementations for supported matrix shapes and types. RDNA4
has FP8/BF8 and INT4 hardware matrix operations as well as FP16/BF16/INT8;
actual availability through HIP wrappers depends on the installed stack.
AMD's [WMMA examples](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/)
show the gfx12 intrinsic suffix and changed fragment layout. Do not substitute
MFMA intrinsics or assume an RDNA3 fragment layout is compatible.

For a standalone diagnostic kernel, compile for the actual target:

```bash
hipcc -O3 --offload-arch=gfx1201 --save-temps -c kernel.hip -o kernel.o
```

Use the task's declared build commands for evaluation. Add fast-math, launch
bounds, or unrolling only after checking their effect on correctness and
resource usage. Do not rewrite protected build/test files to force a result.

## Profiling

Inside the AKA runtime, use a separate diagnostic run of your application:

```bash
rocprofv3 --hip-trace --kernel-trace --output-format csv -- ./my_app
```

This collects HIP API and kernel traces, **not hardware counters**. Inspect
kernel durations and dispatch counts first. Query available counters with
`rocprofv3 --list-avail`; support and prerequisites are version/device dependent.
Do not assume `rocprof-compute`/Omniperf support from the executable's presence,
or infer cache hit rate or WMMA utilization from timing alone. The
[rocprofv3 guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/how-to/using-rocprofv3.html)
describes tracing versus counter collection and RDNA counter prerequisites.

Profiling can perturb execution. Collect scored performance separately with
the original harness, warmups, samples, synchronization, and state resets.
