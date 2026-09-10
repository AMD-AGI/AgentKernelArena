# AMD RDNA 4 (gfx1201) Kernel Optimization Context

This guide targets AKA's `RDNA4` configuration (`gfx1201`). The architecture
token identifies an instruction set, not a board's VRAM capacity, clock rate,
cache capacity, or enabled compute-unit count. Read the actual device properties;
do not assume that every gfx1201 device has the specifications of an RX 9070 XT
or a Radeon AI PRO R9700.

## Execution and resource limits

- The normal HIP/Triton configuration here uses **wave32**. LLVM also supports
  wave64 for this target. Use the compiled wave size and HIP `warpSize` when
  reasoning about lane operations; do not transplant wave64 reductions or masks.
- A WGP contains two CUs, each with two SIMD32 units. Occupancy is accounted for
  **per SIMD**, with up to 16 resident wave32 waves per SIMD. Keep SIMD, CU, and
  WGP units explicit when comparing compiler reports and device APIs.
- VGPR allocation, LDS allocation, block size, and wave size jointly constrain
  residency. Use compiler resource reports and occupancy APIs for the actual
  kernel. A universal threshold such as "fewer than 96 VGPRs" is not a reliable
  optimization rule; higher occupancy is not always faster.
- WGP LDS capacity is not a single workgroup's allocation budget. The tested
  gfx1201 runtime exposes **64 KiB per workgroup**. Query `sharedMemPerBlock`
  and account for both static and dynamic allocations before choosing tiles.
- Check available VRAM before experiments. An allocation failure does not
  authorize reducing the task's required shapes or skipping correctness cases.

The [LLVM target guide](https://llvm.org/docs/AMDGPUUsage.html#processors)
documents wave-size target features. AMD's
[occupancy guide](https://gpuopen.com/learn/occupancy-explained/) explains SIMD
residency and the distinction between occupancy and performance.

## Matrix instructions and numeric formats

RDNA4 uses **WMMA**, not the CDNA MFMA instruction family. Its hardware matrix
capabilities include FP16, BF16, FP8/BF8, and integer formats including INT8 and
INT4. INT4 support does not imply native FP4 or arbitrary mixed-format support.
See AMD's [RDNA4 WMMA guide](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/)
for instruction rates and examples.

Hardware support and compiler/library support are separate. Validate the exact
input format, accumulator type, layout, and shape with the installed toolchain;
inspect generated code before claiming a WMMA path. RDNA4 HIP WMMA intrinsics
use gfx12-specific signatures and layouts. Do not copy RDNA3 intrinsics or
CDNA FP8/MFMA code unchanged. Floating-point reassociation, lower precision, and
fast-math options must still satisfy the task's original correctness contract.

## Memory and launch tuning

Coalesce neighboring lanes' accesses and preserve alignment and tail handling
when vectorizing. Stage reused data through LDS only when the saved traffic
outweighs synchronization and occupancy costs. Cache sizes and bandwidth vary
by product; measure locality benefits rather than assuming a fixed cache-line
size or a universal cache-resident working-set limit.

Check whether a device API's multiprocessor count denotes CUs or WGPs for the
installed runtime. Do not blindly multiply that count or hardcode a full-chip
CU count into a persistent grid. Sweep launch geometry under the task's
unchanged benchmark policy.

## Profiling and task integrity

Use `rocprofv3` for HIP/kernel tracing in the configured runtime. Tracing
provides dispatch durations and API activity; it does not by itself collect
occupancy, cache-hit, or matrix-utilization counters. Counter availability
depends on the GPU and installed profiler. See the language guides for commands.
The presence of a profiler does not qualify AKA evaluation-tool sidecars.

Keep profiling separate from scoreable timing. Honor `platform_support` and
protected harness boundaries; a conflicting architecture requirement or
unsupported API is a compatibility issue to report, not a guard to bypass.
