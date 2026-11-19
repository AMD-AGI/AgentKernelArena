# AMD RDNA 3 (gfx1100) Kernel Optimization Context: Radeon PRO W7900

This guide targets AKA's `W7900` configuration: a Radeon PRO W7900 or W7900
Dual Slot reporting `gfx1100`. AKA refuses the run on any other board, so the
product facts below apply to the device you are running on. Still read the
actual device properties before sizing work; clocks, free VRAM, and sharing
with other processes vary.

## Execution and resource limits

- HIP and Triton compile gfx1100 for **wave32** by default. LLVM also supports
  wave64 for this target (`-mwavefrontsize64`). Use the compiled wave size and
  HIP `warpSize` for lane arithmetic; do not transplant CDNA wave64 masks.
- The W7900 has **96 CUs in 48 WGPs**. A WGP holds two CUs, each with two SIMD32
  units. HIP `multiProcessorCount` and torch `multi_processor_count` report
  **48** (the WGP count) on this board, while `rocminfo` reports 96 compute
  units. State which unit you mean before sizing a persistent grid.
- Each gfx1100 SIMD has 1536 VGPRs, allocated to waves in granules of 24, with
  up to 256 VGPRs per wave and at most 16 resident waves per SIMD. HIP's
  default 1024-thread launch bound caps a kernel at 192 VGPRs, so a
  register-heavy kernel spills unless a smaller `__launch_bounds__` lets the
  compiler use up to 256. Use the compiler resource report and occupancy APIs
  for the real kernel; higher occupancy is not automatically faster.
- The per-workgroup LDS limit is **64 KiB**. Query `sharedMemPerBlock` and count
  static plus dynamic allocations before choosing tiles.
- The board has 48 GB of GDDR6. Check free VRAM before experiments; an
  allocation failure does not authorize shrinking task shapes or skipping
  correctness cases.

The [LLVM target guide](https://llvm.org/docs/AMDGPUUsage.html#processors)
documents gfx1100 wave-size features. AMD's
[occupancy guide](https://gpuopen.com/learn/occupancy-explained/) explains SIMD
residency and the difference between occupancy and performance.

## Matrix instructions and numeric formats

RDNA3 has **WMMA**, not the CDNA MFMA family; MFMA builtins and `v_mfma_*`
code do not compile for gfx1100. gfx11 WMMA is dense 16x16x16 with **FP16,
BF16, INT8 and INT4** inputs. FP16/BF16 accumulate in FP32 (or FP16/BF16) and
INT8/INT4 in INT32. There is **no FP8/BF8, FP4, or sparse WMMA** on gfx1100;
those are gfx12 or CDNA features. Code that relies on fnuz FP8, MXFP4, scaled
dot, or AITER's CDNA kernels is not portable to this device.

The gfx11 operand layout differs from gfx12: in wave32, lanes 16-31 must hold
a **copy** of the A and B data in lanes 0-15, and the FP32 result in VGPR `v`
of lane `l` is element `D[2*v + l/16][l % 16]`. Do not reuse RDNA4 (gfx12)
fragment layouts or intrinsic signatures. AMD's
[RDNA3 WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/) walks through it.

Hardware support and compiler support are separate questions. Validate the
exact input format, accumulator, layout, and shape with the installed
toolchain and inspect the generated code for `v_wmma_*` before claiming a
matrix path. Lower precision, reassociation, and fast-math must still satisfy
the task's original correctness contract.

## Memory and launch tuning

Coalesce neighboring lanes' accesses and keep alignment and tail handling
correct when vectorizing. The W7900 has a 6 MB L2 and a 96 MB Infinity Cache in
front of 864 GB/s of GDDR6, so a working set that fits in the Infinity Cache can
time well above the DRAM rate; measure with the task's real shapes. Stage reuse
through LDS only when saved traffic outweighs synchronization and occupancy
costs.

RDNA3 can dual-issue some VALU pairs in wave32 (`v_dual_*` in disassembly).
The compiler chooses them; do not count on a 2x ALU rate. WMMA and VALU work in
the same wave are largely serial. Sweep launch geometry under the task's
unchanged benchmark policy.

## Profiling and task integrity

Use `rocprofv3 --kernel-trace` for dispatch durations. Hardware counters
(`--pmc`) depend on the amdgpu perf level, which the operator controls; do not
change it yourself. On a W7900 at `profile_peak`, `SQ_WAVES` was populated
while `GRBM_COUNT` read zero, so check a counter against a value you can
predict (such as the launched wave count) before trusting it, and treat zeros
as missing data. A profiler on the PATH does not qualify AKA evaluation-tool
sidecars; they are refused on gfx1100.

Keep profiling separate from scoreable timing. Honor `platform_support` and the
protected harness boundaries. A task whose `required_arch` excludes gfx1100 is
skipped by design; report it rather than bypassing the guard.
