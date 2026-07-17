# AMD CDNA 3 (`gfx942`) Kernel Optimization Context

Use this shared architecture guidance together with the target GPU's model
profile. The model profile is authoritative for XCD count, active compute units,
HBM capacity and bandwidth, and partition topology.

## Execution model

- CDNA 3 executes Wave64 wavefronts. Cross-lane operations must preserve the
  64-lane execution model.
- A workgroup can contain multiple wavefronts, up to 1024 work-items.
- An XCD is the compute chiplet. Each XCD contains 40 physical CUs, with 38
  active CUs at the aggregate device level and two disabled for yield.
- Each XCD has 4 MiB of L2 cache. Never infer aggregate L2 capacity without
  consulting the model profile and the visible accelerator partition.

## Memory hierarchy and locality

- Each CU has 64 KiB of LDS and a 32 KiB L1 vector cache. Keep LDS accesses
  bank-friendly and use padding when a tile layout would otherwise conflict.
- The MI300-series packages described by the model profiles have 256 MiB of
  last-level Infinity Cache. It is shared package infrastructure, not a single
  low-latency cache local to every XCD.
- Coalesce adjacent lanes onto aligned contiguous memory regions. Avoid random
  traffic that repeatedly crosses XCD or memory-partition boundaries.
- Size tiles against the resources of the *visible logical device*. A CPX, QPX,
  or DPX partition exposes fewer XCDs and less HBM than the full SPX device.
- Do not hardcode HBM capacity or peak bandwidth from another `gfx942` model.

## Matrix and scalar data types

- CDNA 3 matrix cores support FP64, FP32, TF32, FP16, BF16, FP8, and INT8
  operations. Use an MFMA shape that is valid for the selected input and
  accumulator types.
- Align LDS loading, register tiling, and output layout with the selected MFMA
  instruction. Inspect generated code when performance depends on a particular
  lowering.
- CDNA 3 uses the FNUZ FP8 variants. Do not assume the OCP FP8 or MXFP formats
  available on later CDNA generations.

## Runtime topology checks

Before topology-sensitive tuning, inspect the environment rather than assuming
the full physical package is visible:

1. Confirm `gcnArchName` is `gfx942` and inspect the device name, CU count, and
   visible memory through PyTorch/HIP device properties.
2. Use `amd-smi` or `rocminfo` when available to identify accelerator and memory
   partition modes.
3. Benchmark and compare kernels under the same partition mode. A launch tuned
   for an eight-XCD SPX device may be inappropriate for a one-XCD CPX device.

## Kernel-generation constraints

1. Preserve synchronization semantics. Replace a workgroup barrier with
   wave-level synchronization only when all communicating work-items are proven
   to remain in the same wavefront.
2. Bound VGPR and LDS use based on measured occupancy; avoid register spilling
   and do not assume that maximum occupancy always gives maximum throughput.
3. Fuse memory-bound operations when it reduces measured global-memory traffic,
   but retain numerically equivalent behavior and validate every supported
   shape.
4. Prefer runtime-derived CU and memory counts over constants embedded in launch
   heuristics.

Reference: [AMD Instinct MI300 series microarchitecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html)
