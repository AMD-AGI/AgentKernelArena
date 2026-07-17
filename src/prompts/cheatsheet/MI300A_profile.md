# AMD Instinct MI300A Model Profile

## Physical resources

- Architecture target: CDNA 3, `gfx942`
- Compute topology: 6 XCDs, 228 active CUs (38 active CUs per XCD)
- Aggregate L2: 24 MiB (4 MiB per XCD)
- Last-level Infinity Cache: 256 MiB
- Memory: 128 GB HBM3 shared by the integrated CPU and GPU, with approximately
  5.3 TB/s peak theoretical bandwidth
- Package topology: three Zen 4 CPU chiplets (24 CPU cores) and six CDNA 3 XCDs

## APU and partition-aware tuning

MI300A is an APU, not a discrete MI300X-class accelerator. CPU and GPU traffic
can share HBM capacity and bandwidth, so leave allocation headroom and avoid
assuming that the GPU owns all 128 GB. Unified physical memory also does not make
every placement or access path equally fast; use the framework's device tensors
and measure migrations and CPU/GPU contention when they are part of a task.

Accelerator partitioning can expose fewer than six XCDs, including one-XCD CPX
logical devices. Detect the visible CU count, memory capacity, accelerator
partition, and memory partition before choosing persistent grids or whole-device
tile counts. Never reuse MI300X/MI325X constants of eight XCDs or 304 CUs.

References:

- [AMD CDNA 3 and MI300A overview](https://www.amd.com/en/technologies/cdna.html)
- [ROCm accelerator hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [AMD SMI GPU partitioning](https://rocm.docs.amd.com/projects/amdsmi/en/latest/conceptual/partition.html)
