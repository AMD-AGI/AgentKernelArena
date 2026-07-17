# AMD Instinct MI325X Model Profile

## Physical resources

- Architecture target: CDNA 3, `gfx942`
- Compute topology: 8 XCDs, 304 active CUs (38 active CUs per XCD)
- Aggregate L2: 32 MiB (4 MiB per XCD)
- Last-level Infinity Cache: 256 MiB
- Memory: 256 GB HBM3E, 6.0 TB/s peak theoretical bandwidth

## Partition-aware tuning

The full SPX device exposes eight XCDs and 256 GB. Accelerator partitioning can
instead expose DPX (four XCDs, 128 GB), QPX (two XCDs, 64 GB), or CPX (one XCD,
32 GB) logical devices. Memory partitioning is independent. Detect the visible
configuration and do not carry over MI300X's smaller per-partition memory limits.

MI300X and MI325X have the same active-CU topology, but their memory technology,
capacity, and peak bandwidth differ. Re-benchmark memory-bound launch choices on
MI325X rather than treating MI300X results as interchangeable.

References:

- [AMD Instinct MI325X accelerator overview](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
- [ROCm MI300-series workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
