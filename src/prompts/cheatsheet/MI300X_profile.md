# AMD Instinct MI300X Model Profile

## Physical resources

- Architecture target: CDNA 3, `gfx942`
- Compute topology: 8 XCDs, 304 active CUs (38 active CUs per XCD)
- Aggregate L2: 32 MiB (4 MiB per XCD)
- Last-level Infinity Cache: 256 MiB
- Memory: 192 GB HBM3, 8192-bit interface, 5.3 TB/s peak theoretical bandwidth

## Partition-aware tuning

The full SPX device exposes eight XCDs and 192 GB. Accelerator partitioning can
instead expose DPX (four XCDs, 96 GB), QPX (two XCDs, 48 GB), or CPX (one XCD,
24 GB) logical devices. Memory partitioning is independent. Detect the visible
configuration and do not assume CPX, SPX, or a particular NPS mode.

Tune grid sizing and persistent-work scheduling against the visible CU count.
Use 192 GB only as a physical-SPX upper bound; leave headroom for the runtime and
other allocations.

References:

- [AMD Instinct MI300X accelerator specifications](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [ROCm MI300-series workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
