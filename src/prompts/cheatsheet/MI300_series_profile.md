# Generic AMD Instinct MI300-Series Profile

`target_gpu_model: MI300` is the backward-compatible generic `gfx942` target.
It does not identify a physical SKU. Do not assume one model's topology or HBM
limits until the visible device has been inspected.

| Model | XCDs | Active CUs | HBM | Peak HBM bandwidth |
| --- | ---: | ---: | ---: | ---: |
| MI300X | 8 | 304 | 192 GB HBM3 | 5.3 TB/s |
| MI325X | 8 | 304 | 256 GB HBM3E | 6.0 TB/s |
| MI300A | 6 | 228 | 128 GB shared HBM3 | approximately 5.3 TB/s |

Identify the model from `torch.cuda.get_device_name()`, HIP device properties,
or `amd-smi static`, then apply the matching resource limits. For reproducible
model-specific tuning, prefer `target_gpu_model: MI300X`, `MI325X`, or `MI300A`
so that AgentKernelArena loads the exact profile.

References:

- [ROCm accelerator hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [ROCm MI300-series workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
