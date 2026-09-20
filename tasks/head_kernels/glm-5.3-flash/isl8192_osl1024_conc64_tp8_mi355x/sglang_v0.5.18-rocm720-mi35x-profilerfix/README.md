# GLM-5.3-Flash workload

ISL **8192**, OSL **1024**, concurrency **64**, tensor parallel world size **8**.
Each leaf is an isolated operator replay from one rank on MI355X (`gfx950`).

Current public runtime: `docker.io/rocm/hyperloom@sha256:da36f56f24cb2897a56be52dd43774c1a75b1500308ed7db3ba32fb8db4d259c`.

Capture image (historical provenance only): `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix`.
The directory name identifies the serving capture; the current runtime above
is public and does not require access to the original cluster.

[workload.json](workload.json) records the exact configuration, task selectors,
and per-kernel runtime requirements. Each leaf carries its own source, frozen
contract, configuration, runners, environment preflight, and shape declaration.

| Kernel | Exact shapes | Task selector |
| --- | --- | --- |
| [elementwise_copy_cluster](elementwise_copy_cluster/README.md) | [SHAPES.md](elementwise_copy_cluster/SHAPES.md) | `head_kernels/glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/elementwise_copy_cluster` |
| [fused_moe_kernel](fused_moe_kernel/README.md) | [SHAPES.md](fused_moe_kernel/SHAPES.md) | `head_kernels/glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/fused_moe_kernel` |

All tasks require fresh matching-image GPU task-validator qualification.
The metadata and CPU layout checks do not establish GPU correctness or speedup.
