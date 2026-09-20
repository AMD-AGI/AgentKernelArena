# Kimi-K3 workload

ISL **8192**, OSL **1024**, concurrency **64**, tensor parallel world size **8**.
Each leaf is an isolated operator replay from one rank on MI355X (`gfx950`).

Capture image: `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830`.

[workload.json](workload.json) records the exact configuration, task selectors,
and per-kernel runtime requirements. Each leaf carries its own source, frozen
contract, configuration, runners, environment preflight, and shape declaration.

| Kernel | Exact shapes | Task selector |
| --- | --- | --- |
| [fwd_grouped_kernel_stage1](fwd_grouped_kernel_stage1/README.md) | [SHAPES.md](fwd_grouped_kernel_stage1/SHAPES.md) | `head_kernels/kimi-k3/isl8192_osl1024_conc64_tp8_mi355x/sglang-rocm-k3_rocm720-mi35x-k3-20260727-tl312-08011830/fwd_grouped_kernel_stage1` |
| [moe_gemm1_stage1](moe_gemm1_stage1/README.md) | [SHAPES.md](moe_gemm1_stage1/SHAPES.md) | `head_kernels/kimi-k3/isl8192_osl1024_conc64_tp8_mi355x/sglang-rocm-k3_rocm720-mi35x-k3-20260727-tl312-08011830/moe_gemm1_stage1` |
| [moe_gemm2_stage2](moe_gemm2_stage2/README.md) | [SHAPES.md](moe_gemm2_stage2/SHAPES.md) | `head_kernels/kimi-k3/isl8192_osl1024_conc64_tp8_mi355x/sglang-rocm-k3_rocm720-mi35x-k3-20260727-tl312-08011830/moe_gemm2_stage2` |

All tasks require fresh matching-image GPU task-validator qualification.
The metadata and CPU layout checks do not establish GPU correctness or speedup.
