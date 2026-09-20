# DeepSeek-V4-Pro workload

ISL **8192**, OSL **1024**, concurrency **64**, tensor parallel world size **8**.
Each leaf is an isolated operator replay from one rank on MI355X (`gfx950`).

Capture image: `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix`.

[workload.json](workload.json) records the exact configuration, task selectors,
and per-kernel runtime requirements. Each leaf carries its own source, frozen
contract, configuration, runners, environment preflight, and shape declaration.

| Kernel | Exact shapes | Task selector |
| --- | --- | --- |
| [dsa_sparse_mla_attn](dsa_sparse_mla_attn/README.md) | [SHAPES.md](dsa_sparse_mla_attn/SHAPES.md) | `head_kernels/deepseek-v4-pro/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/dsa_sparse_mla_attn` |
| [moe_stage1_grouped_gemm_silu_flydsl](moe_stage1_grouped_gemm_silu_flydsl/README.md) | [SHAPES.md](moe_stage1_grouped_gemm_silu_flydsl/SHAPES.md) | `head_kernels/deepseek-v4-pro/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/moe_stage1_grouped_gemm_silu_flydsl` |
| [moe_stage2_down_proj_reduce_opus_a8w4](moe_stage2_down_proj_reduce_opus_a8w4/README.md) | [SHAPES.md](moe_stage2_down_proj_reduce_opus_a8w4/SHAPES.md) | `head_kernels/deepseek-v4-pro/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/moe_stage2_down_proj_reduce_opus_a8w4` |

All tasks require fresh matching-image GPU task-validator qualification.
The metadata and CPU layout checks do not establish GPU correctness or speedup.
