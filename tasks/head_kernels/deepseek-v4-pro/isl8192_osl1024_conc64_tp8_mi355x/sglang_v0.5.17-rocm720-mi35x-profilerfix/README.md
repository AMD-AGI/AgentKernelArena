# DeepSeek-V4-Pro workload

ISL **8192**, OSL **1024**, concurrency **64**, tensor parallel world size **8**.
Each leaf is an isolated operator replay from one rank on MI355X (`gfx950`).

Current public runtime: `docker.io/rocm/hyperloom@sha256:1f5464829559b086eb66f9b803cb9c7a817438c43edff2d5ef59b46a186745f6`.

Capture image (historical provenance only): `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix`.
The directory name identifies the serving capture; the current runtime above
is public and does not require access to the original cluster.

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
