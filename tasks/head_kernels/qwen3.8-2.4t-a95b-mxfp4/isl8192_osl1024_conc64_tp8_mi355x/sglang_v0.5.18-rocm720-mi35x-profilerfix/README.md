# Qwen3.8-2.4T-A95B-MXFP4 workload

ISL **8192**, OSL **1024**, concurrency **64**, tensor parallel world size **8**.
Each leaf is an isolated operator replay from one rank on MI355X (`gfx950`).

Capture image: `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix`.

[workload.json](workload.json) records the exact configuration, task selectors,
and per-kernel runtime requirements. Each leaf carries its own source, frozen
contract, configuration, runners, environment preflight, and shape declaration.

| Kernel | Exact shapes | Task selector |
| --- | --- | --- |
| [dense_bf16_gemm_cluster](dense_bf16_gemm_cluster/README.md) | [SHAPES.md](dense_bf16_gemm_cluster/SHAPES.md) | `head_kernels/qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/dense_bf16_gemm_cluster` |
| [fused_moe_2stage_mxfp4](fused_moe_2stage_mxfp4/README.md) | [SHAPES.md](fused_moe_2stage_mxfp4/SHAPES.md) | `head_kernels/qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/fused_moe_2stage_mxfp4` |
| [fused_recurrent_gated_delta_rule_decode](fused_recurrent_gated_delta_rule_decode/README.md) | [SHAPES.md](fused_recurrent_gated_delta_rule_decode/SHAPES.md) | `head_kernels/qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/fused_recurrent_gated_delta_rule_decode` |
| [gemma_fused_add_rmsnorm](gemma_fused_add_rmsnorm/README.md) | [SHAPES.md](gemma_fused_add_rmsnorm/SHAPES.md) | `head_kernels/qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/gemma_fused_add_rmsnorm` |
| [paged_attention_decode](paged_attention_decode/README.md) | [SHAPES.md](paged_attention_decode/SHAPES.md) | `head_kernels/qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/paged_attention_decode` |

The two-stage MXFP4 MoE leaf is one captured composite stage-1 + stage-2
seam. Its profile rows do not establish independent stage oracles.

All tasks require fresh matching-image GPU task-validator qualification.
The metadata and CPU layout checks do not establish GPU correctness or speedup.
