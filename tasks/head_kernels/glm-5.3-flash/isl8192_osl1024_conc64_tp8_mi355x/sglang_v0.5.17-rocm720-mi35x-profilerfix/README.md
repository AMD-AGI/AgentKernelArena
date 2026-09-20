# GLM-5.3-Flash workload

ISL **8192**, OSL **1024**, concurrency **64**, tensor parallel world size **8**.
Each leaf is an isolated operator replay from one rank on MI355X (`gfx950`).

Capture image: `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix`.

[workload.json](workload.json) records the exact configuration, task selectors,
and per-kernel runtime requirements. Each leaf carries its own source, frozen
contract, configuration, runners, environment preflight, and shape declaration.

| Kernel | Exact shapes | Task selector |
| --- | --- | --- |
| [ck_gemm_a8w8_blockscale_bpreshuffle](ck_gemm_a8w8_blockscale_bpreshuffle/README.md) | [SHAPES.md](ck_gemm_a8w8_blockscale_bpreshuffle/SHAPES.md) | `head_kernels/glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/ck_gemm_a8w8_blockscale_bpreshuffle` |
| [gemm_a16w16_bf16_cijk](gemm_a16w16_bf16_cijk/README.md) | [SHAPES.md](gemm_a16w16_bf16_cijk/SHAPES.md) | `head_kernels/glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/gemm_a16w16_bf16_cijk` |

All tasks require fresh matching-image GPU task-validator qualification.
The metadata and CPU layout checks do not establish GPU correctness or speedup.
