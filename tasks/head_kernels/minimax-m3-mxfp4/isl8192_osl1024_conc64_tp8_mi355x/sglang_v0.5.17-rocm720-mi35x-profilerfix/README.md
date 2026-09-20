# MiniMax-M3-MXFP4 workload

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
| [decode_score_kernel](decode_score_kernel/README.md) | [SHAPES.md](decode_score_kernel/SHAPES.md) | `head_kernels/minimax-m3-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/decode_score_kernel` |
| [gqa_share_sparse_decode_kernel](gqa_share_sparse_decode_kernel/README.md) | [SHAPES.md](gqa_share_sparse_decode_kernel/SHAPES.md) | `head_kernels/minimax-m3-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/gqa_share_sparse_decode_kernel` |
| [gqa_share_sparse_fwd_kernel](gqa_share_sparse_fwd_kernel/README.md) | [SHAPES.md](gqa_share_sparse_fwd_kernel/SHAPES.md) | `head_kernels/minimax-m3-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/gqa_share_sparse_fwd_kernel` |

All tasks require fresh matching-image GPU task-validator qualification.
The metadata and CPU layout checks do not establish GPU correctness or speedup.
