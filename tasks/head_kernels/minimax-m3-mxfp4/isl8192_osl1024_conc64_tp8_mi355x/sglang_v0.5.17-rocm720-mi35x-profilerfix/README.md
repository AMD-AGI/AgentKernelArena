# MiniMax-M3-MXFP4 workload

ISL **8192**, OSL **1024**, concurrency **64**, tensor parallel world size **8**.
Each leaf is an isolated operator replay from one rank on MI355X (`gfx950`).

Capture image: `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix`.

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
