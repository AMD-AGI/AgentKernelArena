# Kimi-K3 workload

ISL **8192**, OSL **1024**, concurrency **64**, tensor parallel world size **8**.
Each leaf is an isolated operator replay from one rank on MI355X (`gfx950`).

Current public runtime: `docker.io/rocm/hyperloom@sha256:1f5464829559b086eb66f9b803cb9c7a817438c43edff2d5ef59b46a186745f6`.

Capture image (historical provenance only): `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830`.
The directory name identifies the serving capture; the current runtime above
is public and does not require access to the original cluster.

[workload.json](workload.json) records shared workload context, distinct historical
scenarios, task selectors, and per-kernel runtime requirements. Each leaf carries its own source, frozen
contract, configuration, runners, environment preflight, and shape declaration.

All three Kimi tasks have workload scoring disabled by their task-local
`ut/meta.json` policy: **zero enabled/scored benchmark cases**, with all **nine
former timing cases retained as unscored diagnostics** and all **11 shape records**
preserved.

ISL 8192 is the input sequence length, not a universal prefill chunk. The stage-1
[historical workload](moe_gemm1_stage1/ut/workload.json) records the 0828 cycle1
`--chunked-prefill-size 8192` flags. The stage-2
[historical workload](moe_gemm2_stage2/ut/workload.json) records
`chunked_prefill_size=16384` from its original live server log and retains both
M=16384 and M=8192 cases. These are distinct historical scenarios. Stage-1 routing
also comes second-hand from the 0817 stage-2 capture; its decode launch variant
is assumed. Attention is decode-only: its regime declares no prefill chunk,
while its analytic weight model carries 16384 with zero prefill calls.

| Kernel | Exact shapes | Task selector |
| --- | --- | --- |
| [fwd_grouped_kernel_stage1](fwd_grouped_kernel_stage1/README.md) | [SHAPES.md](fwd_grouped_kernel_stage1/SHAPES.md) | `head_kernels/kimi-k3/isl8192_osl1024_conc64_tp8_mi355x/sglang-rocm-k3_rocm720-mi35x-k3-20260727-tl312-08011830/fwd_grouped_kernel_stage1` |
| [moe_gemm1_stage1](moe_gemm1_stage1/README.md) | [SHAPES.md](moe_gemm1_stage1/SHAPES.md) | `head_kernels/kimi-k3/isl8192_osl1024_conc64_tp8_mi355x/sglang-rocm-k3_rocm720-mi35x-k3-20260727-tl312-08011830/moe_gemm1_stage1` |
| [moe_gemm2_stage2](moe_gemm2_stage2/README.md) | [SHAPES.md](moe_gemm2_stage2/SHAPES.md) | `head_kernels/kimi-k3/isl8192_osl1024_conc64_tp8_mi355x/sglang-rocm-k3_rocm720-mi35x-k3-20260727-tl312-08011830/moe_gemm2_stage2` |

The 2026-09-20 public-runtime MI355X run passed stage-1 environment preflight but
failed native correctness at `prefill_M8192:single:0` against the independent
reference. Single-launch correctness remains unresolved; see the stage-1
[shape guide](moe_gemm1_stage1/SHAPES.md) for the recorded evidence.

All tasks require fresh matching-image GPU task-validator qualification.
The metadata and CPU layout checks do not establish GPU correctness or speedup.
