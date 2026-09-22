# Reported five-model results

These are **user-supplied historical results**, preserved as reported. They are
not new GPU verification of the restored branch or a new framework
`task_validator` PASS. The [machine-readable record](../../tools/headkernel-reported-results.json)
contains the same claims and task mappings.

All five rows report **`sglang:v0.5.17`, ISL 8192, OSL 1024, CONC 64, TP 8**.
The E2E gains describe the reported model-serving results; GPU-time shares,
kernel speedups and roofline percentages describe the reported kernel results.
These different quantities are not recomputed from one another. The next steps
and ETA remain reported plans, not claims that the work has finished.

| Model | Configuration | Kernel End-to-End Gain | HeadKernel Gain | Next step |
| --- | --- | --- | --- | --- |
| MiniMax M3 | `sglang:v0.5.17`<br>ISL/OSL/CONC/TP: 8k/1k/64/8 | **+4.52%** | Top1 kernel: decode_score_kernel with 11% GPU time, ~1.58× speedup and roofline 60%→86%. Top3 kernel: gqa_share_sparse_fwd_kernel with 7.63% GPU time, ~1.11× speedup and roofline 78%→86%. | Continue improve other head kernel roofline utilization |
| Kimi K3 | `sglang:v0.5.17`<br>ISL/OSL/CONC/TP: 8k/1k/64/8 | **+11.09%** | Top1 kernel: fwd_grouped_kernel_stage with 10% GPU time, ~10× speedup and roofline 20%→61%. | Continue improve head kernel roofline utilization |
| DeepSeek V4 Pro | `sglang:v0.5.17`<br>ISL/OSL/CONC/TP: 8k/1k/64/8 | **+131%** | Top1 kernel: Flash MLA with 44% GPU time, ~8.6× speedup and roofline 18%→51%. | Continue improve head kernel roofline utilization |
| GLM5.3 Flash | `sglang:v0.5.17`<br>ISL/OSL/CONC/TP: 8k/1k/64/8 | **+1.4%** | Top1 kernel: fused moe with 24% GPU time, already has 98% empirical roofline utilization. Top2&3 kernel: ~12% GPU time, GEMM tune optimization. | Test on AgentX config ETA 9/22 |
| Qwen3.8 2.4T | `sglang:v0.5.17`<br>ISL/OSL/CONC/TP: 8k/1k/64/8 | **+2.4%** | Top1 kernel: MoE1+ MoE2 with 33% GPU time, ~1.54× speedup and roofline 73%→82%. | Optimize other top kernels in workload |

The reported labels map to the current flat task paths as follows:

- MiniMax: [decode score](../../tasks/headkernel/minimax-m3__decode_score_kernel/)
  and the reported [sparse prefill](../../tasks/headkernel/minimax-m3__gqa_share_sparse_fwd_kernel/).
  **The 7.63% mapping is unresolved:** the user labels it prefill, while
  [manifest row MM-3](../../tools/manifest.json) assigns 7.63% and roofline
  78% → 86% to [sparse decode](../../tasks/headkernel/minimax-m3__gqa_share_sparse_decode_kernel/).
  The table preserves the user's label and values without silently reassigning them.
- Kimi: [grouped attention stage 1](../../tasks/headkernel/kimi-k3__fwd_grouped_kernel_stage1/).
- DeepSeek: the reported “FlashMLA” label maps to
  [DSA sparse MLA attention](../../tasks/headkernel/deepseek-v4-pro__dsa_sparse_mla_attn/),
  whose actual restored implementation is ROCm TileLang DSA. This mapping does
  not identify a CUDA FlashMLA implementation in the task.
- GLM: [fused MoE](../../tasks/headkernel/glm-5.3-flash__fused_moe_kernel/),
  [BF16 GEMM](../../tasks/headkernel/glm-5.3-flash__gemm_a16w16_bf16_cijk/), and
  [FP8 GEMM](../../tasks/headkernel/glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle/).
  The two original GEMM entries remain **`NOT_BUILT` placeholders**, not currently
  verified tasks. They are linked as the reported tuning targets.
- Qwen: MoE1 + MoE2 map to the single
  [composite two-stage MoE task](../../tasks/headkernel/qwen3.8-2.4t__fused_moe_2stage_mxfp4/).

The common reported `sglang:v0.5.17` label does not replace current execution
metadata. Restored task declarations span SGLang v0.5.17, v0.5.18 and a custom
Kimi image; public runtime candidates have separate identities. See the
[runtime guide](../how-to/headkernel-upstream-runtime.md) and
[public runtime record](../../tools/headkernel-public-runtimes.json). No equality
between those images and the reported serving environment is established here.

This results record does not change task sources, harnesses, configurations,
correctness requirements, or benchmark controls. Source restoration, historical
results, and fresh GPU or framework validation remain separate evidence.
