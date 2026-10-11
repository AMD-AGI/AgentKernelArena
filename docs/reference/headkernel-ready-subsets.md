# Published head-kernel subsets and remaining qualification

Status as of **2026-10-09 13:12 UTC**: **23 active task mappings across five models, five published scoped ready entries, and 18 mappings without a published ready entry**. The five models are DeepSeek V4 Pro, GLM 5.3 Flash, Kimi K3, MiniMax M3, and Qwen3.8 2.4T.

The active inventory comprises 18 refreshed mappings and five Qwen mappings. Historical catalogs and evidence snapshots retain their original scope and dates; their counts and provisional flags do not establish current readiness. The five entries below qualify only their pinned task definitions, inputs, and checked protocols. **Whole-suite readiness, complete model-workload qualification, and serving/E2E gains are not established.**

## Five published entries

Follow the guide at its pinned commit to obtain the one-task config, fixture materialization instructions, qualification receipts, and mandatory post-evaluation checks.

The refreshed tasks and their configs are published on the linked ready branches. Check out the pinned commit from the selected guide before following its commands. This restored-upstream branch does not contain every refreshed task path listed below.

| Task | Pinned guide | Qualified scope and limits |
| --- | --- | --- |
| `headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8` | [DeepSeek FP8 quantization · `23cb3067`](https://github.com/AMD-AGI/AgentKernelArena/blob/23cb3067ab81f03264ec6027b97be17156f291a9/ready/ds-quant/README.md) | Three shapes: 8192×1536, 8192×2048, and 8192×7168. Workload counts cover sampled prefill steps; no full-workload extrapolation. |
| `headkernel/deepseek-v4-pro__unified_paged_attention_prefill` | [DeepSeek MLA prefill · `55c8f713`](https://github.com/AMD-AGI/AgentKernelArena/blob/55c8f713b06b206e0ae1d389e6726e65140078c0/ready/mla-prefill/README.md) | Two M8192/H16 cases. Legacy M1/M64 coverage is outside this publication. |
| `headkernel/kimi-k3__lean_attention_decode` | [Kimi Lean decode · `c39e94c8`](https://github.com/AMD-AGI/AgentKernelArena/blob/c39e94c8e148a171173eb276a9e495ab821c1649/ready/kimi-lean/README.md) | One batch64 structural case with a paired sampled sequence-length distribution. Timing is not exhaustive across all 1,024 admitted lengths. |
| `headkernel/minimax-m3__gemm_afp4wfp4_kernel` | [MiniMax FP4 GEMM · `ef7474dd`](https://github.com/AMD-AGI/AgentKernelArena/blob/ef7474dd9e29e74428ca09b4c488832db1fd220e/ready/minimax-fp4/README.md) | All 14 captured cases. The guide and approval disclose accepted limits in preserved framework operational receipts. Other MiniMax heads are outside scope. |
| `headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm` | [Qwen fused add + RMSNorm · `f501ef65`](https://github.com/AMD-AGI/AgentKernelArena/blob/f501ef65403bdf76a122918c405f6494bc6eab38/ready/qwen-rmsnorm/README.md) | BF16 shapes 8192×8192 and 64×8192. Complete trusted numerical evidence and fresh framework PASS are linked through an explicit protection-only compatibility review. Shapes originate in the documented SGLang 0.5.18 capture; source/runtime qualification uses 0.5.20, without a new full-serving-capture claim. |

These are unchanged-source qualification controls. Their measured ratios are not accepted optimization gains. A changed candidate requires its own complete correctness, timing-quality, source-control, and review evidence under the guide's protocol.

## Runtime

The published entries use Linux amd64 and AMD MI355X (`gfx950`) with **SGLang 0.5.20**. The public tag is [`lmsysorg/sglang:v0.5.20-rocm724-mi35x`](https://hub.docker.com/r/lmsysorg/sglang/tags?name=v0.5.20-rocm724-mi35x). Reproduce with the exact tested digest:

```text
docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
```

Each guide specifies its external fixtures and runtime setup. Selecting a task or downloading this image does not qualify another task. Isolated replay uses the task's rank-local inputs; it does not require a model server or full-model checkpoint.

## Remaining active mappings

Every task below remains unpublished as a ready entry. Names in the second column follow `headkernel/<model>__<task>`; the model column supplies the exact prefix. CPU preparation, targeted diagnostics, and partial GPU phases do not substitute for complete qualification.

| Model prefix | Task | Current remaining gate |
| --- | --- | --- |
| `deepseek-v4-pro` | `unified_paged_attention_decode` | Complete framework and six-phase trusted qualification of the expanded four-case contract, including nine observed KV widths. |
| `deepseek-v4-pro` | `moe_stage1_grouped_gemm_silu_flydsl` | Complete current qualification of the expanded observed-distribution contract. Older fixed-case checks do not cover it. |
| `deepseek-v4-pro` | `moe_stage1_grouped_gemm_silu_opus_a8w4` | Complete current distribution qualification; an older performance attempt did not finish. |
| `deepseek-v4-pro` | `moe_stage2_down_proj_reduce_opus_a8w4` | Prior run stopped during reference correctness after 42/375 settings. Full comparison, framework qualification, and evidence readback remain outstanding. |
| `glm-5.3-flash` | `ck_gemm_a8w8_blockscale_bpreshuffle` | New publication review rejected the incomplete 5/6-phase comparison. Candidate performance lacks 28 matched series / 2,800 samples and final completion evidence. |
| `glm-5.3-flash` | `gemm_a16w16_bf16_cijk` | No proposed fixed native MFMA arithmetic rule passes the required gate. Acceptance repair and full qualification remain unresolved. |
| `glm-5.3-flash` | `fused_moe_kernel` | Original native graph-versus-eager failure remains unresolved. A later diagnostic did not reproduce it and did not qualify the task. |
| `kimi-k3` | `dense_bf16_gemm_cijk` | Full 40-case native, trusted, and framework qualification. A targeted stream diagnostic covers only its recorded case. |
| `kimi-k3` | `attn_residual_aggregate_hip` | Complete 69-case qualification. Interrupted performance observations reached 26/69 trusted cases and 13/69 framework cases. |
| `kimi-k3` | `moe_gemm1_stage1` | Fresh qualification of the current routing and paired-work contract; older routing snapshots are not current qualification. |
| `kimi-k3` | `moe_gemm2_stage2` | Fresh qualification of the current routing and paired-work contract; older routing snapshots are not current qualification. |
| `minimax-m3` | `decode_score_kernel` | Complete current framework and trusted qualification of the 16-case paired contract. |
| `minimax-m3` | `gqa_share_sparse_decode_kernel` | Complete current framework and trusted qualification of the 24-case paired contract. |
| `minimax-m3` | `gqa_share_sparse_fwd_kernel` | Complete current framework and trusted qualification of the 40-case paired contract. |
| `qwen3.8-2.4t` | `dense_bf16_gemm_cluster` | Source refresh is recorded; current SG0.5.20 runtime and framework qualification remain pending. |
| `qwen3.8-2.4t` | `fused_moe_2stage_mxfp4` | Source refresh is recorded; current SG0.5.20 runtime and framework qualification remain pending. |
| `qwen3.8-2.4t` | `fused_recurrent_gated_delta_rule_decode` | Source refresh is recorded; current SG0.5.20 runtime and framework qualification remain pending. |
| `qwen3.8-2.4t` | `paged_attention_decode` | Source refresh is recorded; current SG0.5.20 runtime and framework qualification remain pending. |

Two configured historical packages are outside the 23 active mappings: DeepSeek `dsa_sparse_mla_attn` is superseded in this recipe by unified paged prefill/decode, and GLM `elementwise_copy_cluster` recorded aliases rather than materialized GPU copies. Full original served MoE payload capture and broader coverage beyond the published scopes also remain separate outstanding work.
