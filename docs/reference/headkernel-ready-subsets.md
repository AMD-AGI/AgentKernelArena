# Consolidated head-kernel tasks and qualification evidence

Status for **2026-10-12**: `head-kernels-top5` contains all **23 SG0.5.20 source task mappings** across DeepSeek V4 Pro, GLM 5.3 Flash, Kimi K3, MiniMax M3 and Qwen3.8 2.4T. Five subsets retain historical qualification at their original commits. The completed October 11 qualifications below have fresh trusted and framework evidence, with explicit revision limits. **The complete 23-task branch is not yet qualified.** Complete model-workload qualification and serving/E2E gains remain unestablished.

Use [HEADKERNELS.md](../../HEADKERNELS.md) for the consolidation sources and [the unified validator config](../../example_configs/headkernel_sg520_current_mi355x.yaml) for task selection. All task paths and five preserved guides are in this branch. The [runtime catalog](../../tools/headkernel-runtime-targets.json) distinguishes current source/runtime selection, historical capture provenance and pending qualification.

## Completed October 11 qualifications

These receipts retain the actual GPU execution revision, `7bc14698e1044931a3c893b3dbabc9676b0e57e6`. Their guides identify separately reviewed compatible revisions and the exact input sets each verifier accepts. A later branch tip needs its own compatibility assessment; copying a receipt does not qualify changed code.

| Task | Guide and verified evidence | Scope |
| --- | --- | --- |
| `headkernel/deepseek-v4-pro__unified_paged_attention_prefill` | [Current MLA prefill handoff](../../ready/current/mla-prefill/README.md): six trusted phases, all 12 framework checks, 400 trusted and 200 framework samples; OCI archive fully read back. | Both published M8192/H16 cases. The guide links explicit `0df9540d` compatibility; its 505-input verifier remains bound to the qualified snapshot. |
| `headkernel/minimax-m3__gemm_afp4wfp4_kernel` | [Current MiniMax FP4 handoff](../../ready/current/minimax-fp4/README.md): six trusted phases, all 12 framework checks, 4,200 samples and 168 controls; OCI archive fully read back. | All 14 cases. Final report and marker were recovered byte-for-byte against hashes saved before preemption. The verifier accepts the qualified inputs and separately reviewed `5854b2ac` inputs. |

Both are unchanged-source controls with **no accepted optimization gain**. The archive guides disclose the original evidence and any recovery or transport limits. Other completed individual phases, framework-only passes and interrupted runs remain partial qualification evidence.

## Additional completed framework evidence

Qwen paged attention completed a fresh **framework-only** run at `8956e210472abff5fc9d133fbe4ccd2c65064f79` on job 208015. It passed all 12 checks, its one case and 100 ordered samples, including 118 checked invocations. The independent audit verified all 37 final workspace input observations, the report/marker/integrity chain and external attempt records, durable mirrors, and owned-container retirement before `FINAL` at **2026-10-11 17:29:54 UTC**.

The [immutable publication receipt](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__paged_attention_decode/8956e210-framework-208015/ARCHIVE-VERIFIED.json) records full readback of the archive and metadata. The archive preserves 158 payload files, two contained source aliases and its manifest. The original SG0.5.18 TP8/EP8, page-1 fixture provenance remains unchanged; the tensor payload stays at its existing pinned OCI object.

The separate attention and recurrent trusted attempts failed before compilation: staging retained the exact Git bytes but removed required executable permissions from two attention UT helpers and one recurrent helper. Neither contributes a completed trusted phase or full task qualification. The contemporaneous RMSNorm run retained 200 numerical samples and an input-integrity PASS, but its official framework result was FAIL with the normalization error `checks must be a mapping`. The original draft type was not retained. These records come from the now-closed allocation 208015. The outcomes remain separate; no new complete-kernel qualification or accepted gain is claimed.

## Five historical ready subsets

The local guides below preserve the original commands, input pins, raw-evidence references and scope limits. Their immutable published versions are included for reproducing those earlier results. Their `verify.py` and `INPUT-PINS.json` files are deliberately unchanged: they check their original checkout and are expected to reject this merged checkout where shared pins differ. **Do not use a historical verification script or receipt as evidence that the current merged framework passes.** Current qualification requires newly bound checks and GPU reports; no old pin is silently refreshed.

At the consolidation base, all five task trees matched their historical qualified trees. Shared framework files differ, including scoring, guards, report handling and device binding. Qwen's pinned timing-quality module also differs. Task-tree equality is not runtime compatibility approval.

| Task | Guide in this branch | Historical published snapshot | Recorded scope |
| --- | --- | --- | --- |
| `headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8` | [DeepSeek quantization](../../ready/ds-quant/README.md) | [`23cb3067`](https://github.com/AMD-AGI/AgentKernelArena/blob/23cb3067ab81f03264ec6027b97be17156f291a9/ready/ds-quant/README.md) | Shapes 8192×1536, 8192×2048 and 8192×7168. Counts cover sampled prefill steps only. |
| `headkernel/deepseek-v4-pro__unified_paged_attention_prefill` | [DeepSeek MLA prefill](../../ready/mla-prefill/README.md) | [`55c8f713`](https://github.com/AMD-AGI/AgentKernelArena/blob/55c8f713b06b206e0ae1d389e6726e65140078c0/ready/mla-prefill/README.md) | Two M8192/H16 cases; legacy M1/M64 coverage is outside this publication. |
| `headkernel/kimi-k3__lean_attention_decode` | [Kimi Lean decode](../../ready/kimi-lean/README.md) | [`c39e94c8`](https://github.com/AMD-AGI/AgentKernelArena/blob/c39e94c8e148a171173eb276a9e495ab821c1649/ready/kimi-lean/README.md) | One batch 64 structural case with a paired sampled length distribution; not exhaustive timing of all 1,024 admitted lengths. |
| `headkernel/minimax-m3__gemm_afp4wfp4_kernel` | [MiniMax FP4 GEMM](../../ready/minimax-fp4/README.md) | [`ef7474dd`](https://github.com/AMD-AGI/AgentKernelArena/blob/ef7474dd9e29e74428ca09b4c488832db1fd220e/ready/minimax-fp4/README.md) | All 14 captured cases, with disclosed and independently accepted historical framework receipt limits. |
| `headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm` | [Qwen fused add + RMSNorm](../../ready/qwen-rmsnorm/README.md) | [`f501ef65`](https://github.com/AMD-AGI/AgentKernelArena/blob/f501ef65403bdf76a122918c405f6494bc6eab38/ready/qwen-rmsnorm/README.md) | BF16 8192×8192 and 64×8192. Historical manual trusted-route numerical evidence and protected framework PASS are connected by an explicit compatibility review. This does not qualify current ordinary scoring. |

These are unchanged-source controls, not accepted optimization gains. All five Qwen timing-quality adapters are integrated. Complete source-bound qualification remains outstanding for the Qwen set; the framework-only results above retain their separate scope. A changed candidate requires its own complete correctness, source-control, timing-quality and review evidence.

## Current runtime and input preparation

All 23 source tasks target Linux amd64 and MI355X (`gfx950`) with the following **SGLang 0.5.20** image. The public tag is [`lmsysorg/sglang:v0.5.20-rocm724-mi35x`](https://hub.docker.com/r/lmsysorg/sglang/tags?name=v0.5.20-rocm724-mi35x); use the digest from the catalog for reproduction:

```text
docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
```

The [portable setup guide](../how-to/headkernel-upstream-runtime.md) covers unified preparation of all 23 tasks. Each catalog row declares a protocol: `trusted_task_v1` retains the 18 manifest/native routes, while `headkernel_artifacts_v1` verifies Qwen's original whole-object fixtures or explicit fixture-free registration. The helper does not interpret legacy Qwen data as an `EXTERNAL-MANIFEST` or claim a new capture or qualification.

Qwen source/runtime refresh does not replace its SG0.5.18 shape and fixture provenance. That capture used TP8/EP8, page 1 attention and FP32 model-default Mamba. The later retained measured metadata record 103483 instead describes TP8/EP1, context 11264, page 16 attention and BF16 Mamba, without a complete image/raw-log record. A new complete SG0.5.20 Qwen capture, matching actual dispatch and tensor contracts, remains pending. Do not equate it with the non-Qwen context 9218 recipe or infer current coverage from older shapes.

## Workload archives in OCI

The immutable [workload README](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261009/index-v2/README.md) and [workload INDEX.json](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261009/index-v2/INDEX.json) select **six packages for DeepSeek, Kimi and MiniMax**, supporting the four historical subsets from those three models. They contain **25,583 payload files / 35,277,287,593 bytes**, with source inventories, full publication readback records, exact checkpoint/public-model pins, launch/source/setup files, sampled traces where captured and representative tensors. They do not supply a full Qwen SG0.5.20 capture or qualify GLM.

These links are rclone OCI object-store URIs. Fetch the small versioned index first, then use its exact per-package download commands; downloading the entire dated parent would also include superseded versions.

```bash
export GOMAXPROCS=1
export RCLONE_CONFIG=/path/to/your/rclone.conf
WORKLOAD_INDEX=oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261009/index-v2
rclone copy "$WORKLOAD_INDEX" /path/to/local-nvme/workload-index \
  --transfers 64000 --progress --buffer-size 0
```

| Index object | SHA-256 |
| --- | --- |
| `INDEX.json` | `01e4b4efdcc4ee0ab6fafa8eb56ece727e915ca79ef112c1bb1e787998330dd6` |
| `PACKAGE-DESCRIPTOR.json` | `8bccc0de617d9a9494caa220aa7bbf0cd13f3c98d59b5a98addfe557d28f57f6` |

The recovered supplements preserve 690 files from the 24 formerly unexamined historical rank directories. Producer failures and partial outcomes remain unchanged. Primary MiniMax 194969 collected no Torch profile; the 16 sampled traces from 194299 and 15 traces from 194550 remain separate, with 194550 TP0 DECODE still missing. The historical analyzer sources `intake.py` and `quant_inventory.py` remain unavailable and disclosed. Full model weights are external at verified immutable public revisions; package-specific guides describe retrieval without requiring the original cluster or NFS paths.

## Remaining gates for the other historical subsets

The two completed October 11 handoffs above do not discharge the remaining gates for the other three historical subsets. Their original published receipts remain unchanged.

| Task | Remaining gate |
| --- | --- |
| DeepSeek `per_group_quant_fp8` | Preserve the earlier timing rejection and the later native-event SIGSEGV as separate attempts. Input-freeze and graph/event lifetime repairs are integrated; their full current qualification remains pending. |
| Kimi `lean_attention_decode` | Complete current framework/trusted validation and applicable runtime compatibility review, preserving the historical sampled-length scope. |
| Qwen `gemma_fused_add_rmsnorm` | A closed-guard framework run at `31d7bdde` passed 12 checks and 200 samples but had a final input-observation gap. The later `8956e210` run failed official report formatting despite retained numerical samples and an input-integrity PASS. A valid complete framework result remains required; retained `294e65a0` numerical evidence keeps its explicit compatibility scope. |

## Eighteen tasks without historical ready-subset receipts

These 18 tasks do not have one of the five historical ready-subset receipts. Their complete qualification remains pending. Names in the second column follow `headkernel/<model>__<task>`; the model column supplies the exact prefix. CPU preparation, targeted diagnostics, framework-only passes and partial GPU phases keep their stated scopes. Current framework completion also requires the [post-validator integrity chain](../how-to/task-validator.md); a prior PASS does not silently establish compatibility with that changed lifecycle.

| Model prefix | Task | Current remaining gate |
| --- | --- | --- |
| `deepseek-v4-pro` | `unified_paged_attention_decode` | Complete framework and six-phase trusted qualification of the expanded four-case contract, including nine observed KV widths. |
| `deepseek-v4-pro` | `moe_stage1_grouped_gemm_silu_flydsl` | Complete current qualification of the expanded observed-distribution contract. Older fixed-case checks do not cover it. |
| `deepseek-v4-pro` | `moe_stage1_grouped_gemm_silu_opus_a8w4` | Complete current distribution qualification; an older performance attempt did not finish. |
| `deepseek-v4-pro` | `moe_stage2_down_proj_reduce_opus_a8w4` | Job 207739 stopped during reference correctness after 362/375 settings: 195/195 prefill and 167/180 decode. Full comparison, framework qualification and evidence readback remain outstanding. |
| `glm-5.3-flash` | `ck_gemm_a8w8_blockscale_bpreshuffle` | Native HIP graph-event timing is integrated and the full reference correctness gate passed. Complete trusted/native comparison and framework qualification were interrupted and remain pending. |
| `glm-5.3-flash` | `gemm_a16w16_bf16_cijk` | Native partial-reference acceptance was repaired and all 20 reference correctness cases passed. Complete trusted/native timing and framework qualification remain pending after preemption. |
| `glm-5.3-flash` | `fused_moe_kernel` | Original native graph-versus-eager failure remains unresolved. A later diagnostic did not reproduce it and did not qualify the task. |
| `kimi-k3` | `dense_bf16_gemm_cijk` | Full 40-case native, trusted, and framework qualification. A targeted stream diagnostic covers only its recorded case. |
| `kimi-k3` | `attn_residual_aggregate_hip` | Complete 69-case qualification using the integrated complete-case route. Job 207739 preserved reference compile/correctness for all 69 cases; its last performance snapshot recorded 18/69 trusted and 11/69 framework cases before cancellation. |
| `kimi-k3` | `moe_gemm1_stage1` | Fresh qualification of the current routing and paired-work contract; older routing snapshots are not current qualification. |
| `kimi-k3` | `moe_gemm2_stage2` | Fresh qualification of the current routing and paired-work contract; older routing snapshots are not current qualification. |
| `minimax-m3` | `decode_score_kernel` | Complete current framework and trusted qualification of the 16-case paired contract through the integrated complete-case route; interrupted samples are not a completed union. |
| `minimax-m3` | `gqa_share_sparse_decode_kernel` | Complete current framework and trusted qualification of the 24-case paired contract through the integrated complete-case route. |
| `minimax-m3` | `gqa_share_sparse_fwd_kernel` | Complete current framework and trusted qualification of the 40-case paired contract through the integrated complete-case route. The earlier timeout replacement was prepared but not launched before external preemption. |
| `qwen3.8-2.4t` | `dense_bf16_gemm_cluster` | A five-case framework pass at `8cdd7b0d` and its 500 samples are archived. The later trusted run passed reference compile/correctness but was interrupted during performance. Complete trusted and applicable current framework-integrity gates remain pending. |
| `qwen3.8-2.4t` | `fused_moe_2stage_mxfp4` | Native enum identity across relocated verified caches was repaired. Full trusted comparison and framework finalization remain pending; raw GPU phases alone do not qualify it. |
| `qwen3.8-2.4t` | `fused_recurrent_gated_delta_rule_decode` | Authoritative source-bound admission is integrated. The `8956e210` trusted attempt stopped before compilation on a UT helper's executable-permission check, with unchanged Git bytes. Complete trusted and framework qualification remain pending. |
| `qwen3.8-2.4t` | `paged_attention_decode` | The `8956e210` framework-only pass, 100 samples, 37 input observations and full integrity/retirement evidence are archived with verified readback. The trusted attempt stopped on executable-permission checks before compilation; a complete trusted comparison and full-task review remain pending. Earlier `0df9540d` framework evidence remains separate. |

Two configured historical packages are outside the 23 active mappings: DeepSeek `dsa_sparse_mla_attn` is superseded in this recipe by unified paged prefill/decode, and GLM `elementwise_copy_cluster` recorded aliases rather than materialized GPU copies. Full original served MoE payload capture and broader coverage beyond the published scopes also remain separate outstanding work.
