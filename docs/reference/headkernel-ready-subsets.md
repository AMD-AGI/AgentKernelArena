# Consolidated head-kernel tasks and qualification evidence

Status for **2026-10-12**: `head-kernels-top5` contains all **23 SG0.5.20 source task mappings** across DeepSeek V4 Pro, GLM 5.3 Flash, Kimi K3, MiniMax M3 and Qwen3.8 2.4T. Five subsets retain historical qualification at their original commits. **Seven tasks have completed scoped qualifications; 16 remain unfinished.** The completed qualifications below retain their actual trusted and framework execution identities, with explicit revision limits. **The complete 23-task branch is not yet qualified.** Complete model-workload qualification and serving/E2E gains remain unestablished.

Use [HEADKERNELS.md](../../HEADKERNELS.md) for the consolidation sources and [the unified validator config](../../example_configs/headkernel_sg520_current_mi355x.yaml) for task selection. All task paths and five preserved guides are in this branch. The [runtime catalog](../../tools/headkernel-runtime-targets.json) distinguishes current source/runtime selection, historical capture provenance and pending qualification. Its task readiness flags apply only to the linked exact-commit scoped approvals; global readiness remains false for the complete 23-task branch.

## Completed October 11 qualifications

These receipts retain the actual GPU execution revision, `7bc14698e1044931a3c893b3dbabc9676b0e57e6`. Their guides identify separately reviewed compatible revisions and the exact input sets each verifier accepts. A later branch tip needs its own compatibility assessment; copying a receipt does not qualify changed code.

| Task | Guide and verified evidence | Scope |
| --- | --- | --- |
| `headkernel/deepseek-v4-pro__unified_paged_attention_prefill` | [Current MLA prefill handoff](../../ready/current/mla-prefill/README.md): six trusted phases, all 12 framework checks, 400 trusted and 200 framework samples; OCI archive fully read back. | Both published M8192/H16 cases. The guide links explicit `0df9540d` compatibility; its 505-input verifier remains bound to the qualified snapshot. |
| `headkernel/minimax-m3__gemm_afp4wfp4_kernel` | [Current MiniMax FP4 handoff](../../ready/current/minimax-fp4/README.md): six trusted phases, all 12 framework checks, 4,200 samples and 168 controls; OCI archive fully read back. | All 14 cases. Final report and marker were recovered byte-for-byte against hashes saved before preemption. The verifier accepts the qualified inputs and separately reviewed `5854b2ac` inputs. |

The [explicit `18bb4d67` compatibility review](mla-fp4-compatibility-18bb4d67.json) approves reuse of both completed qualifications for their unchanged published scopes at `18bb4d67bea577f6951f51a95cf524c9c425f952`. It records unchanged task and measurement code and confirms that the preserved report/marker pairs pass the current read-only legacy archive completion gate. The GPU reports remain labeled `7bc14698`; this approval claims **no new GPU execution or observed post-validator lifecycle at `18bb4d67`**. It is separate from the strict snapshot verifiers: their accepted pins and all old reports remain unchanged. New live runs still require current post-run integrity, and changed candidates, new shapes, group execution, model/serving qualification and later branch tips require their own evidence or review.

Both are unchanged-source controls with **no accepted optimization gain**. The archive guides disclose the original evidence and any recovery or transport limits. Other completed individual phases, framework-only passes and interrupted runs remain partial qualification evidence.

## Completed October 12 RMSNorm qualification

[Current Qwen fused add + RMSNorm](../../ready/current/qwen-rmsnorm/README.md) is approved for the two BF16 shapes 8192×8192 and 64×8192 at `18bb4d67`. Job 209533 passed all 12 framework checks and 200 ordered samples, with the full post-validator integrity chain, all 44 final workspace entries and owned-container retirement verified before `FINAL` at **2026-10-12 15:09:40 UTC**.

The [explicit numerical compatibility approval](../../ready/current/qwen-rmsnorm/COMPATIBILITY-18bb4d67.json) retains the six trusted phases, 400 samples and 72 compiled controls from `294e65a0`, job 203064, without relabeling that execution or claiming a new trusted comparison. The new archive and all seven remote objects passed full readback, including 300 payload files and two relative symlinks. **No accepted optimization or serving/E2E gain is claimed.** Earlier failed frameworks and historical pins remain unchanged; changed candidates require complete current evaluation.

## Completed October 12 paged-attention qualification

[Current Qwen paged-attention decode](../../ready/current/qwen-attention/README.md) is the fourth completed scoped task, approved for `decode_m64_live|decode` at `18bb4d67`. Job 209572 completed all six fresh trusted phases and 200 ordered reference/candidate samples, with per-phase input/device bindings and owned retirement before `FINAL` at **2026-10-12 15:42:10 UTC**.

The [644-check compatibility review](../../ready/current/qwen-attention/COMPATIBILITY-18bb4d67.json) accepts the complete framework execution at `8956e210`, job 208015: 12 checks, 100 samples and its full integrity chain. It claims no new framework execution at `18bb4d67`, excludes incomplete earlier trusted attempts and accepts no optimization gain. All seven new publication objects and every member of the 786-file/15-symlink payload passed full readback. Original SG0.5.18 TP8/EP8 page-1 fixtures remain pinned; no new SG0.5.20 serving capture is claimed.

## Completed October 12 recurrent-GDN qualification

[Current Qwen recurrent gated-delta decode](../../ready/current/qwen-recurrent/README.md) is the fifth completed scoped task, approved for its batch-64 and batch-1 cases at `eed4df57`. Job 209670 completed all six fresh trusted phases and 400 ordered samples, with complete input bindings and owned retirement before `FINAL` at **2026-10-12 16:49:33 UTC**.

The [1,334-check compatibility review](../../ready/current/qwen-recurrent/COMPATIBILITY-eed4df57.json) accepts the retained `18bb4d67`/209533 framework execution: all 12 checks, 200 samples and its full integrity chain. No framework execution at `eed4df57` is claimed. Earlier incomplete trusted runs contribute no phases or samples, and no gain is accepted. All seven publication objects and every member of the 680-file/15-symlink payload passed full streamed readback. Original SG0.5.18 TP8/EP8 fixture and FP32-state provenance remain unchanged; no new SG0.5.20 serving capture is claimed.

## Completed October 12 MoE qualification

[Current Qwen MXFP4 MoE](../../ready/current/qwen-moe/README.md) joins a fresh `096452d9` framework PASS on job 209857 with the completed `eed4df57` trusted comparison on job 209670. Both original captured cases passed all 12 framework checks and all six trusted phases, with 200 framework and 400 trusted samples. The replay audit, eight mirrored integrity observations and owned retirement passed before framework FINAL at **2026-10-12 19:08:01 UTC**.

The [exact compatibility review](../../ready/current/qwen-moe/COMPATIBILITY-096452d9.json) records unchanged task, timing, numerical, quality and trusted-runtime code. The trusted execution retains its original eed4 identity; no new trusted run at 096 is claimed. All eight OCI publication objects were read back, including every byte and mode of 313 archive members. No gain or new serving capture is claimed. The earlier framework WARN remains unchanged and is not promoted.

## Completed October 12 Kimi Lean qualification

[Current Kimi Lean attention decode](../../ready/current/kimi-lean/README.md) is the seventh completed scoped task, approved at `eed4df57` for the original B64, page-1 structural case and sampled sequence lengths 8193–9216. The actual framework execution on job 209907 passed all 12 checks, with **100 candidate + 100 reference samples**, 110 checked pairs and all **39 canonical post-validator input observations**, before preemption. The host finalizer then failed while importing PyYAML, and its cleanup verified owned-container retirement.

The [explicit compatibility and CPU-recovery approval](../../ready/current/kimi-lean/COMPATIBILITY-RECOVERY-eed4df57.json) preserves that host failure and binds a separate `RECOVERY-FINAL.json`; it invents no original `FINAL.json` or missing host input receipts. The complete trusted comparison remains attributed to `20c1a949`/job 200712, with compiled controls separately attributed to job 201533. All eight publication objects and 384 payload files passed full streamed readback. **No new GPU execution during recovery, trusted rerun, exhaustive length timing, accepted optimization gain or serving/E2E gain is claimed.**

## Completed October 12 Dense GEMM qualification

[Current Qwen Dense GEMM](../../ready/current/qwen-dense/README.md) joins the retained `eed4df57` framework execution on job 209670 with the complete fresh trusted execution on job 210191. All 12 framework checks, six trusted phases and five original cases passed, retaining 500 framework and 1,000 trusted samples. Both owned-container retirements preceded their FINAL records; trusted FINAL completed at **2026-10-12 23:34:02 UTC**.

The [scoped compatibility review](../../ready/current/qwen-dense/COMPATIBILITY-230de1f5.json) verifies unchanged task and relevant runtime/quality code at `230de1f5`, without relabeling the eed4 executions. Independent review replayed the full join, all six raw phase validators, 26 before/after input entries per phase and 217 source-verifier entries. All eight OCI objects and all 849 archive members passed full remote readback. Earlier interrupted comparisons contribute no phases or samples; no gain or new serving capture is claimed.

## Additional completed framework evidence

The [October 12 Qwen partial-evidence publication](qwen-partial-evidence-20261012.json) preserves two more complete individual gates from `eed4df57`, job 209670. Dense GEMM passed all 12 framework checks across five cases and 500 samples; the separate later complete trusted comparison and joined qualification are recorded above. MoE completed all six trusted phases across two cases and 400 samples; that attempt's framework remains **WARN** because `replay_validation_valid` was unset. The original report is unchanged; the separate later framework PASS and joined qualification are recorded above. Both gates include verified input integrity and owned-container retirement. All 19 OCI export objects and 279 archive members passed full download and hash verification; the linked publication pointer pins the remote readback receipt and archives. **Those partial publications alone do not establish full qualification, and neither adds an accepted gain.**

Qwen paged attention completed a fresh **framework-only** run at `8956e210472abff5fc9d133fbe4ccd2c65064f79` on job 208015. It passed all 12 checks, its one case and 100 ordered samples, including 118 checked invocations. The independent audit verified all 37 final workspace input observations, the report/marker/integrity chain and external attempt records, durable mirrors, and owned-container retirement before `FINAL` at **2026-10-11 17:29:54 UTC**.

The [immutable publication receipt](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261012/qualification-evidence/qwen3.8-2.4t__paged_attention_decode/8956e210-framework-208015/ARCHIVE-VERIFIED.json) records full readback of the archive and metadata. The archive preserves 158 payload files, two contained source aliases and its manifest. The original SG0.5.18 TP8/EP8, page-1 fixture provenance remains unchanged; the tensor payload stays at its existing pinned OCI object.

The separate attention and recurrent trusted attempts failed before compilation: staging retained the exact Git bytes but removed required executable permissions from two attention UT helpers and one recurrent helper. Neither contributes a completed trusted phase or full task qualification. The contemporaneous RMSNorm run retained 200 numerical samples and an input-integrity PASS, but its official framework result was FAIL with the normalization error `checks must be a mapping`. The original draft type was not retained. These records come from the now-closed allocation 208015. Those failed attempts remain excluded from the completed qualifications above; no gain is claimed.

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

These are unchanged-source controls, not accepted optimization gains. All five Qwen timing-quality adapters are integrated. All five Qwen task mappings now have completed scoped handoffs above; full model-workload and serving qualification remain unestablished. The framework-only results retain their separate scope. A changed candidate requires its own complete correctness, source-control, timing-quality and review evidence.

## Current runtime and input preparation

All 23 source tasks target Linux amd64 and MI355X (`gfx950`) with the following **SGLang 0.5.20** image. The public tag is [`lmsysorg/sglang:v0.5.20-rocm724-mi35x`](https://hub.docker.com/r/lmsysorg/sglang/tags?name=v0.5.20-rocm724-mi35x); use the digest from the catalog for reproduction:

```text
docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
```

The [portable setup guide](../how-to/headkernel-upstream-runtime.md) covers unified preparation of all 23 tasks. Each catalog row declares a protocol: `trusted_task_v1` retains the 18 manifest/native routes, while `headkernel_artifacts_v1` verifies Qwen's original whole-object fixtures or explicit fixture-free registration. The helper does not interpret legacy Qwen data as an `EXTERNAL-MANIFEST` or claim a new capture or qualification.

Qwen source/runtime refresh does not replace its SG0.5.18 shape and fixture provenance. That capture used TP8/EP8, page 1 attention and FP32 model-default Mamba. The later retained measured metadata record 103483 instead describes TP8/EP1, context 11264, page 16 attention and BF16 Mamba, without a complete image/raw-log record. A new complete SG0.5.20 Qwen capture, matching actual dispatch and tensor contracts, remains pending. Do not equate it with the non-Qwen context 9218 recipe or infer current coverage from older shapes.

## GLM FP8 GEMM: primary and diagnostic timings

For `headkernel/glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle`, the active [task contract](../../tasks/headkernel/glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle/config.yaml) selects `scoring_baseline.kind: native_production`. The editable [candidate scaffold](../../tasks/headkernel/glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle/source/kernels.py) is a newly authored Triton port of the observed operator ABI; it is not the original CK implementation from the image.

The performance entrypoint calls [production_comparison.py](../../tasks/headkernel/glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle/scripts/production_comparison.py) with the bound performance request. For each case it measures the pinned AITER `gemm_a8w8_blockscale_bpreshuffle` production callable and the submitted Triton port on the same captured ABI and fresh numerical challenge sequence. The nested `native_production_comparison` records those `native_production` and `candidate_port` legs as score inputs. In contrast, the outer `reference_performance.cases` series measures the frozen supplied Triton port and remains a diagnostic/secondary comparison; it is not the native-production denominator or numerator used to form the primary matched ratio.

The [native-baseline consumer](../../src/native_baseline.py) validates the request, source/image/provenance bindings, complete case coverage and raw samples before deriving the score. Its raw primary aggregate is the **arithmetic mean of the per-case `native_mean_ms / candidate_port_mean_ms` ratios**. Accepted results still require the existing source-change and timing-quality gates. A ratio above one describes an isolated operator comparison and establishes no serving/E2E gain; an aggregate can also coexist with individual case regressions. A standalone unbound call to `production_comparison.py` is diagnostic only. The protected [NATIVE-BASELINE.json](../../tasks/headkernel/glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle/provenance/NATIVE-BASELINE.json) retains legacy diagnostic-only/frozen-port descriptions alongside its source pins; those descriptions do not reflect the active bound scoring route above.

Coverage is the **14 recorded structural cases** in [cases.json](../../tasks/headkernel/glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle/cases.json), including distinct source records with equal shapes. Preserve the captured physical layouts: AITER FP8 weight preshuffle, activation scales with stride `(1, M)`, row-major block-128 weight scales and BF16 output. Fresh challenges vary numerical activations while the matched native/port pair preserves the captured weights, scales and ABI; CPU reference checks validate the outputs. These are structural cases with representative operands and fresh challenges, not a replay of every original tensor value or a new serving workload. This clarification adds no GPU result or qualification.

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

A separate [GLM workload handoff guide](oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261012/glm-5.3-flash/glm-final-capture-v4-194841/GUIDE.md) now preserves the existing `glm-final-capture-v4-194841` capture. The [branch-local publication pointer](glm-workload-handoff-20261012.json) pins the guide, per-file manifest, full destination readback and completion receipt by SHA-256. Publication verified **272 payload files / 99,133,731 bytes**, including 16 gzip traces for TP0–7 EXTEND/DECODE. This is a supplement to the immutable `index-v2`; that index still selects the same six packages from three models and its objects and hashes are unchanged.

The recorded GLM workload used 64 prompts, ISL 8192 / OSL 1024, concurrency 64, TP8, context 9218 and seed 42. Its SGLang 0.5.20 runtime included **18 backport/overlay files; stock-image equivalence is false**. Eight profiler steps were requested per stage; these sampled traces do not establish all graph replays or every live shape/scalar argument. Eight rank directories from the earlier failed `glm-served-capture-v3-194603` attempt remain inaccessible, with the readable failure/source records preserved. The selected latest launch/config/source/trace records have no access gaps. Model weights, the dataset body and operator tensor bodies remain external with pinned identities or references. **The publication adds no model/GPU run, GLM task qualification or performance/E2E gain.**

With rclone and configured OCI credentials, fetch the small metadata first, then follow the guide for the bundle; no original Crusoe, Spur or node session is required. Reuse the transfer environment above:

```bash
GLM_WORKLOAD=oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261012/glm-5.3-flash/glm-final-capture-v4-194841
rclone copy "$GLM_WORKLOAD" glm-workload \
  --include /GUIDE.md --include /MANIFEST.json \
  --include /PUBLICATION-READBACK.json --include /COMPLETE.json \
  --transfers 64000 --progress --buffer-size 0
```

The guide records the original server arguments and source-path mapping. Retained launch scripts contain historical paths that need remapping for reproduction; publication did not test a new portable execution.

## Remaining gates for the other historical subsets

The completed scoped handoffs above do not discharge the remaining gate for DeepSeek quantization. Kimi Lean now has the explicitly approved recovery handoff above; all original historical receipts remain unchanged.

| Task | Remaining gate |
| --- | --- |
| DeepSeek `per_group_quant_fp8` | Preserve the earlier timing rejection and the later native-event SIGSEGV as separate attempts. Input-freeze and graph/event lifetime repairs are integrated; their full current qualification remains pending. |

## Eighteen tasks without historical ready-subset receipts

These 18 tasks do not have one of the five historical ready-subset receipts. Paged attention, recurrent GDN, MoE and Dense GEMM now have completed scoped handoffs above; complete qualification remains pending for the other 14. Names in the second column follow `headkernel/<model>__<task>`; the model column supplies the exact prefix. CPU preparation, targeted diagnostics, framework-only passes and partial GPU phases keep their stated scopes. Current framework completion also requires the [post-validator integrity chain](../how-to/task-validator.md); a prior PASS does not silently establish compatibility with that changed lifecycle.

| Model prefix | Task | Current scoped status or remaining gate |
| --- | --- | --- |
| `deepseek-v4-pro` | `unified_paged_attention_decode` | Complete framework and six-phase trusted qualification of the expanded four-case contract, including nine observed KV widths. |
| `deepseek-v4-pro` | `moe_stage1_grouped_gemm_silu_flydsl` | Complete current qualification of the expanded observed-distribution contract. Older fixed-case checks do not cover it. |
| `deepseek-v4-pro` | `moe_stage1_grouped_gemm_silu_opus_a8w4` | Complete current distribution qualification; an older performance attempt did not finish. |
| `deepseek-v4-pro` | `moe_stage2_down_proj_reduce_opus_a8w4` | Job 207739 stopped during reference correctness after 362/375 settings: 195/195 prefill and 167/180 decode. Full comparison, framework qualification and evidence readback remain outstanding. |
| `glm-5.3-flash` | `ck_gemm_a8w8_blockscale_bpreshuffle` | Native HIP graph-event timing is integrated and the full reference correctness gate passed. The 209816 framework fatal-signal crash remains unresolved (SIGSEGV unconfirmed). Separate r3 diagnostic 209907 timed out at 600 seconds after 14 outer cases / 1,400 samples; native comparison remained incomplete. Complete trusted/native comparison and framework qualification remain pending. |
| `glm-5.3-flash` | `gemm_a16w16_bf16_cijk` | Native partial-reference acceptance was repaired and all 20 reference correctness cases passed. Complete trusted/native timing and framework qualification remain pending after preemption. |
| `glm-5.3-flash` | `fused_moe_kernel` | Original native graph-versus-eager failure remains unresolved. A later diagnostic did not reproduce it and did not qualify the task. |
| `kimi-k3` | `dense_bf16_gemm_cijk` | Full 40-case native, trusted, and framework qualification. A targeted stream diagnostic covers only its recorded case. |
| `kimi-k3` | `attn_residual_aggregate_hip` | Complete 69-case qualification using the integrated complete-case route. Job 207739 preserved reference compile/correctness for all 69 cases; its last performance snapshot recorded 18/69 trusted and 11/69 framework cases before cancellation. |
| `kimi-k3` | `moe_gemm1_stage1` | Fresh qualification of the current routing and paired-work contract; older routing snapshots are not current qualification. |
| `kimi-k3` | `moe_gemm2_stage2` | Fresh qualification of the current routing and paired-work contract; older routing snapshots are not current qualification. |
| `minimax-m3` | `decode_score_kernel` | Complete current framework and trusted qualification of the 16-case paired contract through the integrated complete-case route; interrupted samples are not a completed union. |
| `minimax-m3` | `gqa_share_sparse_decode_kernel` | Complete current framework and trusted qualification of the 24-case paired contract through the integrated complete-case route. |
| `minimax-m3` | `gqa_share_sparse_fwd_kernel` | Complete current framework and trusted qualification of the 40-case paired contract through the integrated complete-case route. The earlier timeout replacement was prepared but not launched before external preemption. |
| `qwen3.8-2.4t` | `dense_bf16_gemm_cluster` | Completed scoped qualification: retained eed4 framework 209670 and fresh full trusted 210191, all five cases. Earlier partial attempts 209816, 209857, 209907, 209931 and 210005 remain excluded. See the current guide above. |
| `qwen3.8-2.4t` | `fused_moe_2stage_mxfp4` | Completed scoped qualification at `096452d9`: fresh 12-check framework PASS plus complete retained `eed4df57` trusted evidence; original WARN preserved. |
| `qwen3.8-2.4t` | `fused_recurrent_gated_delta_rule_decode` | [Scoped qualification complete](../../ready/current/qwen-recurrent/README.md): six trusted phases at `eed4df57`/209670 and explicitly compatible 12-check framework evidence at `18bb4d67`/209533. Earlier partial runs are excluded; no accepted gain or new serving capture. |
| `qwen3.8-2.4t` | `paged_attention_decode` | [Scoped qualification complete](../../ready/current/qwen-attention/README.md): six trusted phases at `18bb4d67`/209572 and explicitly compatible 12-check framework evidence at `8956e210`/208015. Earlier incomplete attempts are excluded; no accepted gain or new serving capture. |

Two configured historical packages are outside the 23 active mappings: DeepSeek `dsa_sparse_mla_attn` is superseded in this recipe by unified paged prefill/decode, and GLM `elementwise_copy_cluster` recorded aliases rather than materialized GPU copies. Full original served MoE payload capture and broader coverage beyond the published scopes also remain separate outstanding work.
