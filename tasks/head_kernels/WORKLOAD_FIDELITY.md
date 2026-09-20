# Workload fidelity of the top-five head-kernel tasks

All 18 task contracts have been checked for the generated-input integration through
`15f43104`, including the MiniMax observed-control fix (`1567c4d0`) and Kimi
qualification gates (`ad9d721b`, `c0ed5a10`). This document records **input and score
eligibility**, not a blanket GPU-validation result. See [VALIDATION.md](VALIDATION.md)
for actual runs and outcomes.

Six aggregate workload-performance paths are explicitly blocked because required
original controls are missing or estimated: GLM fused MoE, all three Kimi tasks,
and both MiniMax decode tasks. Twelve operator performance paths remain enabled,
with the profile-only, conditional, compacted and partial-coverage limits below.
A passing operator test does not establish the original workload's end-to-end
throughput, original image identity, or complete invocation distribution.

A subsequent archive/UT check also found incomplete mandatory sequence inputs
in both DeepSeek MoE tasks. Their five captured benchmark shapes are retained,
but only 183 of the 256 required sequence calls have corresponding inputs. The
remaining 73 calls use M1920. The original UT silently skipped them; the current
tasks explicitly fail input completeness before GPU setup. Neither task can join
the native-verified subset until those inputs are supplied. See their task-local
`ut/sequence_coverage_evidence.json` records; no reduced-sequence pass is allowed.

The required structural inputs are committed with the tasks. No external tensor
archive is required by the normal path. **Fixture-free does not mean capture-complete:**
large numerical tensors are generated locally, while retained structural controls
are decoded from checked metadata. The generated numbers are not the original
activation/weight sample or a reproduction of its numerical distribution. Valid
seeded FP8/BF16/MXFP4 values do not establish original model accuracy.

| Task | Current score eligibility | Retained observations and remaining limits |
| --- | --- | --- |
| [deepseek-v4-pro__dsa_sparse_mla_attn](deepseek-v4-pro/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/dsa_sparse_mla_attn/ut/meta.json) | 4 observed cases enabled; 2 derived M1 probes unscored | Captured sparse indices, pool dimensions and padded FP8 strides are retained. The compact-pool fallback is removed. Generated numbers and new runtime timing are not the original numerical sample or end-to-end replay. |
| [deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl](deepseek-v4-pro/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/moe_stage1_grouped_gemm_silu_flydsl/ut/meta.json) | 5 retained benchmark cases; qualification blocked by missing sequence inputs | Exact routing, packed layouts and per-case launch kwargs remain bound to the recorded cases. Defined secondary scale/output behavior is checked; unexplained scratch padding is not treated as a meaningful output. |
| [deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4](deepseek-v4-pro/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/moe_stage2_down_proj_reduce_opus_a8w4/ut/meta.json) | 5 retained benchmark cases; qualification blocked by missing sequence inputs | Preserves the whole stage-two GEMM/reduction segment, captured routing, per-case atomic versus route-out choices and caller-output contract. This is a bounded operator case set. |
| [glm-5.3-flash__gemm_a16w16_bf16_cijk](glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/gemm_a16w16_bf16_cijk/ut/meta.json) | 7 observed M64 profile cases enabled; 20 others correctness-only | All 27 shape combinations remain available for correctness, but M1/M8192 priors and unprofiled families are excluded from scoring. Profile support is not a frozen original operand payload. |
| [glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle](glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/ck_gemm_a8w8_blockscale_bpreshuffle/ut/meta.json) | 7 observed M64 profile cases enabled; 14 others correctness-only | Preserves FP8 block scaling, preshuffled weights and transpose-contiguous activation scales. The other Cartesian M combinations are generalization cases, not observed performance samples. |
| [glm-5.3-flash__elementwise_copy_cluster](glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/elementwise_copy_cluster/ut/meta.json) | 8 conditional producer-contract cases enabled | Generated scale inputs model the stated producer layout. The source seed is a prior candidate; raw initialization layouts and producer-condition branches must not be relabeled exact original timed payloads. |
| [glm-5.3-flash__fused_moe_kernel](glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/fused_moe_kernel/ut/meta.json) | Aggregate workload score blocked | Original archives contain only initialization/post-call M19/M1/M8192 records. M64/M16384 exist only in counters. Retained-routing semantic probes and generated robustness cases are unscored. |
| [kimi-k3__fwd_grouped_kernel_stage1](kimi-k3/isl8192_osl1024_conc64_tp8_mi355x/sglang-rocm-k3_rocm720-mi35x-k3-20260727-tl312-08011830/fwd_grouped_kernel_stage1/ut/meta.json) | Aggregate workload score blocked | Original pool/context mapping is missing. Archived 395/524352-row subsets and inferred context-8704 padded pools are different scenarios. The prior-candidate seed remains explicitly identified. |
| [kimi-k3__moe_gemm1_stage1](kimi-k3/isl8192_osl1024_conc64_tp8_mi355x/sglang-rocm-k3_rocm720-mi35x-k3-20260727-tl312-08011830/moe_gemm1_stage1/ut/meta.json) | Aggregate workload score blocked | Stage-one routing is second-hand and the decode launch variant is assumed. M1 is boundary-only. A separate independent single-launch reference gate supplements, rather than renames, the inherited median check. |
| [kimi-k3__moe_gemm2_stage2](kimi-k3/isl8192_osl1024_conc64_tp8_mi355x/sglang-rocm-k3_rocm720-mi35x-k3-20260727-tl312-08011830/moe_gemm2_stage2/ut/meta.json) | Aggregate workload score blocked | The two prefill cases have trace support, but decode controls/weights remain estimated. The retained 0817 chunk-16384 scenario does not establish equivalence to the 0828 chunk-8192 scenario. |
| [minimax-m3__decode_score_kernel](minimax-m3-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/decode_score_kernel/ut/meta.json) | Serving workload score blocked | Histogram-derived sequence lengths and paging choices are not exact served per-call controls. Short captured initialization cases and generated robustness proxies remain unscored. |
| [minimax-m3__gqa_share_sparse_decode_kernel](minimax-m3-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/gqa_share_sparse_decode_kernel/ut/meta.json) | Serving workload score blocked | The context-8704 and sparse block choices are analytic proxies. Correctness and replay probes do not qualify them as observed serving cases; there is no warmup fallback score. |
| [minimax-m3__gqa_share_sparse_fwd_kernel](minimax-m3-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/gqa_share_sparse_fwd_kernel/ut/meta.json) | 1 observed M8192 prefill case enabled | Uses the full captured request-table structure [4097,11268], stride [11268,1], original int64 slot ID [4], full K/V pool dimensions and retained top-k/ragged controls. M1/M186 warmups are excluded. |
| [qwen3.8-2.4t__dense_bf16_gemm_cluster](qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/dense_bf16_gemm_cluster/ut/meta.json) | 5 recorded shape/dispatch cases enabled | Five exact shape/stride and dispatch-table contracts are preserved, including the native retained control. Runtime preparation sets the intended table before imports. The scored baseline is the configured native operator, while PyTorch is the independent correctness reference; the candidate body is bound independently of cached operator registration. |
| [qwen3.8-2.4t__fused_moe_2stage_mxfp4](qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/fused_moe_2stage_mxfp4/ut/meta.json) | 2 persisted composite cases enabled | Retains M64/M8192 router indices, weights, mask, packed layout and is_shuffled attributes. M5191/M16384 occur in telemetry but were not persisted; the two stages share one captured composite oracle. |
| [qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode](qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/fused_recurrent_gated_delta_rule_decode/ut/meta.json) | 2 retained structural cases enabled | Retains B64/B1 shapes, all 199 state slots, exact state indices and output/state alias contract. Numerical state is generated, and reset/transition checks are not a replay of the entire original stream. |
| [qwen3.8-2.4t__gemma_fused_add_rmsnorm](qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/gemma_fused_add_rmsnorm/ut/meta.json) | 2 recorded shape contracts enabled | Preserves M8192/M64, strides, epsilon and two fresh nonaliasing outputs. Generated values are appropriate to this value-independent structure, but do not reproduce original end-to-end performance. |
| [qwen3.8-2.4t__paged_attention_decode](qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/paged_attention_decode/ut/meta.json) | 1 archived compacted case enabled; original-pool fidelity not established | The shipping helper still uses 521351 compacted pages and remapped indices, rather than the original 693536-page pools. One persisted case does not cover the 1024 signatures in capture telemetry. |

**Read score policy separately from case inventory.** A task may preserve a shape
for correctness, replay, robustness or semantic testing while disallowing it from
performance scoring. The current Kimi catalogs report zero scored cases while retaining their nine
semantic/diagnostic cases. A nominal semantic inventory is therefore not a count
of eligible scored cases. The governing `workload_scoring`,
`performance_contract` or observed-case selector must permit the score. A missing
required record must fail, not fall back to a small warmup or an analytic proxy.

**Missing capture contracts for blocked workloads**

The following is the concrete evidence needed to reopen each blocked score. It is
not a request to fabricate replacement values or implement another capture tool.
Every record must identify the actual source run/rank, source revision and loaded
runtime, preserve pre-call state and output aliases, and bind its control buffers
to the observed device launch and workload phase.

- **GLM fused MoE:** obtain authentic pre-call M64 and M16384 records. These need
  hidden states `[M,4096]`, exact top-k IDs/weights `[M,8]`, expert weights
  `[288,512,4096]` and `[288,4096,256]`, the corresponding block scales, physical
  strides/storage relationships, and the original in-place hidden-state/output
  alias. Preserve quantization, interleaving, activation/scaling flags and the
  actual per-case tile/grid selection. All nine recovered archives from source
  run `20260829T045924Z-d82dabe1`, cycle1 `fused_moe_kernel_task`, contain only
  M19/M1/M8192 heavy records. `CAPTURE_MAX=2`, after-call snapshots, eager capture
  and zero timed repetitions explain why M64/M16384 counters cannot supply this
  contract. An authenticated matching capture elsewhere, or fresh capture, is
  required; bootstrap sampling cannot replace it.
- **Kimi grouped attention:** retain the true per-sequence lengths and original
  `kv_indptr`, page indices, slot-pool allocation/strides and live split counts.
  Preserve Q `[B,12,576]`, the latent K pool, the V view over its first 512 lanes,
  caller output/LSE buffers and their aliases. The 256 split-slot capacity is not
  a measured split count. Record `has_mla`, page size, scale, launch options and
  the selected variant. Neither `ISL + OSL/2 = 8704` nor a `B*context + 64` padded
  pool reconstructs the original 833385-row pool and its addresses.
- **Kimi MoE stage one:** capture routing at the stage-one seam, rather than
  reconstructing it from the 0817 stage-two prefill. Bind the exact sorted token
  and expert IDs, valid counts, padding markers and top-k policy to each intended
  prefill/decode call. Preserve A `[M,3584]`, packed W1 `[896,768,1792]`, scales
  `[688128,112]`, output `[M,16,384]`, their strides and caller-owned output.
  Observe the decode variant instead of assuming the prefill
  `t32x64x256_w3_xcd4_kw2` choices. The new independent reference checks each
  individual launch for three trials on each of three semantic shapes; it does
  not make second-hand routing an observed scenario or turn a median into a
  single-launch pass.
- **Kimi MoE stage two:** bind the intended source run and chunk policy first.
  The retained 0817 source has prefill chunk 16384; the 0828 workload uses 8192.
  Preserve `[M,16,384]` intermediate states, packed W2 `[896,3584,192]`, scale
  arrays, exact sorted routing/weights, output `[M,3584]`, and atomic-output
  zeroing versus route-out/reduction behavior. The two retained prefill cases
  have trace support. The declared decode controls/aggregate weights need actual
  trace evidence rather than regime-floor estimates, including their real
  persistent/atomic/reduce and split choices. Until that coverage exists, the
  aggregate score remains disabled even if the retained semantic cases pass.
- **MiniMax decode score:** obtain exact served B1/B64 calls, including true
  sequence lengths, maximum context, original `[4097,11268]` request table and
  strides, int64 slot IDs, paging rows and full `[4358330,1,128]` K allocation.
  Bind Q `[B,1,128]`, block/top-k/init/local-block settings, score type, page size,
  scale flags and actual launch geometry to those calls. A histogram's
  representative context 8554 and generated ragged pattern are not such a record.
- **MiniMax GQA decode:** obtain the same original request-table/slot/length and
  K/V-pool contracts, plus Q `[B,8,128]` and the actual selected sparse block IDs
  `[1,B,16]`, sink/scale choices and launch split/grid. The analytic context 8704
  and generated top-k/paging choices remain unscored robustness inputs.

**Other limits that must remain visible**

The two GLM GEMM tasks now score only the seven M64 families supported by the
retained profile; all 27/21 correctness shapes remain. This fixes the earlier
promotion of Cartesian M1/M8192 priors. It does not recover original raw tensor
payloads or prove the same vendor solution on a different runtime. GLM copy's
eight timing cases depend on its producer-layout contract and a prior-candidate
seed; they are not identical to every raw initialization layout.

The three generated Qwen tasks preserve their persisted structural case sets.
For fused MoE, telemetry records M64/M5191/M8192/M16384, while only M64/M8192 were
persisted. For paged attention, the retained pool and indices were compacted:
the current test explicitly checks `[521351,1,1,256]` pools and compact index IDs.
Restoring the original 693536-page allocation **and original page-ID mapping**
requires that mapping's evidence; enlarging a buffer alone would not do it.
Missing variants require corresponding controls or an explicitly narrower scope.
A B1 recurrent case has recorded observations and must not be labeled warmup-only
without its phase evidence.

**Benchmark, source and runtime scope**

Arena compares the committed starting implementation and candidate through a
protected repeated-operator benchmark. Current paths use materialized graph
measurement and input-state restoration; original workload evidence includes
explicit eager GLM execution and eager-prefill/graph-decode distinctions. Resetting
state and immutable input storages between samples is not a replay of the original
cache or temporal stream. The evaluator's equal arithmetic mean of per-case ratios
is also distinct from recorded invocation frequencies or serving-weighted results.
These differences must remain named when interpreting measurements.

The Kimi attention and GLM copy starting sources are declared prior candidates.
They must not be relabeled as untuned stock. Public runtime digest pins differ
from historical capture images, including the custom Kimi capture stack. A tag,
source symbol, shape match or successful compilation does not prove loaded-source
or device-dispatch equivalence. Current GPU checks and device traces provide their
own evidence; they do not retroactively certify the original image or end-to-end
experiment.

**What the earlier mistake does and does not establish**

The initial local integration promoted some histogram/analytic, inferred or
boundary cases into an "exact observed workload" interpretation before that claim
was proved. That was an integration and acceptance error; some of those proxy
recipes were inherited from standalone upstream UTs and still needed explicit
scope and score exclusion. The corrected gates and observed-case selectors address
that mistake.

For the specific GLM fused-MoE package, the
[original-UT comparison](WORKLOAD_FIDELITY_EVIDENCE.json) is more precise: the
archived original harness already resampled captured router rows with replacement,
generated normal-distributed activations, and timed M64/M8192/M1 while omitting
observed M16384. Six case-building functions are AST-identical to the initial
suite. The initial suite did not invent M64 or introduce that generation policy;
it overclassified an inherited controlled proxy as exact captured-call fidelity.
The archived status states zero authored candidates, a null candidate, no accepted
patch and no end-to-end validation. The surviving result is a harness smoke test,
not evidence of useful optimization on wrong shapes. A complete original optimizer
prompt was not found in the inspected packet.

The original GLM UT's primary weighted score used **measured baseline latency
multiplied by analytic call counts**: decode 1024, prefill 64, and transient M1
one call. Its metadata's normalized 0.7/0.3/0 fields must not be quoted as the exact
primary-score shares. Neither that scorer nor the current equal-case Arena score
is an end-to-end model measurement.

Original archive omissions, changed chunk policies and partial captures are
provenance limitations. They do **not** establish that GEAK was globally given
wrong dimensions, executed the wrong kernel in every original workload, or lost a
verified optimization gain. Such a conclusion needs the original per-call
launch/input evidence. This matrix distinguishes what was retained, what was
approximated, and what is now refused instead of treating missing evidence as a
historical failure result.
