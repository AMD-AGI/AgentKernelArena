# PR107 integration with pinned main — 2026-09-15

This record retains the `c1dc5e09` merge decisions and the subsequent qualification
checkpoint. For source-specific CPU, GPU, quality-loop and agent-matrix proofs, see the
[2026-09-15 verification checkpoint](verification-2026-09-15.md).

The frozen `e8ec5d6b` campaign remains historical evidence for all **438** retained
task packages, with each report's actual source, runtime and model identity.
The integration and its successors do not replace those reports.

## Qualification checkpoint

| Scope | Recorded outcome |
| --- | --- |
| Original c1 revalidation | All 55 outcomes retained: **49 PASS / 1 WARN / 5 FAIL**. One QKV task-level PASS carries an explicit outer bytecode-inventory exception; its original outer FAIL remains recorded. |
| Fresh `400fcb9d` task validators | **6 PASS / 1 Pack semantic FAIL**. Both MoE tasks passed full validation. Paired study 141240 passed independent reparse of 76 public actions, including 32 measured actions / 512 case observations. |
| Full 400f CPU suite | **13,996 passed / 4 fixture failures / 6 skipped**, plus 6 passing subtests. Failed output is preserved. |
| `9c4c99f1` successor | Fresh Pack GPU **PASS**, independently audited: **11 correctness / 5 scored cases**. Full CPU: **14,013 passed / 1 old Pack source-hash fixture failure / 6 skipped**, plus 6 passing subtests, **612.74 s**. Full CPU qualification remains open. |
| Current task applicability at 9c | All **438 task packages** have runtime-applicable validator PASS evidence with their original framework/runtime identities; this is not one fresh final-framework campaign. |
| `f65e1e55` CPU follow-up | Test-only exact Pack source-delta check repaired; **168 focused PASS**. The immutable full core CPU run is in progress, with its result **PENDING**. The GEAK Codex extension requires separate qualification. |
| Five saved candidates | **35 formal actions PASS** at 400f; AWQ/apply_write task bytes match 9c. **Zero new optimization searches**. Original search/source evidence is retained. |
| Agent matrix | **117 completed / 113 accepted**: Codex 45/45, Claude Code 22/21, Forge 45/42, GEAK 5/5. Completion and acceptance are separate; 63 completion pairs remain. |

The corrected 438-task aggregate reuses 383 historical reports already qualified
for the e8 task versions, preserving their individual actual worker revisions;
e8 is their applicability checkpoint, not their common execution revision.
The remaining applicability checkpoints are c1 (48), 400f (six) and 9c (one).
These counts describe applicability, not execution revisions. The v2 aggregate
retains worker IDs as recorded, including abbreviated IDs, without changing any
PASS or source binding.

The original Pack GPU probe passed the 2-D all-empty zero-grid path and failed
the 3-D all-empty `reshape(0, -1)` before JIT. The successor uses
`flatten(start_dim=1)`, retaining the original launch and scored cases; it does
not bypass the work with an early return. Job 141241's fresh full validator
covered both new unscored controls and passed; this does not erase the earlier
semantic FAIL or qualify the separate failed CPU run.

The paired MoE study reports median after/before latencies of **0.9357/0.9330**
for instruction baseline/candidate and **0.9488/0.9484** for hard MoE. With unchanged
candidate code, after candidate/baseline ratios are **1.0007/1.0011**. This is a
harness preparation-boundary effect, not agent optimization gain. Source binding,
measurement limits and the completed independent raw audit are recorded in the
dated checkpoint.

Authentication is repaired, while the shared Claude/GEAK quota reset remains
**2026-09-16 00:00 UTC**. The user's revised plan supersedes midnight dispatch:
both old waiters are **DISARMED**, and no new Fable/Opus dispatch is authorized.
Remaining Claude Code 23 tasks have an explicit Sonnet 5 / medium configuration
prepared. GEAK's remaining 40 have a draft native Workflow Codex implementation
that is not yet qualified.
Each campaign remains on **HOLD** until its own source, model, backend and runtime
are qualified; Sonnet qualification is independent of the GEAK extension.
Historical model provenance and the 117/113 matrix are unchanged. No new Forge
runs are required. Legacy GEAK source-39 evidence still lacks complete original
tool-ID/session correlation and is not qualification
of the later strict completion gate.

The source classification and reconciliation below describe the original c1
merge. Later Pack changes and framework repairs are separately bound in the
dated verification checkpoint; no historical outcome is silently upgraded.

## Pinned inputs and scope

- Refactor parent: `e8ec5d6b4bc9d62b38af59a66a1797da42d3f30f`.
- Main parent: `0acf65b3a967ef1025dbfc5fd4b415b259e3bd43`.
- Common base: `4fd19785e545ba7b36bf53ab279dbaca2f5160f5`.
- Review covered all 76 upstream-touched task roots: 42 conflicted and 34
  automatically merged. Automatic merging was not treated as semantic approval.
- All 438 task configs remain byte-identical to e8 and use TaskSpec v2. All
  original manifest case rows remain exact; every added row is correctness-only.
- The 61 requested README cleanups remove 499 trailing-whitespace lines and
  excess EOF blank lines in 33 files, preserving text in that cleanup step.
  Two of those READMEs also receive a separate semantic correction for the new
  MoE preparation boundary; their cleanup and final hashes are both retained.
  README differences are classified separately from executable task changes.

The merge follows upstream retirement of `geak_v3`, `geak_v3_triton`, and
`mini_swe_triton`, including registration, Docker routing, and obsolete tests.
Canonical GEAK, the `geak_v4` compatibility integration, Forge and its public
rewrite alias, Cursor, Claude Code, Codex, and task_validator remain registered.

Shared task loading, protocol, sessions, evaluator, scoring, harness guard, and
canonical performance helpers remain byte-identical to e8. The only `src/`
changes are registration and Docker routing for the retired integrations.

Quality-loop changes combine v2 materialized-source exclusion with upstream
artifact filtering and publication guards: generated reports, compiled objects,
named benchmark tensor outputs and ELF executables cannot enter task commits;
intentional tensor/binary inputs are retained. Pending-path enumeration is
NUL-delimited, and publication rechecks old accepted manifests. Input fixtures
and each declared materialized source retain their v2 containment boundaries.
`agents/quality_loop/backend.py` and `prompts.py` remain e8 bytes. The separately
owned stdin/raw-role-evidence fixes are deliberately outside this merge.

## Source applicability and verification

| Task classification relative to e8 | Count | Evidence treatment |
| --- | ---: | --- |
| Entire task tree unchanged | 328 | Retain exact historical task evidence |
| Documentation only; runtime unchanged | 54 | Retain runtime evidence; bind new README separately |
| Added task-local regression test only | 1 | Retain runtime evidence |
| Changed protected runner, reference, preparation or correctness controls | 55 | Fresh full task_validator required |

The 55 changed tasks keep original candidate kernel functions and scored cases.
Where candidate and harness share a file, the declared editable symbols have
identical ASTs. Moving host scalar preparation in the two ROCm MoE tasks does
change the measurement boundary; it requires explicit paired timing evidence.
A task hash alone does not establish agent/framework equivalence: selected-agent
code and shared execution are checked separately from these task classifications.
Quality-loop execution/publication changes need their own review and smoke test.

CPU verification for this candidate includes:

- 4,123 focused task-migration, public-runner, independent-oracle and negative
  control checks passed. Device literals are redirected only in temporary CPU
  fixtures; this is not GPU validation.
- Targeted integration/quality checks include candidate dispatch
  through every new vLLM control, preserved FP8 input bytes/strides, nested input
  restoration, native translation-unit macro isolation, missing output writes,
  disabled routing weights, and missing reduction work. The installed FP8 guard
  additionally has 14 full-correctness regressions for original/new dtype and
  rank paths and read-only/metadata/numerical failures.
- `make check-perf-helpers` passed with canonical helpers unchanged.
- `make check-docker-runner` passed.
- The first complete CPU pass exposed 27 historical fixture assumptions; after
  separating new controls, 982 focused regressions passed while retaining the
  original fingerprints and per-case failure classifications.
- Full CPU-suite result and remaining skip reasons are recorded in the merge
  handoff. The final run uses the pinned `AKA_FORGE_PROBE_PYTHON` environment
  and records its source-tree identity at both start and completion.

The first full CPU invocation used an excessively long shared-filesystem temp
path, causing AF_UNIX socket setup failures. That interrupted output is retained;
the complete rerun uses a short local temporary path. No runtime policy was
relaxed for this environment issue.

Reproducible per-file SHA-256 and per-task classifications, manifest bindings,
all 76 decisions, original failed test logs, and test commands are retained in
the review bundle `logs/pr107-main-integration-20260915/`. Its
`audit_candidate.py` checks actual candidate bytes against both pinned parents;
`candidate-applicability.json` lists every changed file and all 438 task rows.
Logs are review artifacts, not committed task assets.

## Semantic choices and GPU requalification

Retaining the current stronger checks is intentional. In particular:

- Quantized MoE keeps the input-derived forward-error bound for all original,
  timed, and replayed cases, plus the old allclose gate and exact controls. Main's
  alternate scored-input magnitudes are not adopted. A new unscored INT4 control
  verifies disabled routing multiplication.
- Groupwise FP8 controls run through the installed protected guard, extended
  for 3-D inputs and explicit output dtype. Platform clamp limits and original
  2-D/default-dtype references and gates remain unchanged. A reviewer reproduced
  false failures before this repair; those CPU diagnostics are preserved.
- Bitmatrix controls use current true set-membership semantics. The upstream
  reference still incorrectly attributed padding lanes to expert 31.
- Linear attention preserves the public unwritten-padding output contract and
  complete mutable-cache checks. Monkeypatching candidate allocations to zeros
  would change that public contract and is not adopted.
- Existing real backward RMS timing and the explicitly declared fixed-launch
  multreduce kernel supersede older upstream wrapper changes.
- HIP MLA isolates the editable header in a separate candidate translation unit,
  preventing candidate macros from rewriting the protected host evaluator.
- New vLLM public branches run via task-owned candidate dispatch and explicit
  unscored manifests. Existing original/timed/pristine/replay contracts remain.

The original GPU requalification plan used the reviewed frozen merge commit and
a qualified pinned runtime described in
[runtime qualification](runtime-upgrade-qualification.md). Its work items were:

1. Validate HIP MLA's separate compilation and equal work/timing boundaries, and
   nonuniform backward gradients for `assign_score_withk`.
2. Run paired baseline/candidate event measurements for the instruction and
   ROCmBench MoE GEMM tasks. Verify unchanged launch arguments, outputs, scored
   cases, warmups and sample counts while host scalar reads occur before timing.
3. Validate the four changed GEAK task harnesses, including MLA's unscored RoPE
   branches and protected references/inputs.
4. Validate new ROCm tail/precision controls and all 42 changed vLLM task roots.
   Every changed root needs a framework-finalized full validator `PASS`, including
   compile, correctness, and performance, on its final exact source bytes.
5. Review quality-loop repair/publication behavior after separately integrating
   the backend evidence work. CPU mocks do not prove real model activity.

All 55 original task outcomes are now recorded, with repairs and remaining gates
listed above. Newly exposed baseline/compiler or coverage issues remain failures
until separately repaired and requalified; no cases or tolerances are relaxed to
match the baseline. The original merge construction and this documentation update
did not themselves run GPU/provider jobs, push the PR, change its base ref or
mutate campaign controls. Actual subsequent runs are identified in the dated
verification record.

## Per-task reconciliation

Paths below are relative to `tasks/`. “Combine” means a fresh validator is needed;
“Retain” means e8 executable task bytes are kept (documentation or local test-only
differences are separately recorded).

| Task | Decision | Semantic evidence |
| --- | --- | --- |
| `hip2hip/gpumode/FusedLeakyReLU` | Retain | Upstream changes only task-local helper/reference paths, import spellings and protective comments (verified file mapping base/main). V2 already protects every noneditable path and declares explicit runner/ref/baseline; retain coherent existing layout, avoiding Git cross-task directory rename inference. Current case/replay controls and entire task tree unchanged. |
| `hip2hip/gpumode/NormalAttention_embedded_gaussian` | Retain | Upstream changes only task-local helper/reference paths, import spellings and protective comments (verified file mapping base/main). V2 already protects every noneditable path and declares explicit runner/ref/baseline; retain coherent existing layout, avoiding Git cross-task directory rename inference. Current case/replay controls and entire task tree unchanged. |
| `hip2hip/gpumode/Sigmoid` | Retain | Upstream changes only task-local helper/reference paths, import spellings and protective comments (verified file mapping base/main). V2 already protects every noneditable path and declares explicit runner/ref/baseline; retain coherent existing layout, avoiding Git cross-task directory rename inference. Current case/replay controls and entire task tree unchanged. |
| `hip2hip/others/assign_score_withk` | Combine | Port upstream analytic backward reference and nonuniform-gradient correctness on all5 original shapes; freeze oracle and readonly inputs before candidate. Original forward and forward+sum-backward scored inputs, timings and1e-3 gates unchanged. |
| `hip2hip/others/mla_decode` | Combine | Separate editable header into protected candidate translation unit to prevent macros affecting host reference/launch/timing. Keep current kernel bytes, five shapes, routing_controls valid-length work, exact input generators, output_validation gates, actual measured/replay/pristine checks. Keep v2 action paths; add upstream launch guidance. |
| `instruction2triton/rocmbench/moe_gemm` | Combine | Use upstream prepared launch callable, hoisting static GPU scalar out of timed work; preserve original device kernel/grid/cases/event method and v2 independent output gate. Fresh comparative GPU required. |
| `triton2triton/geak_eval/L1/mla_decode` | Combine | Keep current all-coordinate input-derived rounding bound/convex range and original no-RoPE scored work. Port upstream RoPE branch as two explicit unscored rows with private transformed oracle inputs and rotated-key output check. Do not replace original cases with alternating RoPE or use fixed relaxed outlier threshold. |
| `triton2triton/geak_eval/L1/refk_fp8_blockwise_mm` | Combine | Protected get_inputs inlines original _generate_input with identical RNG/generator/order/FP8-conversion/strides. Protected reference only inlines identical block constants and equivalent scale-dimension unpacking; retain current _timed_contract precise all-output/pristine/replay gates. Source/module boundary changes require fresh validation, no numerical or scored input change. |
| `triton2triton/geak_eval/L1/refk_identity` | Combine | Protected get_inputs, identity_pytorch and identity_triton bodies AST-equivalent to original kernel helpers excluding docstrings; remove editable reference/config exports while JIT _identity_kernel and original timed wrapper behavior unchanged. Source/module boundary changes require fresh validation, no numerical or scored input change. |
| `triton2triton/geak_eval/L2/fast_rms_layernorm` | Retain | checked_call already checks both forward output and backward gradient for Gemma on/off on every original shape; pristine inputs, nonuniform-gradient/tail controls and actual timed replay are additional to upstream forward assertions. |
| `triton2triton/geak_eval/L2/topk` | Retain | Exact source-gather equality, exact torch.topk values and unique in-range indices in check_topk imply the upstream tighter 1e-4 check. Both original approximate gates retained; ties and strided/two-stage controls already explicit. |
| `triton2triton/geak_eval/L3/fused_mxfp4_quant_moe_sort` | Retain | Upstream only emits benchmark sidecar. V2 captures every case with metadata and timed checks through _arena_actions/_arena_checks. Keep full case manifest and runtime capability failure. |
| `triton2triton/geak_eval/L3/fused_qkv_rope` | Combine | Move upstream launch/input/reference helpers into protected harness, retaining e8 checked_benchmark (pristine reference, actual measured outputs, poison/perturbed replay, finally restore); omit superseded CapturedGraphRun. |
| `triton2triton/geak_eval/L3/fused_rms_fp8` | Retain | All three upstream optional wrappers already run via seven explicit CONTROL_CASES and independent _contract_oracles, checking each optional output, scale and FP8 code. Existing all-output timed replay and 1-ULP raw-quant allowance plus original reconstructed gates retained. |
| `triton2triton/geak_eval/L3/gemm` | Retain | Upstream only emits a performance_report.json sidecar. V2 _arena_eval.capture_performance already captures every benchmark call into public per-case envelope; _arena_actions preserves full-output/pristine/replay guards. No missing numerical coverage. |
| `triton2triton/rocmbench/easy/test_block_copy` | Combine | Port three unscored odd-length padding tails; preserve exact defined-output/invalid integer NaN rules and all original cases. |
| `triton2triton/rocmbench/easy/test_randn` | Combine | Port nine unscored seed/repeat/tail tests. Existing exact Philox performance and original statistical gates retained. |
| `triton2triton/rocmbench/hard/moe_gemm` | Combine | Hoist static GPU scalar read into metadata construction as upstream; preserve event method/cases/gates and current independent pre/post measured check. Add upstream FP32 expert oracle as extra gate. Timing host-boundary changes require fresh comparative GPU evidence. |
| `triton2triton/rocmbench/hard/rmsnorm_bwd` | Retain | Current path already times declared rms_bwd_kernel and checks dx/raw dg with pristine input + actual replay; retain current event-only qualification and RNG/output layout. Upstream forward-to-backward fix is superseded. |
| `triton2triton/rocmbench/hard/test_tma_store_gemm` | Combine | Port four unscored transpose/output-dtype/K-block controls without changing scored rows. |
| `triton2triton/rocmbench/hard/triton_multreduce_matmul_kernel` | Retain | Current v2 explicitly declares fixed-launch core triton_matmul_kernel and autotuned wrapper as allowed entrypoints; its measured core is now a declared editable target. Retain fixed tile/USE_DOT=False/EVEN_K path instead of changing timing to .fn heuristic wrapper. |
| `triton2triton/rocmbench/medium/test_cast_matmul` | Combine | Port two upstream ragged cases in addition to current strided/tail controls; retain precise oracle and current measured replay. |
| `triton2triton/rocmbench/medium/test_triton_swizzle2d` | Combine | Keep original test ID and golden table. Add three separately named unscored sentinel/group-order/dtype controls from upstream. |
| `triton2triton/vllm/triton_apply_grammar_bitmask` | Combine | Masked vocab tails below/above 8192 with nonidentity selected rows. Port missing cases via disjoint correctness-only indices and explicit protected manifest, using existing guarded loader. Preserve existing runner numerical/generator/performance bodies except early control dispatch. |
| `triton2triton/vllm/triton_apply_write` | Combine | Empty writes, zero-length segments and 1024/2048 write-cap tails. Additional controls only; original numerical, scored input and benchmark bodies retained. |
| `triton2triton/vllm/triton_awq_dequantize` | Combine | FP32/BF16 scale dtypes and signed packed/tail group boundaries. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_bad_words` | Combine | Irregular speculative prefixes plus empty logits/no-bad-words no-op contracts. Additional controls only; original numerical, scored input and benchmark bodies retained. |
| `triton2triton/vllm/triton_bincount` | Retain | Existing control [3,1,0] over 1031 tokens/65 vocabulary covers request subset, zero prompt/prefill, bit31/32/64 and 1024-crossing tails. Full immutable integer oracle and exact replay cover both outputs and inactive rows; upstream adds no missing branch. |
| `triton2triton/vllm/triton_chunked_prefill_paged_decode` | Combine | Strided cache/query and head80/x4, head96/x16 with filtered decode/window/ALiBi. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_compute_identity` | Combine | Odd token counts and signed near-cancelling top-k scales. Port missing cases via disjoint correctness-only indices and explicit protected manifest, using existing guarded loader. Preserve existing runner numerical/generator/performance bodies except early control dispatch. |
| `triton2triton/vllm/triton_compute_slot_mappings` | Combine | Empty and ragged INT64 mappings, strided tables and explicit full padding kernel check. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_copy_and_expand_eagle_inputs` | Retain | Existing _arena_checks checks all six output tensors with dtype/device/arity plus pristine inputs, poisons outputs and validates actual TimedRun rerun. Upstream six-output/captured replay fix is already covered. |
| `triton2triton/vllm/triton_decode_attn_stage1` | Combine | Ragged split/page/head tails with explicit capped logits. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_decode_attn_stage2` | Combine | Nonuniform short sequences and split-count/empty-split boundaries. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_ep_scatter_1` | Retain | Existing deterministic counts [0,1,127,128,129,257,0] cover zero/alignment/one-and-multiple-tile branches, exact offsets/allindices/pristine inputs and timed replay; upstream boundary-count control is covered. |
| `triton2triton/vllm/triton_ep_scatter_2` | Combine | FP32 singleton/topk1 and greater-than-program-cap multi-column dispatch. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_expert_kernel` | Combine | Sub-tile/multi-tile M/N/K and BF16 strided public GEMM. Port missing cases via disjoint correctness-only indices and explicit protected manifest, using existing guarded loader. Preserve existing runner numerical/generator/performance bodies except early control dispatch. |
| `triton2triton/vllm/triton_fla_layernorm` | Retain | Existing pristine independent reference checks all output/mean/rstd and diagnostic nonuniform affine + both RMS/LayerNorm and gate orders. Upstream full auxiliary-output check already covered. |
| `triton2triton/vllm/triton_fla_layernorm_gated` | Combine | FP16/BF16, singleton and 1025-wide tails, epsilon and activation variants. Port missing cases via disjoint correctness-only indices and explicit protected manifest, using existing guarded loader. Preserve existing runner numerical/generator/performance bodies except early control dispatch. |
| `triton2triton/vllm/triton_fused_moe_gptq_awq` | Combine | Retain current original 5 scored inputs and old allclose plus input-derived FP16/FP32 forward-error bound on every measured/correctness/replay path. Reject upstream magnitude/variant changes to original scored cases and nonzero-only gate. Existing exact INT4/INT8 explicit/default basis controls cover upstream quantization variants; add unscored INT4 with provided-but-disabled routed weights. |
| `triton2triton/vllm/triton_fused_moe_lora` | Combine | BF16 sorted routed LoRA and split-K2/no-L2-cache specialization. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_gather_block_tables` | Retain | Current diagnostic source5x1031, counts0/1/3/1024/1031, mapping[4,1,4], nonzero full destination sentinel already covers repeated sparse rows, empty/full/tail widths, 1024 iteration crossing and untouched rows. Preserve exact full-buffer checker; upstream reference signature change superseded. |
| `triton2triton/vllm/triton_kda_dot_kkt_intra` | Retain | Existing scripts.contract_checks.check_outputs requires exact tuple type/arity and each tensor shape/dtype/device, disallows readonly aliases and checks full numerical values before historical zip loop. Upstream output-count fix is covered; retain its regression adapted to actual stronger checker. |
| `triton2triton/vllm/triton_layernorm_gated` | Combine | Explicit output and grouped BF16 tail in addition to current gate-order controls. Port missing cases via disjoint correctness-only indices and explicit protected manifest, using existing guarded loader. Preserve existing runner numerical/generator/performance bodies except early control dispatch. |
| `triton2triton/vllm/triton_linear_attn_decode` | Retain | Current closed-form controls cover sparse/permuted/padded slots, decay and full unused cache. Padded output is explicitly unwritten; reject upstream monkeypatch of candidate torch.empty to zero (changes wrapper semantics). Current full output contract validates active outputs and complete cache, with exact candidate/timed state observers. |
| `triton2triton/vllm/triton_logit_bias` | Retain | Existing _arena_replay.compare rejects NaN and mismatched signed-infinity masks before unchanged 0.01 finite gates, including actual replay. Upstream masked-logit correctness fix is already covered. |
| `triton2triton/vllm/triton_lora_shrink` | Combine | Split-K first/last partial blocks, BF16 4D weights and inactive LoRA. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_merge_16x16_to_32x32` | Combine | FP16/BF16 partial triangular tiles and odd head counts. Port missing cases via disjoint correctness-only indices and explicit protected manifest, using existing guarded loader. Preserve existing runner numerical/generator/performance bodies except early control dispatch. |
| `triton2triton/vllm/triton_moe_mmk` | Combine | Keep current independent full-output/pristine/actual-replay contract and original5 scored cases. Add main4 M/N/K63/65 tails with original47..50 seeds as explicit correctness-only rows. |
| `triton2triton/vllm/triton_mrope` | Combine | BF16 and partial rotary dimensions; preserve full-head original timing. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_pack_bitmatrix` | Combine | INT16 duplicate IDs, topk31/32, expert31/32/33 and row tails. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. Keep current mathematical membership reference; main still incorrectly attributes padded lanes to expert31. CPU independent set-membership controls reject that oracle/old-kernel error. |
| `triton2triton/vllm/triton_pack_seq` | Combine | All-empty/high-rank and explicit packing block/dtype boundaries. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_paged_prefix_prefill_alibi` | Combine | Ragged cache/new-query boundaries with optional ALiBi, window and scale. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_penalties` | Combine | BF16 vocab8209 and per-penalty no-op/speculative branch separation. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_per_token_group_quant_fp8` | Combine | Explicit wider FP8 output, BF16 3D input and FP32/UE8M0 edge values. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. Extend the installed protected guard for leading dimensions and explicit storage dtype while keeping platform clamps and the original 2D/default reference and all gates; full installed-correctness CPU regression covers the reviewer-reproduced false failures. |
| `triton2triton/vllm/triton_per_token_group_quant_int8` | Combine | Exact FP16 extrema/subnormal/half-step quantization and non-power-of-two epsilon controls. Additional controls only; original numerical, scored input and benchmark bodies retained. |
| `triton2triton/vllm/triton_post_update` | Combine | Second64-request program with sparse mapping, duplicate counts and update tails. Additional controls only; original numerical, scored input and benchmark bodies retained. |
| `triton2triton/vllm/triton_prepare_eagle_docode` | Combine | Hidden-state2049 tail and maximum-request257 second block. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_prepare_mrope_positions` | Combine | Large mixed prefill/decode sparse mappings and masked token tails. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_prepare_pos_seq_lens` | Combine | Keep current pristine/full-output timed checker and all original input/timing/reset behavior. Port upstream sentinel detection to existing unscored actual replay by poisoning pos/seq buffers after timing, including inactive zero-write rows. |
| `triton2triton/vllm/triton_prepare_prefill_inputs` | Combine | Ragged prefill, zero lengths, completed requests and token-block tails. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_ranks` | Combine | FP16 ties and vocabulary tails across 8192/32768. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_reduce_segments` | Combine | Packed variable queries, zero denominator/extreme maxima and FP32 output. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_reshape_and_cache_flash_diffkv` | Combine | Strided BF16 full backing-storage guard plus explicit scaled FP8 cache. Additional controls only; original numerical, scored input and benchmark bodies retained. |
| `triton2triton/vllm/triton_sample_recovered_tokens` | Combine | Singleton, empty/ragged requests and deterministic recovered-distribution branch. Additional controls only; original numerical, scored input and benchmark bodies retained. |
| `triton2triton/vllm/triton_scale_swizzle` | Retain | Existing 129x5 uint8/int8/FP8 all-byte diagnostic simultaneously exercises both padding tails, stronger than separate 129x4 and128x5 cases. Full byte equality and exact measured/replay/source checks retained. |
| `triton2triton/vllm/triton_scaled_mm` | Combine | INT8 and heuristic threshold/tile tails, retaining original FP16 timing. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_silu_mul_fp8_quant_dg` | Combine | BF16 strided grouped activation and group256 branches. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_silu_mul_quant_fp8` | Combine | BF16 UE8M0 and FP32 finite/subnormal edges with provided output. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_ssd_chunk_cumsum` | Combine | Non-power-of-two chunk/head sizes, FP16 strides and partial clamped prefix scans. Additional controls only; original numerical, scored input and benchmark bodies retained. |
| `triton2triton/vllm/triton_ssd_chunk_scan` | Combine | Large dstate/head tails and BF16 irregular chunks with optional D/z/initial state. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_topk_log_softmax` | Combine | Singleton vocab plus below-block and FP16 above-block numerical edges. Port missing cases via disjoint correctness-only indices and explicit protected manifest, using existing guarded loader. Preserve existing runner numerical/generator/performance bodies except early control dispatch. |
| `triton2triton/vllm/triton_unpack_seq` | Combine | INT64 lengths and explicit alternate block dimensions on higher-rank/tail/empty sequences. Port missing cases via disjoint correctness-only indices and explicit protected manifest, using existing guarded loader. Preserve existing runner numerical/generator/performance bodies except early control dispatch. |
| `triton2triton/vllm/triton_update_eagle_inputs` | Combine | Clamp transition combined with1536-wide hidden-state masked block. Only new independent correctness rows are dispatched through the original guarded candidate loader. Original runner/performance bodies retained. |
| `triton2triton/vllm/triton_w8a8_block_int8_matmul` | Retain | Current multidimensional A1x3x67, B65x67, blocks64, BF16/FP32 outputs and zero input controls already cover partial K/N, flattened A/scales and optional dtypes. Existing original default128-block score preserved, all coordinates independently checked before and after actual timing. |
| `triton2triton/vllm/triton_write_zeros_to_output` | Retain | Existing destination observer handles None/views, bitwise zero contract rejects subnormals/nonfinite and fresh actual replay proves writes; original scoring/timing remains. Upstream destination replay fix is covered. |
