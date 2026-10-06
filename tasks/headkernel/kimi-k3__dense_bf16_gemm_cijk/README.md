The finalized capture `kimi-original-heads-v2-194958` supplies40 mandatory dense cases:26 native-wrapper and14 direct-ATen records, all BF16. Three direct cases preserve actual padded activation row strides144 or6288; the decode view retains storage offset6144. Exact per-rank whole-workload notification counts are committed in `cases.json`. No dtype/layout record was dropped.

This is the prepared current SG0.5.20 Kimi-K3 dense GEMM replacement task. It remains unqualified pending fresh GPU phases and framework validation. The real case manifest and fixtures have now been imported. The seven-case Kimi attention/MoE capture does not cover this head. Historical marker and README are preserved under `provenance/` and do not qualify the current image.

`CASE-REQUIREMENTS.json` records18 sampled BF16 wrapper signatures: nine N/K geometries at M8192 and M16384. Their counts belong only to sampled profile windows. Decode GPU symbols show the head remains active, but complete decode arguments and full-workload frequencies require new capture. Every actual wrapper and direct ATen dense record must be retained, including additional shapes, dtypes, offsets, aliases and controls. Unsupported cases stop intake instead of disappearing.

The native wrapper selects a solution from `aiter.tuned_gemm.solMap`, which holds cached function objects. A module-only setattr is a dead binding. `ut/native_dispatch.py` provides a scoped replacement of the solution dictionary and an exactly-once invocation check; the protected runner calls the original wrapper and proves that it reaches the submitted GPU port. It restores every native entry after the call. The matched native-production comparison uses those original entries. CPU regression tests demonstrate both the dead module-only binding and the effective dictionary binding. GPU source controls must subsequently prove actual edited kernel behavior.

The editable implementation is a newly authored Triton replacement port. The native Cijk/Tensile code object itself is not claimed as editable source. Frozen reference and candidate use the same declared port and binding. Correctness and initial case checks compare against CPU FP32 math and also cross-check an independent GPU FP32 reference with TF32 disabled. Timed replays freeze candidate outputs and all immutable input observations on CPU before computing that GPU FP32 reference. No current golden output is available on GPU before candidate observation. Fresh numerical challenges, output poisoning and input checks remain mandatory. All10 warmups and100 measured replays are validated at unchanged rtol0.01/atol0.02. Native comparison uses every real case and the same fresh numerical sequence, retains native capture-time output allocation and reports native_ms/candidate_ms as the primary score, with the frozen-port ratio retained separately. Serving gain still requires end-to-end validation.

The capture integration contract is `capture/SPEC.json`; `capture/dense_bindings.py` supplies the typed owner adapter for the shared runtime recorder. The shared capture owner must install its wrapper before consumers cache aliases, suppress nested ATen duplicates, and bind graph construction/replay notifications to actual served work. This task does not launch or modify a serving capture. `scripts/import_live_operands.py` requires explicit finalized capture-ready and rank0 stop receipts; the old sampled profile cannot satisfy it.

After actual intake, prepare a trusted external fixture manifest and commit its references, preserving raw captured bytes outside Git. Then run all native phases, submitted-source controls, trusted replay and the framework task validator. Neither this CPU preparation nor a fixture stage is a qualification result.

The CPU intake audit found39 of40 native outputs within the FP32 bounds. Decode M64/N2112/K7168 uses pinned FlyDSL split-K7: FP32 partials convert to BF16, then paired BF16 atomics round after each addition. A fresh GPU probe ran ten native repetitions; every pair outside the original mathematical bounds exactly matched one legal paired arrival order. Native calibration now checks the whole output at rtol0.01/atol0.02, then requires bit-exact, input-derived arithmetic proof for every exception pair. Both lanes must share one legal order. Candidate checks stay strictly referenced to FP32 math. This models native numerical conformance without claiming pointwise candidate/native equality. See `provenance/NATIVE-PRECISION.json`; a fresh complete40-case framework run remains required.

`ut/storage_contract.json` binds every case to the original complete A/B/C storage extents. Reset copies every input byte. Unused prefixes, row gaps and tails receive nonzero per-replay canaries; verification snapshots and compares full backing storage before any GPU reference. Output bytes outside the declared view are also guarded. The regression reproduces a guard-admitted `tl.store(A + K, 0, ...)` on all three real padded ABIs and proves the protected verifier rejects it even though logical tensor values are unchanged.

The frozen `scoring_baseline` policy opts this task into native-production scoring.
The host validates the enclosing request, current source hashes, native source
provenance, runtime image, complete case ABI, raw samples, and replay checks for
both existing comparison legs. Means and ratios are recomputed from samples.
Missing, stale, partial, or diagnostic-only evidence cannot fall back to the
port score. No benchmark calls, cases, seeds, tolerances, or timing boundaries
change for this policy.

`task_result.yaml` and `trusted_measurement.json` carry
`baseline_kind: native_production`, the primary native/candidate ratio, and the secondary
`port_to_port_speedup_ratio`, every case ratio, `regressed_case_ids`, and
`all_cases_faster_than_native`. `production_kernel_improvement` is true only when
the primary aggregate ratio exceeds one; individual regressions remain explicit.
A slower correct starter is still valid and measurable. This is an isolated
operator metric and does not assert end-to-end serving improvement.

Candidate outputs retain the pointwise BF16 check (rtol=0.01, atol=0.02) and must also satisfy a scale-relative global accuracy requirement: CPU FP64 Frobenius error divided by the reference norm is at most 2/255. This additional check has no absolute floor; an exactly zero reference requires exactly zero output. The budget uses the precision scale of two final BF16 roundings, independently of observed failures. The pointwise check continues to reject localized corruption. Native precision calibration and its exact source-derived exceptions remain confined to the native production leg. See `provenance/CANDIDATE-PRECISION.json`. The stronger candidate contract requires fresh stock calibration, full framework validation, and independent trusted qualification; earlier results remain scoped to the pointwise-only contract.
