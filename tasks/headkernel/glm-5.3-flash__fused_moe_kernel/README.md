The current GLM-5.3-Flash run uses whole AITER fused MoE with FP8 blockscale weights. The historical SGLang Triton `fused_moe_kernel` does not appear in run `glm-flash-fp8-native-jit-194292`. This task is now an explicitly labelled replacement port for `aiter.fused_moe:fused_moe`.

The original image supplies a precompiled fused GPU code object. Its dispatch and layout sources are pinned in `provenance/NATIVE-SOURCES.json`; an editable body for the observed code object was not found in the extracted source. The new Triton implementation is therefore a port, not a stock-source optimization task. The editable implementation remains the complete replacement port. Primary scoring compares its matched GPU time with pinned native production; frozen-port versus edited-port improvement remains secondary.

The protected runner includes native Opus routing sort, input FP8 quantization, gate/up matrix multiplication with physical shuffled weights, SiLU and intermediate FP8 quantization, down projection and weighted expert reduction. Only the five GPU function bodies are editable. Every host operation, launch setting, fixture, scalar and oracle is frozen.

The original native profile establishes prefill M8192 and decode M64, H4096, I256,288 experts and top-k8. Its per-rank observed counts are336 prefill and42 decode; the decode evidence covers one explicitly linked graph and is not extrapolated. The full-workload importer retains every supported captured MoE case, including structural tail variants, and uses its actual per-rank notification count. The finalized 64-request capture `glm-final-capture-v4-194841` supplies both real cases: M8192 prefill with 2688 notifications per rank and M64 decode with 43008. No structural tail variants occurred. Both cases and their real input/output bytes are mandatory. `scripts/import_fixtures.py` checks the completed native capture, every public scalar and tensor layout, copies raw blobs with hashes, and builds `cases.json`. It fails on missing core stages or an unsupported ABI. No synthetic routing or historical fixture is substituted.

The selected native one-stage path does not forward public `swiglu_limit=10.0` into its fused code-object call. The port follows that unclamped SiLU path. Input quantization and scales matched native bytes in the earlier bounded M64 diagnostic; see `provenance/DIAGNOSTIC-EVIDENCE.json`.

The candidate and frozen reference use the native scale-product/FMA order in both matrix stages. SiLU uses `exp2`, a single hardware reciprocal instruction, and ordinary multiplies. Intermediate FP8 quantization floors the absolute maximum at float32 `1e-6`, computes its multiplier as `448 * hardware_rcp(maximum)`, and computes the dequantization scale as `hardware_rcp(multiplier)`. The source guard admits only the exact pure float32 single-instruction reciprocal form. An earlier opaque multi-instruction SiLU block produced nonrepeatable outputs and remains excluded.

The supplied source passed native comparisons on both M64 and M8192 with four replay blocks, 440 graph replays, and 444 output verifications, retaining rtol=atol=0.02, ten warmups, and 100 measured replays per block. The original and corrected primitives also passed repeatability and quantization checks on the same saved gate/up tensor. These are source-pinned diagnostics; a fresh framework task-validator PASS remains required. The earlier validator failures on unrecorded random challenges have not been reproduced or claimed explained. The host launch grid, existing matrix instruction dimensions, and FP32 route reduction remain unchanged. See `provenance/NATIVE-ARITHMETIC-REPAIR.json`; `provenance/ACTIVATION-ROUNDING-REPAIR.json` preserves the prior arithmetic experiment.

For each replay the runner permutes actual routing rows and creates fresh numerical hidden activations with per-row multipliers and per-element RMS-scaled noise. The captured weights and routing remain the starting data. The CPU input truth is fixed before the candidate runs. Verification snapshots candidate outputs and every immutable input to CPU first, then computes the pinned native MoE reference and compares the CPU observations. No current native golden output exists on GPU before candidate snapshots. Pure output/intermediate poisoning, input checks and fresh references apply to every10 warmups and100 measured replays. Fixed-output lookup or row-permutation-only solutions cannot satisfy this replay contract.

Run `python3 scripts/import_fixtures.py CAPTURE_MANIFEST` after capture. Then run the compile, correctness and performance commands in `config.yaml`, followed by the generic trusted-host retest and framework task-validator. The real fixtures have been imported and `NOT_BUILT` removed. Fresh GPU/framework reports for this exact final contract are still required.

Every performance phase also compares the complete candidate port with the pinned native `aiter.fused_moe` on each exact captured case, using identical token permutations, output checks and timing policy. Native output/workspace allocations occur during graph construction; no extra copy is included in its timed graph. `native_production_comparison.json` reports raw samples, native parity and `native_mean_ms/candidate_mean_ms`. The primary Arena score uses the matched native-production comparison. A port gain alone does not establish native improvement; only a native ratio above1 supports that isolated claim, and serving speedup requires a model rerun.

The protected production comparator also writes a unique `build/native_production_events-*.jsonl` receipt containing every case, leg, seed, iteration, and verification result. Reset and verification receipts are outside device timing. Failures are recorded and propagated unchanged; the final comparison report binds the challenge seed and receipt digest. This makes a fresh-challenge failure reproducible even when the final comparison report is never completed.

After importing the finalized real capture, keep data out of the source commit and use the repository trusted-fixture contract: commit `fixtures/EXTERNAL-MANIFEST.json` plus `cases.json` references, with `trusted_evaluation.fixture_manifest` pointing at that manifest. Case metadata stays `served-tensor-fixture-v1`; raw bytes stay `raw-storage-segment-v1`. The original frozen rank0capture can serve as the local mirror because object keys may differ from the task-relative renamed case files. Use trusted host `--stage-only` for the framework validator input and `--fixture-local-mirror` for fresh retests. Neither the bounded diagnostic nor fixture staging qualifies the task.

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
