# Kimi-K3 FlyDSL A8W4 stage 1

This task optimizes the SGLang 0.5.20 FlyDSL `flydsl_moe_stage1` implementation with the declared async-LDS synchronization repair on MI355X (`gfx950`). It covers both decode and prefill from the verified `kimi-served-194550` workload: 64 requests, ISL 8192, OSL 1024, concurrency 64 and TP 8. The capture and external fixture inventory are sealed. **The targeted 16-trial decode reproduction passes with the repair; full three-case GPU and framework qualification are pending.**

The supplied baseline includes a correctness repair: the partial memory wait before the async-LDS ping-pong barrier is replaced with `rocdl.s_waitcnt(0)`. Original native outputs varied in aligned fragments and violated even the unchanged dequantized 0.02 tolerance. The repaired candidate and frozen reference produced identical outputs in all 16 targeted trials. This baseline is not byte-identical to the stock image. [The repair provenance](provenance/STAGE1-SYNC-REPAIR.json) pins original image, projected source and repaired source hashes; the runtime verifies the original image and the exact declared patch separately.

The operator consumes FP8 E4M3FN activations, packed FP4 E2M1 weights and tiled E8M0 scales. It fuses gate/up GEMMs, interleaved SiTUv2 with beta 4 and linear beta 25, and FP8 output quantization. The native output includes an FP8 payload and tiled E8M0 scales. [cases.json](cases.json) fixes every shape, stride, storage offset, alias, scalar and observed work distribution.

| Stage | Activation shape | Payload shape | Calls per rank | Calls across TP 8 |
|---|---|---|---:|---:|
| Decode | `[64, 3584]` | `[64, 16, 384]` | 94,208 | 753,664 |
| Prefill | `[8192, 3584]` | `[188416, 384]` | 184 | 1,472 |
| Prefill | `[16384, 3584]` | `[319488, 384]` | 2,852 | 22,816 |

The three structural classes account for 777,952 calls across all ranks. Each frequency is counted once. The latest empirical work histograms come from the compatible `kimi-actual-v4-no-stack-194550` workload, recorded in [STAGE1-WORKLOAD-V4.json](provenance/STAGE1-WORKLOAD-V4.json). Numerical control variants contribute to each class's empirical work histogram; they do not create duplicate score cases or duplicate frequencies. Standard Arena scoring uses the arithmetic mean of the three matched per-case speedup ratios.

## Editable source boundary

Only `_emit_moe_gemm1` and `moe_gemm1` bodies in [mixed_moe_gemm_2stage_common.py](source/flydsl/kernels/mixed_moe_gemm_2stage_common.py) are editable. [The source policy](ut/source_guard_policy.json) freezes host compilation and launch code, signatures, decorators, imports, other functions and every harness/reference file. The A16W4 and stage-2 branches remain frozen dependencies.

The guard permits the reviewed DSL operations, arithmetic/control flow and local arithmetic helpers. It rejects arbitrary Python calls, runtime introspection and object/module mutation. [SOURCE-PROVENANCE.json](SOURCE-PROVENANCE.json) records the complete 119-file source projection; the candidate and frozen reference load through private package namespaces with independent closures. The runtime, source guard and generic evaluator all use the same frozen `ut/baseline_src/flydsl/` tree. The adapter checks the original source hashes against the pinned runtime image.

## Fixtures and correctness

The oracle uses three actual served raw fixtures and an independent frozen native source closure. Both implementations must first agree with captured CPU golden values. Generated trials then refresh legal activation values and routes at empirically observed work amounts, and change packed FP4 weight signs while preserving magnitudes and group scales.

The protected runner saves input-storage truth on CPU, executes the candidate, checks metadata and immutable inputs, captures outputs on CPU, and only then computes the reference. Live FP8 payload values retain tolerance 0.02 with the RMS absolute floor. Live E8M0 scale bytes must match exactly at their physical offsets. The graph benchmark retains 10 warm-up replays and 100 measured replays per case, with fresh inputs, output poisoning and validation on every replay. Correctness includes seeds 0, 1 and 2 plus no-op and wrong-output controls. [STAGE1-CONTRACT.md](ut/STAGE1-CONTRACT.md) explains storage layouts, defined output regions and work generation.

[EXTERNAL-MANIFEST.json](fixtures/EXTERNAL-MANIFEST.json) pins 29 external assets totaling 1,666,534,545 bytes. The source commit contains the inventory and exact case references, not the raw blobs. Use the [trusted fixture materializer](../../../docs/how-to/trusted-fixture-artifacts.md) on the host with a verified local mirror or the explicitly approved pinned OCI prefix. The manifest's OCI prefix is planned until qualification and publication are completed.

After materializing the complete task, run these entrypoints inside the pinned Docker image on an allocated `gfx950` GPU:

```bash
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

The compile phase invokes and validates the real specialization for every case. Failure leaves no scoreable success report. Performance emits one separate result for each mandatory structural case. No legacy unit-test or timing fallback is used.

For submitted-source adversarial checks, `scripts/make_submitted_controls.py --output NEW_DIRECTORY` prepares no-op and zero-payload candidate sources that pass the source boundary. Native correctness must reject both, possibly during the compile phase's captured-golden check. The protected task's actual math and tolerances are shared across stock and submitted-source checks.
