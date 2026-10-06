This task implements all 14 FP8 blockscale BP GEMM records from the finalized GLM-5.3-Flash capture `glm-final-capture-v4-194841`: 64 requests, ISL8192, OSL1024, concurrency64, TP8 and context9218. Each distinct notified record remains mandatory, including prefill and decode. Exact shapes, strides, dtypes, scalar controls, served geometry and actual per-rank occurrence counts are committed in `cases.json`. There are seven N/K geometries in both M8192 prefill and M64 decode. B retains the physical AITER (16,16) shuffle; SA has strides [1,M] and SB contains row-major block128 scales.

The supplied source is a newly authored Triton replacement port. Its frozen reference and edited candidate always launch the same declared `gemm_kernel` with identical host dispatch, allocation, correctness and timing boundaries. Only the GPU function body is editable. Imports, decorators, signature, launch geometry, fixtures and all harness files are protected. Source bytes never select a native algorithm. The primary Arena score measures pinned native-production time divided by candidate-port time; local port improvement is retained separately.

Real rank0 operand representatives are pinned through `fixtures/EXTERNAL-MANIFEST.json`; all eight ranks contribute finalized ABI/count evidence. Import rejects missing notified records or unsupported controls and never silently filters shapes, stages or dtypes. Raw metadata and storage segments stay outside Git and are materialized from the frozen mirror by the trusted host. Use the repository trusted-fixture workflow and `--stage-only` to prepare framework input. The planned OCI prefix is metadata, not evidence of an upload.

Correctness uses generated numerical inputs on even seeds and fresh per-element perturbations of real captured activations on odd seeds. Live challenges preserve actual weights and scale layout, use per-row factors0.75..1.25 and5% RMS noise, and recompute the independent CPU FP32 reference. FP8 values are clamped and converted back to E4M3FN. Fixed captured output or row-permutation-only answers cannot satisfy this contract. Expected outputs stay on CPU. Every replay refreshes inputs, poisons output, verifies every output, and checks input immutability. The tolerance remains rtol0.01/atol0.02.

All cases use10 warmups and100 measured replays. The protected performance entrypoint also compares the actual native callable against the candidate on the same fresh numerical sequence and every real ABI. Wrapper records call pinned `aiter.tuned_gemm.gemm_a16w16`; direct records call pinned-schema `torch.mm`; FP8 records call pinned `gemm_a8w8_blockscale_bpreshuffle` with the actual shuffled attribute. Native out=None APIs retain their own output and workspace allocated during graph capture, and no output-copy kernel is added to timing. `native_production_comparison.json` contains raw native and port samples and `native_ms/candidate_ms`, which supplies the explicit native-baseline score. Only a ratio above1 supports isolated native-operator improvement; serving gain still requires an end-to-end rerun.

This final expanded real-capture contract requires fresh GPU compile/correctness/performance, trusted replay and framework task-validator qualification. Earlier generated-only or smaller-contract passes are historical evidence and do not qualify it. CPU dispatch and fixture tests validate contract mechanics without claiming GPU arithmetic results.

Generated FP8 challenges include all254 finite E4M3 codes, raw standard deviation64 and every checkpoint-observed weight-scale exponent from-12 through-7 in every case. This retains the scale-threshold negative-control coverage.

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
