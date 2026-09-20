# Validation status

This is a reviewable, portable suite of **18 operator tasks**, not a fully
qualified benchmark release. All default inputs are generated locally from
committed code and compact structural metadata. **No external tensor archive,
private registry, or cluster filesystem is required.** Optional historical
archives are listed separately in `artifacts.json`.

Native GPU execution, supported workload geometry, and framework validation
are separate requirements. A successful microbenchmark does not establish
full HyperLoom serving equivalence or reproduce a historical end-to-end gain.
The [workload fidelity report](WORKLOAD_FIDELITY.md) records those distinctions.

## Current acceptance restrictions

- MiniMax decode-score and sparse-decode workload scores are blocked: their
  timing contexts and addressing were reconstructed, not captured per call.
  Their correctness and replay probes remain available.
- MiniMax sparse prefill uses the observed M8192 record, including its full
  `[4097,11268]` request table, original slot `[4]` with int64 dtype, captured
  block indices, and original full KV allocations. This needs fresh GPU checks.
- GLM BF16 and FP8 GEMMs retain 27 and 21 mandatory correctness cases, but each
  scores only seven observed M64 cases. The other combinations are unscored
  generalization checks.
- GLM fused-MoE workload scoring is blocked. The original archives contain
  M19/M1/M8192 records; M64/M16384 were counters only. The snapshots were taken
  after an in-place call. Generated routing cannot replace missing pre-call
  serving captures.
- All three Kimi aggregate workload scores are blocked. Attention uses inferred
  context/pools; stage 1 uses second-hand routing and an assumed decode variant;
  stage 2 has estimated decode control weights. All semantic cases remain.
  Stage 1 additionally requires three individual launches at each of three
  shapes to pass an independent reference; a median cannot hide a bad launch.
- DeepSeek scores four observed attention cases and five cases per MoE task.
  Derived M1 attention slices remain mandatory, unscored robustness probes.
  The checker also validates consumed stage-1 scales and defined LSE values.
- Qwen retains partial captured coverage. Two MoE cases do not cover all four
  telemetry shapes, and one compacted paged-attention capture does not cover
  all observed request signatures. The fidelity report bounds each claim.

These restrictions fail closed. A blocked performance phase is not a zero-time
kernel, a passing task, or a speedup.

## Completed native GPU evidence

All entries below used MI355X/gfx950 and pinned public `rocm/hyperloom` images.
The exact image manifest and config digests are in the
[environment matrix](../../docs/reference/top5-head-kernel-environments.md).
Reports were retained with their hashes and source revisions.

| Job | Source revision | Task / scope | Native result |
| --- | --- | --- | --- |
| 158785 | `97555eb2` | GLM BF16 GEMM | 27/27 correctness and 27/27 performance cases passed; 10 warmups and 100 samples per case. This predates the seven-case scoring restriction. |
| 158821 | `97555eb2`, isolated negative-control edit | GLM BF16 GEMM | Zeroed output from the editable function was rejected by the numerical check (error 0.9948567748069763); performance was skipped. |
| 158838 | `97555eb2` | GLM FP8 blockscale GEMM | 21/21 correctness and performance cases passed. This predates the seven-case scoring restriction. |
| 158846 | `97555eb2` | Qwen RMSNorm | Correctness and both performance cases passed. |
| 158846 | `97555eb2` | Qwen dense GEMM | Configuration-selection check failed; performance was skipped. |
| 158923 | `49684b76` | GLM BF16 baseline device trace | All seven observed cases produced actual GPU events. The retained full symbol matched two cases; the old summary lacks per-case mappings and grid/block dimensions, so complete dispatch equivalence is unproved. This was a trace diagnostic, not full correctness. |
| 159038 | `908879dc` | Qwen dense GEMM after configuration repair | All five expected configurations and GPU symbols appeared, but every backend-hook count was zero. Source binding and correctness were not certified; performance was skipped. |

The successful GLM public runtime reported SGLang 0.5.17, PyTorch
`2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`, and Triton 3.6.0.
A newer source revision requires its own qualification; earlier passes are
not silently transferred to later contract changes.

The Qwen dense source-binding investigation found two concrete integration bugs:
AITER's duplicate operator-registration decorator discarded the later candidate
function, and the overlay builder overwrote the corrected binding template with
its old generic literal. The loader now binds the actual source body, and the
builder uses that canonical template. Baseline dispatch fidelity remains checked;
valid candidates may change their backend. A fresh GPU positive and wrong-output
negative control are still required for this final correction.

## CPU and integrity checks

The final combined CPU run at `e3afdbf6` passed **770 tests and 60 subtests**
with no failures or skips. It covers the task catalogs, generated inputs,
source binding, fixed interfaces, runtime identity, native-verifier aggregation,
and the tested integrity controls. Earlier integration-test failures were
corrected before this complete run. The later Qwen overlay-builder correction passed
32 focused integration tests, including the actual generated-overlay path.
The Docker runner's mocked runtime,
agent-selection and isolation checks, and `make check-perf-helpers`, also passed.

Actual-worker adversarial controls independently rejected replacement of trusted
checkers and output encoders, module-alias substitution, and forged Tensor-to-NumPy
serialization. Integer and boolean outputs use exact comparison. Numerical
references run independently and are compared in protected parent memory.
The guards preserve the same attested helper modules across preflight, candidate
loading, correctness, and timing. These are bounded tested protections; the
worker process is not an operating-system security sandbox.

The canonical benchmark retains its warmup/sample policy, actual timed replay
validation, changed-input probes, state restoration, and fixed callable contracts.
Its graph timing and equal-case aggregation differ from some original eager
capture and serving-frequency regimes. Those differences are documented rather
than labeled exact serving performance.

## Release gate

There are **no framework-finalized `task_validator` PASS reports** for this
release. Direct `verify` and `parallel-verify` run the task's native commands
without LLM credentials and retain their real reports; they do not fabricate a
framework validator result.

Every enabled task needs complete native correctness/performance evidence and a
fresh framework-finalized `validation_report.yaml` with `overall_status: PASS`
before PR submission. A timeout, partial report, blocked score, or historical
pass does not satisfy that gate. See the
[validator contract](../../docs/how-to/task-validator.md).
