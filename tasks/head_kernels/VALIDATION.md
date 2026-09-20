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

## Latest verified selection

The [native-verified index](native_verified.json) now contains **five tasks**:
both GLM GEMMs, Qwen dense GEMM, Qwen RMSNorm, and Qwen paged attention.
Qwen paged attention completed all three native phases in job **159222** at
source `113a0900`, including its one declared `decode_m64_live` benchmark case,
10 warmups, 100 samples, input restoration and exact timed graph replay checks.
Its 35 materialized task-file identities match the current source. The verified
archive hash is `f31fcb152982a5fdd419dfb234d256c630f70d04aa3ac8fefddd4e7c88c980d1`.
This qualifies the archived compacted case; it does not recover the original
full-pool addressing or the other uncaptured serving signatures.

Jobs 159221/159222 used fresh two-hour requests accepted by the authorized
preemptible QoS. Both were cancelled by scheduler preemption after 20 minutes,
not by the task phase timeouts. GLM copy, Qwen MoE and Qwen recurrent state passed
correctness but did not finish performance; they remain outside the index.
DeepSeek sparse attention and MiniMax sparse prefill did not complete correctness.
The two DeepSeek MoE workers were stopped separately after verification showed
that 73 required M1920 sequence calls have no captured inputs. These are not
promoted by skipping the missing calls. All jobs are terminal and owned-container
cleanup was confirmed.

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
- DeepSeek defines four observed attention benchmark cases and five per MoE task.
  Both MoE tasks now fail CPU input-completeness checks: their mandatory 256-call
  sequence lacks 73 M1920 inputs. The original UT skipped those calls; the current
  task does not. All five known benchmark cases and the complete required sequence
  remain declared, and both tasks remain unverified. Derived M1 attention slices
  remain mandatory, unscored robustness probes. Consumed stage-1 scales and
  defined LSE values retain their numerical checks.
- Qwen retains partial captured coverage. Two MoE cases do not cover all four
  telemetry shapes, and one compacted paged-attention capture does not cover
  all observed request signatures. The fidelity report bounds each claim.

These restrictions fail closed. A blocked performance phase is not a zero-time
kernel, a passing task, or a speedup.

## Earlier campaign checkpoint (17:39 UTC): task-by-task status

**Final outcome: four tasks passed complete native correctness and performance;
one failed the independent numerical reference; thirteen did not finish full
correctness within the bounded allocations.** The audit and all-18-task attempt
matrix are complete. Full native qualification and exact serving equivalence are
not complete. Six aggregate workload scores remain disabled for missing or
estimated serving controls, overlapping the outcomes below.

Every one of the 18 tasks started native verification on MI355X. Each attempt
used a 12-minute ordinary burst allocation and an immutable source snapshot.
The lease guard reserved cleanup time and stopped unfinished work before lease
expiry. A retained task status of `running` after that stop means **incomplete**,
not a passing check. Syntax/fixed-interface checks passed for all 18 tasks.

The first complete matrix used jobs 159066 (`c0ed5a10`, v0.5.17), 159096
(`15f43104`, v0.5.18), and 159106 (`15f43104`, the three MiniMax tasks). The final
reruns use `06d0838e`: job 159134 covers both GLM GEMMs and the repaired MiniMax
decode packages; job 159135 covers RMSNorm. Qwen dense's passing job 159119 uses
`88c9bfa6`; its task files are unchanged by those later fixes.

| Model / task | Native correctness | Native performance | Evidence / disposition |
| --- | --- | --- | --- |
| DeepSeek sparse MLA | Incomplete at lease cleanup | Not run | 159066; requires a longer full verification window |
| DeepSeek MoE stage 1 | Incomplete at lease cleanup | Not run | 159066; requires a longer full verification window |
| DeepSeek MoE stage 2 | Incomplete at lease cleanup | Not run | 159066; requires a longer full verification window |
| GLM BF16 GEMM | **Passed 27/27** | **Passed 7/7** | 159134, final attested-worker path |
| GLM FP8 GEMM | **Passed 21/21** | **Passed 7/7** | 159134, final attested-worker path |
| GLM elementwise copy | Incomplete at lease cleanup | Not run | 159096; conditional producer-layout contract remains explicit |
| GLM fused MoE | Incomplete at lease cleanup | Workload score blocked | 159096; authentic serving controls are missing |
| Kimi grouped attention | Incomplete at lease cleanup | Workload score blocked | 159066; proxy context/pool mapping is unqualified |
| Kimi MoE stage 1 | **Failed independent single-launch reference** | Not run; score blocked | 159066; `prefill_M8192:single:0` failed, so a median cannot qualify it |
| Kimi MoE stage 2 | Incomplete at lease cleanup | Workload score blocked | 159066; distinct chunk scenarios and estimated decode controls remain explicit |
| MiniMax decode score | Incomplete at lease cleanup | Not run; workload score blocked | 159134; repaired dependency and native preflight passed |
| MiniMax GQA sparse decode | Incomplete at lease cleanup | Not run; workload score blocked | 159134; repaired dependency and native preflight passed |
| MiniMax GQA sparse prefill | Incomplete at lease cleanup | Not run | 159106; full observed M8192 addressing is retained |
| Qwen dense BF16 GEMM | **Passed all five contracts** | **Passed 5/5** | 159119; wrong-output source control 159126 rejected |
| Qwen fused MoE | Incomplete at lease cleanup | Not run | 159096; only two persisted composite cases are claimed |
| Qwen recurrent gated delta | Incomplete at lease cleanup | Not run | 159096; full retained 199-slot state geometry is preserved |
| Qwen RMSNorm | **Passed** | **Passed 2/2** | 159135, final attested-worker path |
| Qwen paged attention | Incomplete at lease cleanup | Not run | 159096; archived compacted-pool scope remains explicit |

The MiniMax packaging correction bundles the **exact** 30,761-byte captured
prefill module, SHA-256
`bf26c1d9ad6ad6d7716727c1ab55d09a786abf59a2cd8a7a390b90285f1e8421`,
in both decode baseline overlays. Missing or modified dependencies and failed
injections now fail immediately and cannot leave a partially initialized module
behind. This repair does not invent the missing serving controls or reopen either
blocked workload score.

The source, configuration, reference and materialized-helper fingerprints of all
four passing tasks matched the final repository tree exactly. The positive Qwen
run and its zero-output control used the same fixed five-case contract. The
negative passed dispatch, callable and source-engagement gates; eager, randomized
and graph numerical parity rejected its outputs. Performance was not run after
rejection. All final jobs are terminal and their evidence archives are hash-verified.

| Job | Evidence archive SHA-256 |
| --- | --- |
| 159134 | `8249b4d4ab91569bf60eacf8d7a1f7472e58a838ea3a7a8c849cf43deb1ae1a2` |
| 159135 | `7592e2dff048be5341ed14eabde31973f5db81651c48cd181d3007b835ea27ca` |
| 159119 | `8cbbd519fdbde8d05dcd6f2d1b937e71ac7f3be6a62968e1e7644f6290755e76` |
| 159126 | `232828b93e27c8d512629a95569e7a1e23ce766b61831c97580d13c85e165ac5` |

The qualification verdict remains **incomplete** wherever a full command did not
finish. None of these operator results certifies complete original serving
execution, a model end-to-end gain, or a framework task-validator PASS.

## Earlier GPU checks and source-binding diagnosis

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
valid candidates may change their backend. Job 159119 subsequently passed all five correctness contracts and all five
performance cases at `88c9bfa6`, with 10 warmups, 100 samples per case, exact graph
replay and state-restoration checks. Job 159126 supplied a separate wrong-output
source control; its correctness failure and skipped performance are retained.
The negative edit is confined to the editable GEMM body.

## CPU and integrity checks

The final combined CPU run at `5b19d604` passed **785 tests and 60 subtests**
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
All 18 tasks now declare trusted worker modules. The last three legacy tasks
(GLM BF16/FP8 and Qwen RMSNorm) gained declarations and reuse of the preloaded
objects in `06d0838e`; nine actual-worker CPU controls cover unchanged execution,
case-function replacement and case-module alias replacement. Jobs 159134/159135 passed native correctness and performance on the new
worker path. The nine new actual-worker controls also passed an independent
rerun in 13.42 seconds on `5b19d604`.
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
