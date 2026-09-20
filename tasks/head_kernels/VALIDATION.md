# Validation status

This branch is undergoing native GPU qualification. It is **not yet a validated
benchmark release**. Publishing this reviewable branch does not imply that all
tasks have passed GPU correctness, performance, or the task validator.

The latest root verification on 2026-09-20 completed **449 CPU tests and
44 subtests**, with **21 explicitly skipped tests requiring unavailable host
PyTorch functionality**. There were no failures. The checks cover workload
layout, fixture integrity and publication, fixed interfaces, source-to-harness
binding, graph-failure handling, runtime image selection, protected imports,
known timer/comparator attacks, and forged or incomplete benchmark reports.

The 17 required external fixtures (33,139,788,731 bytes) have been copied into
an independent owned mirror and their complete SHA-256 hashes verified. The
fixture preparation guide describes how to populate a task-local copy.

## First allocated runtime attempt

On 2026-09-20, burst job `158486` started seven separate Docker task workers
on MI355X node `crsuse2-m2m-055`. The v0.5.18 image/config ID matched the
captured `sha256:760dd38b9b6f2bd11c13011d470eb8e377c3f0d71284a090a710d64a23bd789f`.
All seven Python syntax/fixed-interface checks passed. All seven correctness
commands then failed during the shared AITER runtime import: the ordinary user
could not read bundled FlyDSL cache files while AITER attempted to copy its
installed JIT directory. Performance was not run after those failures.

Commit `084368cf1ad16918dde53f7f227dcb69f2d6c22b` fixes this startup path using
the supported `AITER_JIT_DIR` and `FLYDSL_RUNTIME_CACHE_DIR` overrides and
separate writable worker caches. Its 31 focused runtime CPU tests and mocked
Docker checks passed. This is a tested environment correction, not a successful
GPU rerun. The node-specific retry was cancelled while pending after the
scheduler reported a GPU/resource allocation mismatch.

| Cohort | Registered tasks | Latest completed evidence |
| --- | ---: | --- |
| SGLang v0.5.17 | 8 | GLM BF16 native positive and negative controls completed at `97555eb2`; other tasks and later contract changes remain pending |
| SGLang v0.5.18 | 7 | All seven stopped at the AITER import failure; cache fix requires GPU retest |
| Kimi K3 capture image | 3 | CPU checks; GPU qualification not run |

There are currently **no framework-finalized task-validator PASS reports** for
this release. Native execution evidence is recorded separately below.
The separate MiniMax generated-input draft is not part of this branch; the
14 capture-dependent tasks still require their declared fixtures.

## GLM BF16 native GPU verification

At commit `97555eb20ffc7683ff151b9695b6f025b0386b34`, job `158785` completed on
an MI355X with the pinned public v0.5.17 image. All **27 correctness cases and
27 performance cases passed**, with the expected complete unique case set,
10 warmups and 100 samples per case. All device times were positive and finite,
and actual timed-graph replay, changed-input probing and state restoration
checks succeeded. This was direct verification, not the LLM task-validator
review or a serving end-to-end experiment.

The run observed SGLang `0.5.17`, PyTorch
`2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`, Triton `3.6.0`,
and `gfx950`. It verified manifest
`sha256:1f5464829559b086eb66f9b803cb9c7a817438c43edff2d5ef59b46a186745f6`
and config digest
`sha256:ffe4af630e49b05c812db4a468bfb411c3dbb0e93124801f28349bfa31352dea`.
The engine exposed the manifest digest as its image ID; the complete
manifest-to-config binding was verified. Native correctness/performance report
SHA-256 values are respectively
`d715d6cd57264ddb65cc2216e8fc8bc0e6ce8fe48dee12f229bbf99903109048` and
`f3ba78f36f27cb6b43c876705c0107a2ec8f80ea1ee7721d8cae4f43d51dc6c6`.

Job `158821` then tested a deliberately incorrect implementation in a separate
copy: only the editable `torch_gemm` body was changed to zero its normal output.
Compilation and runtime checks passed; candidate-versus-FP32-reference
comparison rejected it with error `0.9948567748069763`, and performance was
skipped. This demonstrates actual editable-source binding and numerical
rejection for the tested path. Both attempts completed ownership-scoped
container cleanup.

The scenario-fidelity audit subsequently classified only **seven** of the
27 BF16 combinations as supported by the retained workload trace. The other
20 remain mandatory correctness/generalization tests. Commit `9ac67071`
restricts scoring to the seven supported cases; the earlier GPU result does
not certify the later scoring contract or full HyperLoom workload fidelity.
The analogous FP8 task retains 21 correctness cases and scores seven observed
cases. Source kernel bodies and numerical tolerances were not changed by that
classification correction.

Native qualification remains required for every registered task:

- Matching image and actual MI355X/gfx950 runtime identity.
- Complete physical shape and layout evidence, including runtime transformations.
- Successful compilation and correctness over every declared case.
- Validated output from the actual timed replay, with preserved state and ABI.
- Scoreable, consistent timing with no missing or failed cases.
- A fresh framework-finalized task-validator report with `overall_status: PASS`.

Results will be recorded against the exact task and harness versions. A partial,
preempted, skipped, or failed run cannot be counted as a successful validation.
