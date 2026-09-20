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
| SGLang v0.5.17 | 8 | CPU checks; GPU qualification not run |
| SGLang v0.5.18 | 7 | All seven stopped at the AITER import failure; cache fix requires GPU retest |
| Kimi K3 capture image | 3 | CPU checks; GPU qualification not run |

There are currently **no successful native GPU correctness/performance reports
and no framework-finalized task-validator PASS reports** for this release.
The separate MiniMax generated-input draft is not part of this branch; the
14 capture-dependent tasks still require their declared fixtures.

Native qualification remains required for every registered task:

- Matching image and actual MI355X/gfx950 runtime identity.
- Complete physical shape and layout evidence, including runtime transformations.
- Successful compilation and correctness over every declared case.
- Validated output from the actual timed replay, with preserved state and ABI.
- Scoreable, consistent timing with no missing or failed cases.
- A fresh framework-finalized task-validator report with `overall_status: PASS`.

Results will be recorded against the exact task and harness versions. A partial,
preempted, skipped, or failed run cannot be counted as a successful validation.
