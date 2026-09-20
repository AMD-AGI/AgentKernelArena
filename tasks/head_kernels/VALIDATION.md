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

Native qualification remains required for every registered task:

- Matching image and actual MI355X/gfx950 runtime identity.
- Complete physical shape and layout evidence, including runtime transformations.
- Successful compilation and correctness over every declared case.
- Validated output from the actual timed replay, with preserved state and ABI.
- Scoreable, consistent timing with no missing or failed cases.
- A fresh framework-finalized task-validator report with `overall_status: PASS`.

Results will be recorded against the exact task and harness versions. A partial,
preempted, skipped, or failed run cannot be counted as a successful validation.
