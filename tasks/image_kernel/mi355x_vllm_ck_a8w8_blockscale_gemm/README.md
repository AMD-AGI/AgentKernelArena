# mi355x-ci-a8w8-blockscale-gemm-20260720

Self-contained image_kernel harness generated from Hyperloom 2026-07-20 MI355X sessions. See session_cases.json for exact provenance, shapes and dtypes.

## Effective task instructions

Optimize a8w8_blockscale_gemm on MI355X/gfx950. The harness cases are parsed from Hyperloom 2026-07-20 sessions and stored in session_cases.json. Preserve all correctness cases and improve CUDA-graph measured performance.

## Arena v2 contract

The candidate is the existing implementation in the declared image sources.
Its required final language and exact task-relative editable files are in
`config.yaml`; directory names do not select execution behavior. The framework
freezes this initial implementation into a separate baseline workspace. Both
roles run the same protected harness in their own workspace; an absent candidate
or missing image source is an error, never permission to use the installed copy.

Setup runs `python3 scripts/setup_task.py` after declared image materialization
and before baseline capture. It validates source paths and required build assets.
Do not edit `scripts/`, workload files or references. Additional source files
outside `candidate.editable` are dependencies, not editable implementation.
Preserve the original numerical gates, seeds, layouts, dispatch, state handling
and CUDA graph/event timing. `workloads.json` enumerates the complete manifest
independently of reported timings; `session_cases.json`, when present, retains
its original session provenance. Cases marked correctness-only are not scored.

Use the agent-neutral commands:

```bash
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Baseline commands run in the framework's frozen workspace. Each command emits
one `ARENA_EVAL_RESULT=` envelope. A failed dependency, dispatch or output contract
is a failure, not an accepted baseline numerical diagnostic. The original
`task_runner.py` remains the protected operator implementation of these checks;
its generated performance region must be materialized by Arena. Developer
profiling drivers do not supply final evaluation evidence.
This migration has CPU regression coverage; formal GPU task validation and the
optimization campaign are coordinated separately. Runtime source availability
must be checked against the selected immutable image, not inferred from a tag.

The original reduced-size correctness checks are retained; scored-size checks use the same tolerances in addition, so the manifest never declares untested performance shapes as covered.

HIP evaluation uses a fresh task-local JIT directory per action. The runner
requires a successful compilation whose inputs include a declared candidate
translation unit or template header. It records the covered files and rejects
unrelated/precompiled dispatch. This is build-source evidence, not exhaustive
proof that every launched GPU instruction belongs to every editable file.
No compiler-triggered repository cloning or checkout resets are permitted.
