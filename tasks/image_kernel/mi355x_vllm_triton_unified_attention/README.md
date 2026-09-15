# mi355x-ci-unified-attention-20260720

Self-contained image_kernel harness generated from Hyperloom 2026-07-20 MI355X sessions. See session_cases.json for exact provenance, shapes and dtypes.

## Effective task instructions

Optimize unified_attention on MI355X/gfx950. The harness cases are parsed from Hyperloom 2026-07-20 sessions and stored in session_cases.json. Preserve all correctness cases and improve CUDA-graph measured performance.

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

The qualified SGLang runtime stores AITER as a complete source repository at
`/sgl-workspace/aiter`, not as an installed `aiter_meta` wheel directory.
`workspace.sources` explicitly copies that repository to the task's metadata
root; the unified-attention task separately copies its `aiter/` Python package.
Editable task-relative paths and operator semantics remain unchanged. This fixes
source availability only; dispatch, compilation and numerical compatibility
still require full GPU validation on the selected immutable runtime.
