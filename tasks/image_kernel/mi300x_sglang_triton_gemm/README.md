# mi300x_sglang_triton_gemm

Task-local runner for aiter_triton_gemm.

Optimizes AITER's Triton A16W16 GEMM op `aiter.ops.triton.gemm.basic.gemm_a16w16`
(hot op `aten::mm`). The op computes ``Y = X @ W^T`` and dispatches to the
``@triton.jit`` kernels defined in
``aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py``:
  - ``_gemm_a16_w16_kernel``        (the main blocked / split-K matmul)
  - ``_gemm_a16w16_reduce_kernel``  (split-K partial reduction)
Editing that kernel file re-triggers Triton JIT compilation, so agent changes
take effect. This runner:
  - compile:     builds/launches the op with a small case (smoke)
  - correctness: runs the Triton op vs a torch.matmul reference (assert close)
  - performance: benchmarks the op and writes build/performance_report.json

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
its generated performance region must be materialized by Arena. Optional
profiling does not supply final evaluation evidence. Agent CLI adaptation belongs
to the agent integration; use the declared v2 runner for task evaluation, with
the task's full numerical and workload checks.
This migration has CPU regression coverage; formal GPU task validation and the
optimization campaign are coordinated separately. Runtime source availability
must be checked against the selected immutable image, not inferred from a tag.

Formal validation requires checking the exact captured timing invocation. After
measurement the harness negates one data input in place, poisons the captured
output with NaNs, replays that invocation, and compares with a freshly computed
reference using the unchanged ordinary-correctness tolerance. This adds no work
to the measured region and preserves the original warmups and sample counts.
An unobservable event fallback cannot provide this evidence and fails explicitly.
