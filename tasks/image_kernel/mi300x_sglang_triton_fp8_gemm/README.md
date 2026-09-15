# mi300x_sglang_triton_fp8_gemm

Task-local runner for aiter_triton_fp8_gemm.

Optimizes the AITER Triton FP8 scaled GEMM kernel that lives in
`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8.py`. The public entry
`aiter.ops.triton.gemm.basic.gemm_a8w8.gemm_a8w8` computes
`Y = (X @ W^T) * (x_scale * w_scale) + bias` where X and W are FP8 tensors and
x_scale / w_scale dequantize the accumulator back to BF16. All GEMM work is done
by the single `@triton.jit` `_gemm_a8w8_kernel` (an optional
`_gemm_a8w8_reduce_kernel` only reduces split-K partials). This maps to the hot
op `aten::_scaled_mm` (FP8 scaled matmul). Editing the kernel file changes the
Triton source that is JIT-compiled at first call, so the agent's edits take
effect. This runner:
  - compile:     builds the op with a small FP8 call (smoke)
  - correctness: runs the Triton op vs a torch dequant+matmul reference
  - performance: benchmarks the Triton op, writes build/performance_report.json

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
