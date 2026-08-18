---
myst:
    html_meta:
        "description": "Use the AgentKernelArena task_validator agent to run 12 deterministic and review-based quality checks before using GPU kernel tasks in shared experiments."
        "keywords": "AgentKernelArena, task validator, GPU kernel, quality checks, ROCm, HIP, Triton, validation report"
---

# Validate tasks in AgentKernelArena

The `task_validator` agent checks that tasks are correctly configured,
reproducible, and functional. It doesn't optimize kernels — it audits them.
Use it to validate new tasks before merging and to audit existing tasks before
using them in controlled comparisons or RL data collection.

## Run the validator

Save a run configuration such as `config_validator.yaml` with the validator as
the agent and the tasks to check:

```yaml
agent:
  template: task_validator
tasks:
  - hip2hip/gpumode/GELU
  - triton2triton/vllm/triton_rms_norm
  # - all                     # validate every task
target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

Then run:

```bash
make docker-run CONFIG=config_validator.yaml
```

Each task workspace receives a `validation_report.yaml` with per-check results,
and a `validation_summary.yaml` with aggregated statistics is written to the
workspace root. Tasks skipped by `platform_support.status: skip` or by a
non-matching `platform_support.required_arch` are filtered before workspace
creation, so they do not produce a validation report or appear in the summary
counts.

For large validation batches on a multi-GPU server, use the parallel Docker
runner. It starts one validator worker container per GPU and writes the same
reports:

```bash
make docker-parallel-run \
  CONFIG=config_validator.yaml \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix validator_parallel8"
```

Parallel resume skips only validator tasks with a framework-finalized schema-v3
report and matching completion digest. A partial, legacy, or manually copied
`validation_report.yaml` is rerun.

## Validator configuration

The validator's own backend and limits are set in
`agents/task_validator/agent_config.yaml`. This backend-neutral example leaves
the model unset so the selected CLI uses its default:

```yaml
backend: claude_code          # claude_code | codex
timeout_seconds: 1200         # minimum outer limit; auto-raised for command budgets (0 disables)
python_path: null             # null uses the framework/container Python

# Optional model settings for the active backend.
# claude_code: passed as `claude --model` and `claude --effort`
# codex: passed as `codex exec --model` and `model_reasoning_effort`
model: null                   # null uses the selected CLI's default
effort: max

compile_timeout: 600
correctness_timeout: 600
performance_timeout: 600
```

Per-task timeout values in `config.yaml` override these defaults. The outer
backend timeout is expanded to cover the three command budgets plus review time.

## `task_validator` checks

The `task_validator` runs the following checks in order.

| # | Check | What it verifies |
| --- | --- | --- |
| 1 | `config_schema` | All required fields exist with correct types |
| 2 | `source_files_exist` | Every file in `source_file_path` exists |
| 3 | `target_symbols_found` | Every `target_kernel_functions` symbol is defined in source |
| 4 | `compilation` | `compile_command` succeeds within `compile_timeout` |
| 5 | `correctness` | `correctness_command` succeeds within `correctness_timeout` |
| 6 | `performance` | `performance_command` succeeds within `performance_timeout`, if present |
| 7 | `correctness_implementation_review` | The correctness check is meaningful, not trivially passing |
| 8 | `self_contained` | No missing headers/imports; isolated tasks avoid undeclared external repos/paths, and repository tasks declare their upstream in `repo_url` |
| 9 | `gpu_hang_check` | No command hangs or times out |
| 10 | `result_template_compatibility` | Command and per-case output signals can be consumed by the centralized evaluator |
| 11 | `benchmark_integrity` | Every case has scoreable device timing/method metadata, stable identity, and fair state/allocation boundaries; missing exact replay validation is WARN |
| 12 | `harness_integrity` | Harness logic stays protected while co-located target and Triton-JIT implementation nodes remain editable |

## Overall status

- **PASS:** all applicable checks passed; a contract-approved `SKIP` does not
  prevent PASS.
- **WARN:** no failures, but at least one warning (for example, a questionable
  correctness implementation). Acceptable with justification.
- **FAIL:** a check failed/timed out, the backend failed, or the report contract is incomplete; the task must be fixed before merging.

The framework normalizes every report and recomputes `overall_status`; it does
not trust the agent's claimed aggregate. Compile/correctness/performance commands
must exit zero, and stale JSON/YAML output cannot override a failure. The final
CLI exits nonzero when any task validation fails. WARN is non-failing but requires
review.

For performance, `cuda_graph` and `cuda_event_fallback` are the only scoreable
methods. CPU/host timing, missing or mixed methods, candidate-triggered fallback,
invalid/partial cases, missing state restore, or demonstrably asymmetric timed work
fail `benchmark_integrity`. Missing exact output validation from the captured Graph is
WARN by itself; an observed incorrect/stale replay or a demonstrated unsafe state/reset
interaction remains FAIL. The 10-warmup/100-sample pattern is a recommended default
rather than a hard scoring rule.

The validator receives trusted framework facts for the protected harness boundary and
the pre/post scoring lifecycle. Baseline and candidate are separate invocations of the
same protected performance entrypoint, so a task runner does not need an in-process
reference timing path. Judgment-heavy WARN/FAIL results include source-line or runtime
case evidence; genuinely unavailable evidence is reported as WARN rather than inferred
as a failure.

## Result template

A validated task's **compile → correctness → performance** flow must produce results
that populate the standard template:

```yaml
task_name: "<full path relative to tasks/>"
pass_compilation: true/false
compilation_error_message: null
pass_correctness: true/false
correctness_error_message: null
base_execution_time: 0.0          # ms
best_optimized_execution_time: 0.0
speedup_ratio: 0.0
baseline_benchmark_methods: []
optimized_benchmark_methods: []
benchmark_method_consistent: true/false
valid_baseline_cases: 0
valid_optimized_cases: 0
speedup_calculation_error_message: null
optimization_summary: "Framework-generated evaluator summary"
score: 0.0
```

For the full author checklist and self-containedness rules, see
`agents/task_validator/README.md` in the repository and
[Add a task](add-task.md).
