---
myst:
    html_meta:
        "description": "Use the AgentKernelArena task_validator agent to run 10 automated quality checks before using GPU kernel tasks in shared experiments."
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
workspace root.

For large validation batches on a multi-GPU server, use the parallel Docker
runner. It starts one validator worker container per GPU and writes the same
reports:

```bash
make docker-parallel-run \
  CONFIG=config_validator.yaml \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix validator_parallel8"
```

Parallel resume skips validator tasks whose workspace already contains
`validation_report.yaml`.

## Validator configuration

The validator's own backend and limits are set in
`agents/task_validator/agent_config.yaml`. This backend-neutral example leaves
the model unset so the selected CLI uses its default:

```yaml
backend: claude_code          # claude_code | codex
timeout_seconds: 1200         # max time per task validation (0 disables the timeout)
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
| 10 | `result_template_compatibility` | Output maps to the standard `task_result_template.yaml` |

## Overall status

- **PASS** — all applicable checks passed; a contract-approved `SKIP` does not
  prevent PASS.
- **WARN** — no failures, but at least one warning (for example, a questionable
  correctness implementation). Acceptable with justification.
- **FAIL** — at least one check failed; the task must be fixed before merging.

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
