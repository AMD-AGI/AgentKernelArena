# Task Validator Agent

## What This Agent Does

The **task_validator** agent validates that tasks in AgentKernelArena are correctly configured, self-contained, and functional. It does **not** optimize kernels. Instead, it runs 10 automated checks on each task and produces a structured `validation_report.yaml`.

Use it to:
- Audit existing tasks before controlled comparisons or RL data collection.
- Validate new tasks before merging them into the task suite.
- Identify broken tasks (missing files, external dependencies, trivially-passing correctness checks, GPU hangs).

## How to Use

### 1. Create a run configuration

Save the following as `config_task_validator.yaml`:

```yaml
agent:
  template: task_validator
tasks:
  - hip2hip/gpumode/GELU
  - triton2triton/vllm/triton_rms_norm
  - repository/rocprim/device_merge_sort
  # - all                     # validate every task
target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

### 2. Run

```bash
make docker-run CONFIG=config_task_validator.yaml
```

### 3. Read Results

Each task workspace will contain a `validation_report.yaml` with per-check results. A `validation_summary.yaml` is written to the workspace root with aggregated statistics.

### Agent Configuration

Edit `agents/task_validator/agent_config.yaml`. This portable example leaves the
model unset so the selected CLI uses its default:

```yaml
backend: claude_code          # claude_code | codex
timeout_seconds: 1200         # max time per task validation (set 0 to disable timeout)
python_path: null             # null -> auto-use framework-detected interpreter (recommended)

# Optional model settings for the active backend.
# claude_code: passed as `claude --model` and `claude --effort`
# codex: passed as `codex exec --model` and `model_reasoning_effort`
model: null                   # null uses the selected CLI's default
effort: max

compile_timeout: 600
correctness_timeout: 600
performance_timeout: 600
```

## Validation Checks

| # | Check | What It Verifies |
|---|-------|-----------------|
| 1 | **config_schema** | All required fields exist in `config.yaml` with correct types |
| 2 | **source_files_exist** | Every file in `source_file_path` exists in the workspace |
| 3 | **target_symbols_found** | Every function in `target_kernel_functions` is defined in source files |
| 4 | **compilation** | `compile_command` succeeds within the configured `compile_timeout` |
| 5 | **correctness** | `correctness_command` succeeds within the configured `correctness_timeout` |
| 6 | **performance** | `performance_command` succeeds within the configured `performance_timeout`, if present |
| 7 | **correctness_implementation_review** | The correctness check is meaningful (not trivially passing) |
| 8 | **self_contained** | No missing headers/imports; isolated tasks avoid undeclared external paths, while repository tasks declare upstream dependencies |
| 9 | **gpu_hang_check** | No command hangs or times out |
| 10 | **result_template_compatibility** | Task output maps to the standard `task_result_template.yaml` schema |

### Overall Status

- **PASS** — all applicable checks passed; a contract-approved `SKIP` does not prevent PASS
- **WARN** — no failures, but at least one warning (e.g., questionable correctness implementation)
- **FAIL** — at least one check failed

---

## New Task Requirements

Every new task added to `tasks/` must satisfy the following requirements to pass validation.

### Required Directory Structure

```
tasks/<task_type>/[<suite>/...]/<task_name>/
├── config.yaml                  # Task configuration (required)
├── scripts/
│   └── task_runner.py           # Validation runner (recommended pattern)
└── source/
    └── <kernel files>           # .cu, .hip, .py, etc.
```

Alternative structures (Makefile-based, test-file-based) are acceptable as long as all config references resolve.

### Required `config.yaml` Fields

```yaml
# List of source files containing kernel code (relative to task root)
source_file_path:
  - source/my_kernel.cu

# List of kernel function names that must be found in source files
target_kernel_functions:
  - my_kernel_function

# Command(s) to compile or build-check the task
compile_command:
  - python3 scripts/task_runner.py --mode compile

# Command(s) to run correctness validation
correctness_command:
  - python3 scripts/task_runner.py --mode correctness

# Task type: one of hip2hip, cuda2hip, triton2triton, triton2flydsl,
# instruction2triton, torch2hip, torch2flydsl, flydsl2flydsl, repository
task_type: cuda2hip
```

### Optional `config.yaml` Fields

```yaml
# Command(s) to run performance benchmarking
performance_command:
  - python3 scripts/task_runner.py --mode performance

# Legacy compatibility only; the centralized evaluator writes the standard schema.
task_result_template: null

# Prompt overrides for the optimization agent (null = auto-generated)
prompt:
  source_code: null
  instructions: null
  cheatsheet: null
```

### Self-Containedness Rules

A normal isolated-kernel task **must** be fully self-contained. A
`task_type: repository` task can declare an upstream repository with `repo_url`;
its adapter scripts and dependency/setup contract must still be self-contained.
For isolated tasks:

1. **No external repo dependencies.** Do not reference paths like `../../vllm/`, `/opt/external/`, or assume a cloned repo exists in the workspace. All source code the task needs must be inside the task directory.

2. **No missing headers.** Every `#include "foo.h"` in `.cu`/`.hip` files must resolve to a header that ships with the task (or is part of system/ROCm/CUDA includes).

3. **No missing Python imports.** Every `import` or `from X import Y` must resolve to either:
   - Python standard library
   - Packages available in the Docker container environment (torch, numpy, triton, etc.)
   - Local files within the task directory

4. **No external data downloads.** Test inputs must be generated inline (random tensors, synthetic data) or bundled as small files in the task directory.

### Correctness Check Rules

The correctness check **must** be a real validation, not a trivial pass:

1. **Compare against a reference.** Use a CPU/NumPy reference implementation, known-good output tensors, or a PyTorch eager-mode baseline.

2. **Use reasonable tolerances.** For FP32: `atol=1e-3, rtol=1e-3` typical. For FP16/BF16: `atol=1e-2, rtol=1e-2` typical. For FP8/INT8: `atol=1e-1` or custom per-task.

3. **Test multiple shapes.** Don't validate with a single input shape. Use at least 2-3 representative shapes covering small, medium, and large inputs.

4. **Return non-zero exit code on failure.** The correctness command must `sys.exit(1)` or raise an exception if validation fails.

### Compilation Check Rules

1. The `compile_command` must actually compile or syntax-check the source code (not just search for text patterns).
2. Exit code 0 means success, non-zero means failure.
3. A `build/compile_report.json` with `{"status": "ok"}` or `{"status": "fail", "error": "..."}` is recommended.

### Performance Check Rules (if applicable)

1. The `performance_command` should measure kernel execution time and report it in a parseable format.
2. It only needs to report the runtime for the implementation currently in the workspace. The framework runs the same command before and after agent execution and computes speedup.
3. A `build/performance_report.json` with timing data is recommended.
4. Recommended methodology: `10` warmup iterations + `100` measured iterations, and report the average measured runtime (speedup should be derived from averaged runtimes). The validator may mark performance as `WARN` if a task is functional but does not follow or clearly document this methodology.

### Result Template Compatibility

The task's output flow (compile → correctness → performance) must produce results that can populate the standard `task_result_template.yaml`:

```yaml
task_name: "<full path relative to tasks/>"
pass_compilation: true/false
compilation_error_message: null
pass_correctness: true/false
correctness_error_message: null
base_execution_time: 0.0          # in ms
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

### Checklist for New Task Authors

Before submitting a new task, verify:

- [ ] `config.yaml` has all required fields with correct types
- [ ] All `source_file_path` entries exist
- [ ] All `target_kernel_functions` are defined in the source files
- [ ] `compile_command` succeeds with exit code 0
- [ ] `correctness_command` succeeds with exit code 0
- [ ] Correctness check compares against a real reference (not trivially passing)
- [ ] Isolated tasks have no undeclared external paths; repository tasks declare `repo_url` and setup requirements
- [ ] Commands complete within reasonable time (no GPU hangs)
- [ ] Output is compatible with `task_result_template.yaml`

Run the task_validator agent on your task to automatically verify all of the above.
