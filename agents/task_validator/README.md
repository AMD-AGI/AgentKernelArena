# Task Validator Agent

## What This Agent Does

The **task_validator** agent validates that tasks in AgentKernelArena are correctly configured, self-contained, and functional. It does **not** optimize kernels. Instead, it runs 10 automated checks on each task and produces a structured `validation_report.yaml`.

Use it to:
- Audit existing tasks for quality and compliance before publishing to leaderboards.
- Validate new tasks before merging them into the task suite.
- Identify broken tasks (missing files, external dependencies, trivially-passing correctness checks, GPU hangs).

## How to Use

### 1. Set `config.yaml`

```yaml
agent:
  template: task_validator
tasks:
  - hip2hip/rmsnorm
  - cuda2hip/awq_gemm
  - triton2triton/triton_fused_moe
  # - all                     # validate every task
target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

### 2. Run

```bash
python3 main.py
```

### 3. Read Results

Each task workspace will contain a `validation_report.yaml` with per-check results. A `validation_summary.yaml` is written to the workspace root with aggregated statistics.

### Agent Configuration

Edit `agents/task_validator/agent_config.yaml`:

```yaml
backend: claude_code          # claude_code | codex | cursor (codex/cursor not yet implemented)
timeout_seconds: 600          # max time per task validation
python_path: /root/AgentKernelArena/.venv/bin/python3
```

## Validation Checks

| # | Check | What It Verifies |
|---|-------|-----------------|
| 1 | **config_schema** | All required fields exist in `config.yaml` with correct types |
| 2 | **source_files_exist** | Every file in `source_file_path` exists in the workspace |
| 3 | **target_symbols_found** | Every function in `target_kernel_functions` is defined in source files |
| 4 | **compilation** | `compile_command` runs successfully (exit code 0, within 120s timeout) |
| 5 | **correctness** | `correctness_command` runs successfully (exit code 0, within 180s timeout) |
| 6 | **performance** | `performance_command` runs successfully (if present, within 180s timeout) |
| 7 | **correctness_implementation_review** | The correctness check is meaningful (not trivially passing) |
| 8 | **self_contained** | No missing headers, imports, or references to external repos/paths |
| 9 | **gpu_hang_check** | No command hangs or times out |
| 10 | **result_template_compatibility** | Task output maps to the standard `task_result_template.yaml` schema |

### Overall Status

- **PASS** — all checks passed
- **WARN** — no failures, but at least one warning (e.g., questionable correctness implementation)
- **FAIL** — at least one check failed

---

## New Task Requirements

Every new task added to `tasks/` must satisfy the following requirements to pass validation.

### Required Directory Structure

```
tasks/<task_type>/<task_name>/
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

# Task type: one of hip2hip, cuda2hip, triton2triton, torch2hip, instruction2triton, rocprim
task_type: cuda2hip
```

### Optional `config.yaml` Fields

```yaml
# Command(s) to run performance benchmarking
performance_command:
  - python3 scripts/task_runner.py --mode performance

# Override which result template to use (null = default)
task_result_template: null

# Prompt overrides for the optimization agent (null = auto-generated)
prompt:
  source_code: null
  instructions: null
  cheatsheet: null
```

### Self-Containedness Rules

A task **must** be fully self-contained. This means:

1. **No external repo dependencies.** Do not reference paths like `../../vllm/`, `/opt/external/`, or assume a cloned repo exists in the workspace. All source code the task needs must be inside the task directory.

2. **No missing headers.** Every `#include "foo.h"` in `.cu`/`.hip` files must resolve to a header that ships with the task (or is part of system/ROCm/CUDA includes).

3. **No missing Python imports.** Every `import` or `from X import Y` must resolve to either:
   - Python standard library
   - Packages available in the `.venv` environment (torch, numpy, triton, etc.)
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
2. Output should include baseline time and optimized time for speedup calculation.
3. A `build/performance_report.json` with timing data is recommended.

### Result Template Compatibility

The task's output flow (compile → correctness → performance) must produce results that can populate the standard `task_result_template.yaml`:

```yaml
task_name: "<task_type>/<task_name>"
best_optimized_source_file_path:
  - <source files>
best_optimized_kernel_functions:
  - <kernel functions>
pass_compilation: true/false
compilation_error_message: null
pass_correctness: true/false
correctness_error_message: null
base_execution_time: 0.0          # in ms
best_optimized_execution_time: 0.0
speedup_ratio: 0.0
optimization_summary: ""
```

### Checklist for New Task Authors

Before submitting a new task, verify:

- [ ] `config.yaml` has all required fields with correct types
- [ ] All `source_file_path` entries exist
- [ ] All `target_kernel_functions` are defined in the source files
- [ ] `compile_command` succeeds with exit code 0
- [ ] `correctness_command` succeeds with exit code 0
- [ ] Correctness check compares against a real reference (not trivially passing)
- [ ] No `#include` / `import` references to files outside the task directory
- [ ] No hardcoded paths to external repos or data
- [ ] Commands complete within reasonable time (no GPU hangs)
- [ ] Output is compatible with `task_result_template.yaml`

Run the task_validator agent on your task to automatically verify all of the above.
