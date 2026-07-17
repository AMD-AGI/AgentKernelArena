---
myst:
    html_meta:
        "description": "Learn how to create a new GPU kernel task for AgentKernelArena, including directory layout, config.yaml schema, supported task types, and authoring rules."
        "keywords": "AgentKernelArena, add task, GPU kernel, HIP, Triton, CUDA, config.yaml, task types, ROCm"
---

# Add a task in AgentKernelArena

A task is a single GPU kernel optimization problem. Each task lives below its
task-type directory and is described by a `config.yaml`. Optional suite and
difficulty directories can appear between the task type and task name, for
example `tasks/triton2triton/rocmbench/hard/gemm/`.

## Task types

The `task_type` field declares what kind of optimization the task represents.

| `task_type` | Meaning |
| --- | --- |
| `hip2hip` | Optimize an existing HIP kernel |
| `cuda2hip` | Port and optimize a CUDA kernel to HIP |
| `triton2triton` | Optimize an existing Triton kernel |
| `instruction2triton` | Write a Triton kernel from an instruction/spec |
| `torch2hip` | Replace a PyTorch reference with a HIP kernel |
| `torch2flydsl` | Replace a PyTorch reference with a FlyDSL kernel |
| `triton2flydsl` | Translate a Triton kernel to FlyDSL |
| `flydsl2flydsl` | Optimize a FlyDSL kernel (requires FlyDSL) |
| `repository` | Repository-level task |

The repository ships task suites including `hip2hip` (gpumode and others),
`triton2triton` (vLLM and ROCmBench), `torch2hip`, `instruction2triton`,
`torch2flydsl`, `triton2flydsl`, and `flydsl2flydsl`, plus repository-level
tasks under `tasks/repository/`.

## Directory layout

```text
tasks/<task_type>/[<suite>/...]/<task_name>/
├── config.yaml                  # Task configuration (required)
├── scripts/
│   └── task_runner.py           # Compile/correctness/performance runner (recommended)
└── source/                      # or src/
    └── <kernel files>           # .cu, .hip, .py, etc.
```

Makefile-based or test-file-based layouts are also acceptable, as long as every
path referenced in `config.yaml` resolves inside the task directory.

## Required `config.yaml` fields

Most tasks optimize files that are copied into the task workspace. For those
isolated-kernel tasks, all command fields are *lists*, even when there's a
single command.

```yaml
# Source files containing the kernel code (relative to the task root)
source_file_path:
  - source/my_kernel.hip

# Kernel function names that must be defined in the source files
target_kernel_functions:
  - my_kernel_function

# Command(s) to compile or build-check the task
compile_command:
  - python3 scripts/task_runner.py --mode compile

# Command(s) to run correctness validation
correctness_command:
  - python3 scripts/task_runner.py --mode correctness

# One of: hip2hip, cuda2hip, triton2triton, triton2flydsl,
#         instruction2triton, torch2hip, torch2flydsl,
#         flydsl2flydsl, repository
task_type: hip2hip
```

Repository-level tasks (`task_type: repository`) use a different shape because
they clone and optimize an upstream project rather than a small source bundle.
They require `repo_url`, `repository_language`, `compile_command`, and
`correctness_command`; `source_file_path` and `target_kernel_functions` are
optional hints when the target files and symbols are known.

```yaml
repo_url: https://github.com/ROCm/rocPRIM.git
# repo_subdir: rocPRIM        # optional; defaults from repo_url
task_type: repository
repository_language: hip

compile_command:
  - python3 scripts/task_runner.py compile

correctness_command:
  - python3 scripts/task_runner.py correctness
```

## Optional `config.yaml` fields

```yaml
# Command(s) to measure performance
performance_command:
  - python3 scripts/task_runner.py --mode performance

# Optional per-command limits in seconds (framework defaults are 3600).
compile_timeout: 3600
correctness_timeout: 3600
performance_timeout: 3600

# Legacy compatibility only; the centralized evaluator always writes the
# standard task_result.yaml schema.
task_result_template: null

# Prompt overrides for the optimization agent (null = auto-generated)
prompt:
  source_code: null      # override the default source-code section
  instructions: null     # custom instructions
  cheatsheet: null        # reference/cheatsheet content
```

Some specialized launchers and task runners use additional fields such as
`harness_path` or `target_file_path`. Document those fields with the task or
agent that consumes them; they are not part of the common evaluator schema.

## Authoring rules

To produce trustworthy, comparable scores, every task must have a reproducible
setup and must validate correctness meaningfully.

- **Reproducible setup**: Isolated-kernel tasks must not reference external
  repositories, absolute paths, or undeclared downloads. Generate test inputs
  inline or bundle small files in the task directory. Repository-level tasks
  should declare their upstream source in `repo_url` and keep setup commands
  explicit in `config.yaml`.
- **Real correctness check**: Compare against a CPU/NumPy reference, known-good
  output, or a PyTorch eager baseline; use sensible tolerances; test 2–3 shapes;
  and exit non-zero on failure.
- **Real compilation check**: Actually compile or syntax-check the source, not a
  text-pattern search; exit code `0` means success.
- **Performance methodology**: A recommended pattern is 10 warmup iterations plus
  100 measured iterations, reporting the average runtime.

## Performance helper stubs

The shared performance timing helpers are generated from `src/tools/perf/` into each
run workspace. In committed task sources:

- `tasks/*/rocmbench/**/performance_utils_pytest.py` is intentionally a stub.
- The `AKA-GENERATED` region in `triton2triton/vllm/*/scripts/task_runner.py` is
  intentionally a stub block.

Do not hand-edit those stubs. If a task needs shared timing behavior, add the
stub/marker and run `make sync-perf-helpers`. If you need to change timing logic,
edit the canonical file in `src/tools/perf/` and run `make check-perf-helpers`
before pushing. To inspect a task with the real helpers injected, run
`make materialize-perf-task TASK=tasks/...`.

## Validate before merging

Every new task must pass the `task_validator` agent before it's merged. It
runs 10 automated checks and emits a `validation_report.yaml`. See
[Validate tasks](task-validator.md) for the full check list and how to run it.
