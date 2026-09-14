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
| `operator2flydsl` | Reimplement a production operator in FlyDSL and optimize it |
| `repository` | Repository-level task |

The repository ships task suites including `hip2hip` (gpumode and others),
`triton2triton` (vLLM and ROCmBench), `torch2hip`, `instruction2triton`,
`torch2flydsl`, `triton2flydsl`, and `flydsl2flydsl`, plus `operator2flydsl`
tasks under `tasks/SIKL-task/` and repository-level tasks under
`tasks/repository/`.

## `operator2flydsl` tasks

An `operator2flydsl` task points at an operator that already runs in production
and asks for a FlyDSL implementation of it, scored against that production
implementation. Unlike `triton2flydsl` or `torch2flydsl`, nothing in the name
constrains the source: it may be written in any language and may ship inside a
larger project rather than as a self-contained file.

The task adds exactly one field to the isolated-kernel schema:

```yaml
task_type: operator2flydsl

# The single editable file. The implementation lands here, and the harness
# scores whatever it finds.
source_file_path:
  - kernel.py
target_kernel_functions:
  - build_gemm_a16w16_nt_n6144_k6144_module

# The only field this task type adds: the production implementation to
# reimplement. Read-only reference material for the agent, and task-relative --
# an absolute path into the runtime image would escape the workspace.
rewrite_source_file: aiter_source/aiter/tuned_gemm.py

# When that source lives in the runtime image rather than in the task, declare
# it and Arena seeds it into the workspace before the agent starts. Same
# mechanism image_kernel tasks use.
image_repo_path: /sgl-workspace/aiter
repo_subdir: aiter_source
image_repo_exclude:
  - jit

kernel_identity:
  logical_operator: gemm_a16w16_nt_n6144_k6144
  source_owner: aiter
```

Everything else an agent needs is an existing field. The implementation lands in
`source_file_path[0]`, and `kernel_identity` carries the operator's identity and
its owner.

Pick a `repo_subdir` that cannot shadow the package being seeded. A directory
named `aiter` at the workspace root would sit on `sys.path` ahead of the real
package for every command the task runs.

Two things deliberately stay out of the task. How an agent searches for the
implementation -- attempt counts, intermediate filters, time budgets -- is agent
configuration, because a second agent implementing this task type may have no
such notion. And the source's host entry point is prose: name it in
`prompt.instructions` or in the driver's docstring rather than adding a field,
so nothing has to parse it back out.

## `kernel_identity`

`kernel_identity` is a shared contract, not a per-agent field. Both KernelForge
integrations read it through one resolver, and any agent that publishes to a
knowledge base should read it the same way.

| Key | Description |
| --- | --- |
| `logical_operator` | Stable name for the operator, independent of shape or file. Agents use it as the knowledge-base identity, and some derive the required factory symbol from it -- keep it consistent with whatever the harness looks up. |
| `source_owner` | The framework that owns the production implementation (`aiter`, `vllm`, `sglang`). |
| `kernel_kind` | Optional. The editable source language for tasks whose type does not encode it. |

`source_owner` is worth declaring even when an agent could guess it. Inference
generally reads the owner out of the source file's path, and an agent that
copies the source into a scratch workspace destroys exactly that evidence, so
the guess degrades to "unknown" and any recipe the run publishes is filed under
an owner nothing looks for.

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
#         flydsl2flydsl, operator2flydsl, repository
task_type: hip2hip
```

Repository-level tasks (`task_type: repository`) use a different shape because
they clone and optimize an upstream project rather than a small source bundle.
They require `repo_url`, `repository_language`, `compile_command`, and
`correctness_command`; `source_file_path` and `target_kernel_functions` are
optional hints when the target files and symbols are known.

Files shipped by the task outside its declared source/target files are treated
as immutable evaluation inputs during optimization, including JSON case tables
and reference modules. Declare additional editable implementation helpers in
`editable_sources` (a list of task-relative file paths). Keep generated reports
and build artifacts separate from these inputs. See the
[benchmark methodology](../reference/benchmark-methodology.md) for the shared
guard's scope and the function-level boundary in combined kernel/harness files.

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

# Optional platform gate. Omit this block for tasks that run everywhere.
platform_support:
  required_arch: gfx942   # compared with the detected GPU architecture
  status: active          # active | skip
  skip_reason: null       # recommended when status is skip
```

Some specialized launchers and task runners use additional fields such as
`harness_path` or `target_file_path`. Document those fields with the task or
agent that consumes them; they are not part of the common evaluator schema.

Tasks with `platform_support.status: skip`, or with a `required_arch` that does
not match the current run, are skipped before workspace creation. Historical
per-suite fields such as `runnable_on_gfx942` are documentation only.

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
runs 12 checks, including benchmark and harness integrity, and emits a
framework-finalized `validation_report.yaml`. See
[Validate tasks](task-validator.md) for the full check list and how to run it.
