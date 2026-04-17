# AITER Repository Tasks

This directory contains repository-scope tasks for Triton kernels in
[ROCm/aiter](https://github.com/ROCm/aiter).

These tasks intentionally follow the same Level-3 pattern as the rocPRIM
repository tasks:

- `config.yaml` declares the upstream repository plus the three evaluator entry
  points: `compile_command`, `correctness_command`, and `performance_command`.
- `scripts/task_runner.py` is the stable adapter between AgentKernelArena and
  the upstream project.

## Task Inventory

The current task set focuses on Triton kernels that sit on critical LLM paths:

- `mla_decode_rope`
  Kernel family: MLA decode with RoPE.
  Upstream sources:
  `aiter/ops/triton/attention/mla_decode_rope.py`,
  `aiter/ops/triton/_triton_kernels/attention/mla_decode_rope.py`
  Upstream tests:
  `op_tests/triton_tests/attention/test_mla_decode_rope.py`

- `pa_decode`
  Kernel family: paged-attention decode.
  Upstream sources:
  `aiter/ops/triton/attention/pa_decode.py`,
  `aiter/ops/triton/_triton_kernels/attention/pa_decode.py`
  Upstream tests:
  `op_tests/triton_tests/attention/test_pa_decode.py`

- `pa_prefill`
  Kernel family: paged-attention prefill / context attention.
  Upstream sources:
  `aiter/ops/triton/attention/pa_prefill.py`,
  `aiter/ops/triton/_triton_kernels/attention/pa_prefill.py`
  Upstream tests:
  `op_tests/triton_tests/attention/test_pa_prefill.py`

- `unified_attention`
  Kernel family: unified attention over heterogeneous request shapes.
  Upstream sources:
  `aiter/ops/triton/attention/unified_attention.py`,
  `aiter/ops/triton/_triton_kernels/attention/unified_attention.py`
  Upstream tests:
  `op_tests/triton_tests/attention/test_unified_attention.py`

- `moe_routing_sigmoid_top1_fused`
  Kernel family: MoE routing with fused sigmoid top-1 selection.
  Upstream sources:
  `aiter/ops/triton/moe/moe_routing_sigmoid_top1_fused.py`,
  `aiter/ops/triton/_triton_kernels/moe/moe_routing_sigmoid_top1_fused.py`
  Upstream tests:
  `op_tests/triton_tests/moe/test_moe_routing_sigmoid_top1_fused.py`

We intentionally did not use `moe_gemm_a8w8` here even though it is an
important LLM kernel. In fresh-clone validation on the current ROCm/Triton
stack, that path hit a gfx942 FP8 legalization issue in the upstream Triton
codegen, while the routing kernel above validated cleanly end-to-end.

## Config Field Deep Dive

The fields below are the ones we actively use for these tasks, along with where
the framework consumes them.

- `repo_url`
  Used by `src/preprocessing.py` in `_ensure_repo_cloned()` and
  `setup_workspace()` to clone the upstream repo into the task directory before
  copying the full task folder into a per-run workspace.

- `task_type: repository`
  Used by `src/prompt_builder.py` to select the repository-specific prompt path,
  and by `src/prompts/task_type.py` to emit the Level-3 "full repository
  workspace" task framing.

- `repository_language: triton`
  Used by `src/prompt_builder.py::_load_cheatsheet()` to select the Triton
  knowledge cheatsheet for repository tasks.

- `source_file_path`
  Optional but useful. When present, `src/prompt_builder.py` includes these
  files in the generated "Source Code" section of the agent prompt.

- `target_kernel_functions`
  Optional companion to `source_file_path`. Also consumed by
  `src/prompt_builder.py` to tell the agent which wrappers / kernels matter.

- `compile_command`
  Executed by `src/evaluator.py::evaluate_compilation()` inside the duplicated
  workspace. For these tasks the command is always
  `python3 scripts/task_runner.py compile`.

- `correctness_command`
  Executed by `src/evaluator.py::evaluate_correctness()` inside the duplicated
  workspace.

- `performance_command`
  Executed by `src/performance.py::measure_performance()` inside the duplicated
  workspace.

  The performance parser looks for JSON reports in common locations including
  `build/performance_report.json`, so every AITER task runner writes that file.
  Each report entry uses `execution_time_ms`, which is parsed by
  `src/testcases.py::_extract_time_from_dict()`.

- `prompt.instructions`
  Deliberately omitted in these tasks.

  In this codebase, setting `prompt.instructions` overrides the default
  instruction block entirely. We want the default command-aware instructions
  from `src/prompt_builder.py::instructions()` so the compile / correctness /
  performance commands remain visible to the optimizing agent.

## Runtime Design Notes

These tasks target pure-Triton AITER paths and intentionally run with
`ENABLE_CK=0` inside their task runners.

That design choice matters because:

- the selected kernels do not require CK-backed paths;
- it avoids recursive submodule requirements for task setup;
- fresh shallow clones become much cheaper and more robust;
- the runner can validate the real Triton kernels via smoke launches,
  correctness checks against upstream reference code, and curated benchmarks.

Each runner creates a local `--system-site-packages` virtual environment in the
workspace and installs only the lightweight Python packages needed to import the
upstream helpers cleanly.

## Validation Approach

Each task was validated against a fresh temporary workspace using the same
high-level flow as the framework:

1. copy the task folder to a temporary source location;
2. let `src/preprocessing.py::setup_workspace()` clone `https://github.com/ROCm/aiter.git`;
3. duplicate that fully populated task folder into a per-run workspace;
4. execute the exact `compile_command`, `correctness_command`, and
   `performance_command` from the task config;
5. confirm that `src/performance.py::measure_performance()` parses
   `build/performance_report.json` successfully.
