---
myst:
    html_meta:
        "description": "Common AgentKernelArena configuration reference: run config.yaml schema, task config fields, CLI flags, scoring formula, and the agent registry."
        "keywords": "AgentKernelArena, API reference, config.yaml, CLI flags, scoring, agent registry, ROCm, GPU kernel"
---

# AgentKernelArena configuration and API reference

This topic documents run configuration files, per-task
configuration, command-line flags, scoring formula, and agent registry.

## Run configuration

A run configuration defines a single experiment. Start from a file under
`example_configs/` and copy it when creating a new experiment.

| Field | Type | Description |
| --- | --- | --- |
| `agent.template` | string | Agent to run. One of the [supported agents](../how-to/agents.md#supported-agents). |
| `tasks` | list of strings | Task selectors relative to `tasks/`. Use `all` for every task, a category prefix for a group, or a full path for a single task. |
| `target_gpu_model` | string | Target GPU model, for example `MI300` or `MI355X`. Used to select the Docker image architecture, set `PYTORCH_ROCM_ARCH`, and name the workspace. |
| `log_directory` | string | Directory for run logs. |
| `workspace_directory_prefix` | string | Prefix for the workspace directory. Runs are stored under the ignored path `experiments/<prefix>_<gpu>_<agent>`. |

Specialized GEAK and mini-swe integrations also accept some optional top-level
fields:

| Field | Type | Description |
| --- | --- | --- |
| `gpu_ids` | string | Comma-separated GPU IDs exposed to specialized internal workers. This is separate from the host runner's `GPU_IDS` variable. |
| `num_parallel` | integer | Number of GEAK sub-agents/worktrees to run in parallel. mini-swe configures this under its agent config instead. |
| `run_mode` | string | `geak_v3_triton` mode override, such as `quick` or `full`. |

Agent-specific settings remain in `agents/<agent_name>/agent_config.yaml`; see
the selected integration's directory for precedence rules and additional fields.

Example:

```yaml
agent:
  template: cursor

tasks:
  - hip2hip/gpumode/GELU
  - triton2triton/vllm/triton_rms_norm

target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

## Evaluation tools

`evaluation_tools` configures optional, isolated kernel-analysis sidecars. The
section is disabled when it is absent, `null`, or `false`. A configured mapping
with no enabled tools is also disabled unless the host runner supplies
`AKA_EVAL_TOOLS`; for a mapping, that host subset replaces its `enabled` value.
The built-in IDs are `triton_fpsan`, `gpu_asan`, `rocjitsu`,
`rocjitsu_waitcheck`, `rocjitsu_consan`, and `hip_fpsan`.

Sidecar build locks, integrated positive controls, and end-to-end fixtures
currently exist only for `gfx950`; all six startup controls pass in the current
MI355X qualification. Each applicable candidate still needs a task-specific
adapter and attestation, and enabling an image alone does not imply that a
kernel was analyzed. See [Check kernels with evaluation
tools](../how-to/use-evaluation-tools.md) for the support matrix and operational
requirements.

When tools are enabled, the selected scoring image must resolve to the same
immutable local Docker image ID as the pinned
`lmsysorg/sglang-rocm@sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78`
manifest. The runner rejects a different build, launches by the verified ID,
and records both the selected reference and verified ID in plan source evidence.

Worker reports live at repository-root
`.eval-tool-artifacts/<worker-label>`. The runner mounts that specific host
directory read/write into both sidecars and the scoring container; the latter
submount remains writable when the quality-loop repository root is read-only.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `evaluation_tools.enabled` | boolean, string, or list of strings | empty | Tool IDs to plan. `true` expands to all six built-ins; `false` disables the feature. A single string is accepted. Hyphens are normalized to underscores. When the host sets `AKA_EVAL_TOOLS`, its normalized subset is authoritative for both started sidecars and the in-container plan. |
| `evaluation_tools.policy` | `advisory` or `required` | `advisory` | `advisory` always permits performance but records an unsatisfied policy. `required` permits performance only when every applicable selected tool is ready, completes, and reports `clean`. |
| `evaluation_tools.positive_control` | `required`, `optional`, `disabled`, or boolean | `required` | Requires the applicable synthetic known-bug startup control to pass before runtime capability is ready. `optional`, `disabled`, and `false` normalize to not required; the worker still runs and reports its control. |
| `evaluation_tools.timeout_s` | exact integer from 1 through 3600 | `3600` | Default maximum execution time for each selected tool. Booleans, floats, and numeric strings are rejected. |
| `evaluation_tools.runtime_profile` | string or `null` | `null` | Fallback runtime-identity assertion for plans whose tool has no `runtime_ref`. It does not select an image and must exactly match worker health when set. |
| `evaluation_tools.tools` | mapping | `{}` | Per-tool configuration keyed by normalized tool ID. Entries do not enable tools. |
| `evaluation_tools.tools.<id>.runtime_ref` | string or `null` | automatic host image ID | Exact bare local Docker image ID (`sha256:...`) asserted by the plan and compared with worker health. The `image_digest` key is an alias. This field does not select an image; omit it when using automatic host injection. |
| `evaluation_tools.tools.<id>.timeout_s` | exact integer from 1 through 3600 | top-level timeout | Per-tool timeout; it cannot exceed the top-level timeout. |
| `evaluation_tools.tools.<id>.options` | mapping | `{}` | Adapter options, including argv lists and candidate-evidence paths. Reserved framework keys are rejected at run and task level: `positive_control_required`; GPU ASan runtime/preload/library keys; rocJITsu binary/config keys; Waitcheck CLI/C API keys; ConSan hook keys; and HIP-FpSan include/header keys. |

The exact reserved option keys are `positive_control_required` for every tool;
`asan_runtime_dir`, `hip_asan_runtime`, `host_asan_preload`,
`host_asan_lib_dir`, and `normal_rocm_lib_dir` for `gpu_asan`;
`rocjitsu_binary` and `config_path` for `rocjitsu`; `waitcheck_binary` and
`waitcheck_capi_wrapper` for `rocjitsu_waitcheck`; `consan_hook` for
`rocjitsu_consan`; and `include_dir` and `public_header` for `hip_fpsan`. They
are selected and attested by worker health, not YAML.

An explicit `evaluation_tools` mapping and each per-tool configuration reject
unknown fields. Unknown enabled tool IDs are also rejected. If both
`runtime_ref` and `image_digest` are supplied, they must be identical.

Example run-level section:

```yaml
evaluation_tools:
  enabled:
    - gpu_asan
  policy: advisory
  positive_control: required
  timeout_s: 600
  tools:
    gpu_asan:
      options: {}
```

The image is selected separately with a host override. The runner resolves and
attests its immutable local image ID automatically:

```bash
export AKA_EVAL_TOOL_IMAGE_GPU_ASAN='registry.example/eval-tool-gpu-asan@sha256:<digest>'
```

## Command-line flags

The in-container `main.py` entrypoint accepts these flags:

| Flag | Description |
| --- | --- |
| `--config_name <file>` | Config file to load (default `example_configs/quickstart_claude_mi300.yaml` for MI300/MI300X). Pass a matching config explicitly on another GPU |
| `--run-suffix <suffix>` | Suffix appended to the run directory name (letters, numbers, `.`, `_`, `-` only). Useful for labeling A/B runs |
| `--resume-run <run_dir>` | Resume a specific run directory, skipping completed tasks |
| `--resume-latest` | Resume the most recent run in the workspace |

These flags are passed to the in-container entrypoint through `make docker-run`
or `make docker-parallel-run` (`CONFIG=` sets `--config_name`; `RUN_ARGS=`
forwards the rest):

```bash
make docker-run CONFIG=config_triton.yaml RUN_ARGS="--run-suffix with_mcp"
make docker-parallel-run CONFIG=config_triton.yaml GPU_IDS=0,1 RUN_ARGS="--run-suffix with_mcp_parallel"
```

The following flags are internal implementation details used by
`docker-parallel-run` and should not be passed manually in normal use:

| Flag | Description |
| --- | --- |
| `--run-name <run_dir>` | Explicit run directory shared by parallel init, workers, and post-processing |
| `--parallel-init` | Initialize the shared `.parallel/` queue |
| `--parallel-worker` | Claim and execute tasks from the shared queue |
| `--worker-id <id>` | Worker identifier used in queue descriptors and logs |
| `--postprocess-only` | Aggregate results once after all workers finish |

## Docker runner Make targets

The following Make targets are available for running experiments.

| Target | Description |
| --- | --- |
| `make docker-run CONFIG=example_configs/quickstart_claude_mi300.yaml` | Run tasks serially in one Docker container |
| `make docker-parallel-run CONFIG=example_configs/benchmark_cursor_mi355x.yaml GPU_IDS=0,1` | Run one Docker worker per listed GPU, using a shared dynamic task queue |
| `make docker-smoke` | Verify Docker, ROCm runtime visibility, Python imports, and GPU access |
| `make docker-check-agents CONFIG=example_configs/quickstart_claude_mi300.yaml` | Verify the first-class host CLI selected by the config inside Docker (`task_validator` resolves to its backend). Override with `AGENTS=claude_code,codex`; use `AGENTS=all` for all three. Specialized integrations use their own checks |
| `make docker-shell` | Open an interactive shell in the experiment runtime |

`docker-parallel-run` accepts these environment variables:

| Variable | Description |
| --- | --- |
| `GPU_IDS` | Comma- or space-separated host GPU IDs. If omitted, the runner uses `rocm-smi --showid` |
| `RUN_ARGS` | Additional `main.py` flags, such as `--run-suffix`, `--resume-run`, or `--resume-latest` |
| `AKA_LOGICAL_GPU` | Logical GPU index inside a masked worker container. Defaults to `0` and normally should not be changed |

Each parallel worker sets `ROCR_VISIBLE_DEVICES` to the host GPU ID and sets
`HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL` to the
logical GPU index inside the masked container. See
[Run tasks in parallel across multiple GPUs](../how-to/parallel-run.md) for the
full scheduling model.

## Task configuration

Each task is defined by a `config.yaml` in its directory. Command fields are
*lists*.

For isolated-kernel tasks (`hip2hip`, `cuda2hip`, `triton2triton`,
`triton2flydsl`, `instruction2triton`, `torch2hip`, `torch2flydsl`, and
`flydsl2flydsl`):

| Field | Required | Description |
| --- | --- | --- |
| `source_file_path` | Yes | Source files containing the kernel, relative to the task root |
| `target_kernel_functions` | Yes | Kernel function names that must be defined in the source |
| `compile_command` | Yes | Command(s) to compile or build-check |
| `correctness_command` | Yes | Command(s) to validate correctness |
| `task_type` | Yes | One of `hip2hip`, `cuda2hip`, `triton2triton`, `triton2flydsl`, `instruction2triton`, `torch2hip`, `torch2flydsl`, or `flydsl2flydsl` |
| `performance_command` | No | Command(s) to measure performance |
| `compile_timeout` | No | Per-command compilation timeout in seconds (default `3600`) |
| `correctness_timeout` | No | Per-command correctness timeout in seconds (default `3600`) |
| `performance_timeout` | No | Per-command performance timeout in seconds (default `3600`) |
| `task_result_template` | No | Legacy compatibility field. The centralized evaluator writes the standard result schema regardless of this value |
| `platform_support` | No | Optional run-gating metadata; see below |
| `prompt.source_code` | No | Override the prompt's source-code section |
| `prompt.instructions` | No | Custom prompt instructions |
| `prompt.cheatsheet` | No | Reference/cheatsheet content for the prompt |

For repository-level tasks (`task_type: repository`):

| Field | Required | Description |
| --- | --- | --- |
| `repo_url` | Yes | Upstream repository to clone for the task |
| `task_type` | Yes | Must be `repository` |
| `repository_language` | Yes | Primary optimization stack, for example `hip` or `triton` |
| `compile_command` | Yes | Command(s) to compile or build-check |
| `correctness_command` | Yes | Command(s) to validate correctness |
| `performance_command` | No | Command(s) to measure performance |
| `compile_timeout` | No | Per-command compilation timeout in seconds (default `3600`) |
| `correctness_timeout` | No | Per-command correctness timeout in seconds (default `3600`) |
| `performance_timeout` | No | Per-command performance timeout in seconds (default `3600`) |
| `post_clone_install` | No | Setup command(s) to run after cloning the upstream repository |
| `post_clone_install_mode` | No | Controls when `post_clone_install` runs, for example `every_setup` |
| `repo_subdir` | No | Workspace subdirectory for the clone; defaults to the repository name derived from `repo_url` |
| `source_file_path` | No | Optional target source-file hints, relative to the cloned repository root |
| `target_kernel_functions` | No | Optional target function or kernel-symbol hints |
| `platform_support` | No | Optional run-gating metadata; see below |
| `prompt.instructions` | No | Custom prompt instructions |
| `prompt.cheatsheet` | No | Reference/cheatsheet content for the prompt |

See [Add a task](../how-to/add-task.md) for layout and authoring rules.

### Evaluation profile

The evaluator infers a profile from `task_type`, `repository_language`, source
suffixes, and repository paths. Add `evaluation_profile` only when that
inference is insufficient. Recognized profile overrides, including
`submission_paths`, are recorded in
`resolved_task_profile.explicit_overrides`. Unknown profile fields are currently
ignored, so use only the documented keys.

| Field | Type | Description |
| --- | --- | --- |
| `evaluation_profile.language` | string | Canonical values are `triton`, `hip`, `flydsl`, and `unknown`. |
| `evaluation_profile.artifact_kind` | string | `source_aot`, `python_jit`, `hsaco_precompiled`, or `unknown`. |
| `evaluation_profile.framework` | string | Framework identity such as `standalone`, `aiter`, `rocblas`, or `rccl`. |
| `evaluation_profile.instrumentation_control` | string | `compiler_controlled`, `recompile`, `none`, or `unknown`. This describes whether the selected candidate can be rebuilt/instrumented. |
| `evaluation_profile.adapter` | string or `null` | Explicit adapter identity, for example `triton_aot`, `flydsl_aot`, or `hip_fpsan_manual`. It is a claim that must still be supported by adapter options/evidence. |
| `evaluation_profile.source_available` | boolean | Whether source for the selected candidate is available to the evaluator. |
| `evaluation_profile.submission_paths` | string or list of strings | Workspace-relative candidate files captured before agent edits and fingerprinted after optimization. Required when repository/image tasks change files beyond the normal source fields. Absolute paths and `..` are rejected. |
| `evaluation_profile.fpsan_ported` | boolean | Explicit evidence that the HIP reference and candidate were manually ported to HIP-FpSan value semantics. |
| `evaluation_profile.rebuilt_from_source` | boolean | Explicit evidence used when a framework/library path is rebuilt from controlled source. It does not replace artifact attestation. |

### Task-level tool adapters

A task can add adapter options only for tools enabled by the run. The only
allowed task-level structure is:

```yaml
evaluation_tools:
  tools:
    gpu_asan:
      timeout_s: 300
      options:
        command: [python3, scripts/eval_tools/run_gpu_asan.py]
```

`timeout_s` must be between 1 and the run-level value. Task configuration cannot
enable another tool, change the top-level `policy` or `positive_control`, select
another runtime image, increase a timeout, or set any reserved framework option
listed above. Other options are merged over the run-level options.

Commands are argv lists, not shell strings. The built-in adapter keys are:

| Tool | Adapter keys |
| --- | --- |
| `triton_fpsan` | `comparison_command` or `command`; optional invocation-artifact-contained `attestation_path`. |
| `gpu_asan` | `command`; optional invocation-artifact-contained candidate `attestation_path`. ASan runtime/preload/library paths come only from verified sidecar health. |
| `rocjitsu` | HIP uses `launcher` or `command`, with optional `expected_kernel` and invocation-artifact-contained `race_report` whose filename must be `race.log`. Triton/FlyDSL requires `capsule` and the exact `triton_aot`/`flydsl_aot` profile adapter; arbitrary launchers and task-configured race-report paths are rejected. The capsule must be workspace-contained, single-dispatch, contain a golden expected output, be manifest-valid, and target `gfx950`. Binary/config and the trusted replay helper come only from the image/health. Automatic trusted capsule capture from correctness is not implemented. |
| `rocjitsu_waitcheck` | `code_object`, `expected_kernel`, and non-negative integer `kernel_entry`. The workspace-contained unbundled final ELF is SHA-256-bound to the plan; an image-owned inventory must match its exact `gfx950` descriptor before structured C API analysis. |
| `rocjitsu_consan` | `code_object`, focused native `command`, and independent `oracle_command`. The instrumented command must explicitly name and load the code object. Strict record/replay, SHA-256/FNV identity, complete accounting, and an oracle pass are required. Broad AITER/rocBLAS/RCCL runtimes are rejected. |
| `hip_fpsan` | `comparison_command` or `command`; optional invocation-artifact-contained candidate `attestation_path`; requires `evaluation_profile.fpsan_ported: true`. The include path comes only from verified sidecar health. |

Sidecar health attests and injects runtime-internal assets. Candidate/task
configuration cannot override or supply ASan preload/library paths, the
rocJITsu binary/config path, the Waitcheck CLI/C API wrapper, the ConSan hook,
or the HIP-FpSan include path.

Build-attestation JSON must store `artifact_path` relative to the directory
containing that JSON. The artifact must be beside or below the attestation;
absolute paths, `..`, and symlink resolutions that escape the directory are
rejected. The parser resolves the relative path in the scoring namespace and
verifies the declared SHA-256.

Each execution uses a fresh
`<tool>/<plan-fingerprint>/<invocation-id>` artifact directory. Custom
attestation and race-report paths are resolved against that directory; an
absolute path is accepted only when its resolution remains inside it.

### Platform support

`platform_support.status: skip` excludes a task unconditionally. An active task
with `platform_support.required_arch` is run only when that value matches the
detected GPU architecture. If `platform_support` is omitted, the task remains
runnable on every architecture.

## Result schema (`task_result.yaml`)

Each task produces a `task_result.yaml` in its workspace:

| Field | Description |
| --- | --- |
| `task_name` | Full task-directory path relative to `tasks/`, including any suite/difficulty levels |
| `pass_compilation` | Whether the optimized kernel compiled |
| `compilation_error_message` | Error text if compilation failed, else `null` |
| `pass_correctness` | Whether correctness passed |
| `correctness_error_message` | Error text if correctness failed, else `null` |
| `pass_tool_gate` | Whether the selected evaluation-tool policy allows performance to proceed. Defaults to `true` when tools are disabled. This is independent of `pass_correctness`. |
| `tool_policy_satisfied` | Whether every applicable selected tool was ready and completed with a `clean` finding status. Under `advisory`, this can be `false` while `pass_tool_gate` remains `true`. |
| `tool_evaluation` | Versioned complete plan, plan fingerprint, profile, capability, execution, finding, evidence, and decision data. Omitted when no tool is enabled. |
| `base_execution_time` | Baseline runtime in ms |
| `best_optimized_execution_time` | Best optimized runtime in ms |
| `speedup_ratio` | Speedup over baseline |
| `baseline_benchmark_methods` | Timing methods observed while measuring the baseline |
| `optimized_benchmark_methods` | Timing methods observed while measuring the optimized kernel |
| `benchmark_method_consistent` | Whether baseline and optimized timing methods matched |
| `valid_baseline_cases` | Number of baseline test cases with usable timing results |
| `valid_optimized_cases` | Number of optimized test cases with usable timing results |
| `speedup_calculation_error_message` | Error text if speedup could not be calculated, else `null` |
| `optimization_summary` | Framework-generated note identifying the optimizing agent and centralized evaluator |
| `score` | Computed score (see below) |

`tool_evaluation` uses this high-level shape:

| Field | Description |
| --- | --- |
| `schema_version` | Evaluation-tool result schema version. |
| `plan_fingerprint` | SHA-256 over normalized configuration, resolved task profile, plugin versions, captured original/candidate evidence for declared paths, verified scoring-image reference/ID, and the content digest of a configured replay capsule. |
| `plan` | Complete immutable plan: schema, policy, profile, ordered tool records (runtime reference, plugin version, timeout, and options), fingerprint, and source evidence. |
| `policy` | `advisory` or `required`. |
| `overall_status` | `clean`, `finding`, `incomplete`, or `not_applicable`. |
| `resolved_task_profile` | Inferred profile plus auditable explicit overrides. |
| `source_evidence` | Captured-original and candidate fingerprints plus manifest metadata, including `metadata.scoring_runtime.image_id` and `.reference` when tools run. |
| `decision` | `allowed`, `policy_satisfied`, and machine-readable reason strings. |
| `tools.<id>.capability` | Separate `engine`, `adapter`, `runtime`, and resolved `effective` checks. |
| `tools.<id>.result.execution` | `not_run`, `completed`, `tool_error`, or `timeout`. |
| `tools.<id>.result.finding` | `not_evaluated`, `clean`, `found`, or `inconclusive`. |
| `tools.<id>.result.findings` | Structured finding records. |
| `tools.<id>.result.artifacts` | Paths to retained reports/attestations. Raw stdout/stderr is omitted from the YAML summary by default. |

Execution status and finding status are deliberately independent. A sanitizer
can terminate the candidate while producing a valid finding, and a process can
exit zero without proving that an instrumented kernel ran.

The complete `plan.tools` records make the selected runtime, plugin version,
timeout, and options reconstructable from the report. Ordinary option path
strings still do not hash referenced adapter, HSACO, or input contents; include
those files in `evaluation_profile.submission_paths` or add explicit digests. A
configured rocJITsu `capsule` is handled specially: its JSON SHA-256/size are
added to source evidence and the fingerprint, while capsule validation verifies
the manifest's HSACO and blob hashes. This does not prove the capsule came from
the ordinary correctness run.

## Scoring

The score is the sum of three components:

| Component | Points | Condition |
| --- | --- | --- |
| Compilation | `20` | The kernel compiles successfully |
| Correctness | `100` | The kernel passes the correctness check |
| Speedup | `speedup_ratio × 100` | Added only when compilation *and* correctness pass |

The rules, expressed as the framework applies them:

- Compilation fails → score `0`.
- Compilation passes, correctness fails → score `20`.
- Both pass → `120 + speedup_ratio × 100`.

**Example**: A kernel that compiles (`20`), is correct (`100`), and achieves a
`1.58×` speedup scores `20 + 100 + 158 = 278`.

The speedup used for scoring prefers the explicit `speedup_ratio` written by the
evaluator (which weights each test case equally for multi-testcase tasks) and
falls back to `base_execution_time / best_optimized_execution_time` when an
explicit ratio is not present.

This is the default scoring scheme; you can define your own in `src/score.py`.

For an A/B pair, compare completed run directories with:

```bash
python3 src/tools/compare_runs.py <baseline-run-directory> <treatment-run-directory>
```

## Agent registry

Agents register themselves into a shared registry with the `register_agent`
decorator, and the framework loads only the selected agent:

```python
from agents import register_agent

@register_agent("your_agent")
def launch_agent(eval_config, task_config_dir, workspace):
    ...
    return result
```

The selectable agent names are defined by the `AgentType` enum in
`src/module_registration.py`. See [Configure agents and models](../how-to/agents.md)
for the integration steps.
