---
myst:
    html_meta:
        "description": "Step-by-step examples for running AgentKernelArena experiments, A/B testing agent capabilities, auditing tasks, and resuming interrupted runs."
        "keywords": "AgentKernelArena, examples, experiment, A/B test, QA auditor, GPU kernel, ROCm, HIP, Triton"
---

# AgentKernelArena examples

These walkthroughs assume you have completed [installation](../install/install.md)
and can run the Docker runner (`make docker-smoke` passes). The agent selected
by each example must also be installed and authenticated on the host; substitute
another installed agent when appropriate. Serial examples use `make docker-run`;
multi-GPU examples use `make docker-parallel-run`.

## Example 1: Run one agent configuration on one task

Run the single-task Claude Code quickstart: optimize RMSNorm in Triton.
The language comes from the run configuration's `target_language`; the task
names none. The shipped task requires `gfx950`; on other GPUs select a task
whose `platform_support` includes your architecture.

1. Select the MI355X example:

    ```bash
    CONFIG_PATH=example_configs/quickstart_claude_mi355x.yaml
    ```

2. Run:

    ```bash
    make docker-check-agents CONFIG="$CONFIG_PATH"
    make docker-run CONFIG="$CONFIG_PATH"
    ```

3. Inspect the result:

    ```text
    workspace_MI355X_claude_code/
    └── run_<timestamp>/
        └── triton_normalization_rms_norm_bf16_h8192_<timestamp>/
            └── task_result.yaml
    ```

An illustrative excerpt from an accepted `task_result.yaml` looks like this.
The numbers show the default scoring calculation, not a measured result:

```yaml
task_name: triton/normalization/rms_norm/bf16_h8192
pass_compilation: true
pass_correctness: true
pass_tool_gate: true
benchmark_method_consistent: true
workload_consistent: true
candidate_accepted: true
base_execution_time: 0.100
best_optimized_execution_time: 0.080
speedup_ratio: 1.25
score: 0.16
```

At `1.25×`, the default score is `1 - 1.05 / 1.25 = 0.16`.
Compilation and correctness are prerequisites; a correct candidate can still
score zero. See [Scoring](../reference/api-reference.md#scoring).

## Example 2: Run an experiment on a whole task category

Run Claude Code across every task in a category, in Triton, then in HIP for
one operator. Save this configuration as `config_triton.yaml`:

```yaml
agent:
  template: claude_code

target_language: triton   # triton | gluon | hip | flydsl

tasks:
  - triton/normalization
  # - {path: triton/gemm/gemm_a8w8, target_language: hip}   # per-task override

target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-run CONFIG=config_triton.yaml
```

Every task is scored against its own baseline, so the same task set can be rerun with
another `target_language` and compared. Each task gets its own workspace under the same `run_<timestamp>/` directory, and
post-processing aggregates them into a run report.

## Example 3: A/B test an agent capability

Measure whether a new Model Context Protocol (MCP) server, skill, or prompt
change helps. Create separate control and treatment configurations, then change
only the capability under test in the treatment's agent configuration or setup.

```bash
cp config_triton.yaml config_triton_control.yaml
cp config_triton.yaml config_triton_treatment.yaml
# Enable the capability in the treatment's agent configuration or setup.

make docker-run CONFIG=config_triton_control.yaml RUN_ARGS="--run-suffix control"
make docker-run CONFIG=config_triton_treatment.yaml RUN_ARGS="--run-suffix treatment"
```

Keep the task set, hardware, runtime, reward definition and search budget matched.
The suffix only labels a run directory; it does not enable a capability.
Compare the actual run directories printed by Arena:

```bash
# Replace these with the directories from the two runs.
CONTROL_RUN_DIR="workspace_MI355X_claude_code/run_TIMESTAMP_control"
TREATMENT_RUN_DIR="workspace_MI355X_claude_code/run_TIMESTAMP_treatment"
python3 src/tools/compare_runs.py "$CONTROL_RUN_DIR" "$TREATMENT_RUN_DIR"
```

Repeat matched runs when assessing stochastic agents and inspect variation
alongside the aggregate scores. See [A/B testing](../how-to/agents.md#ab-testing-and-ablation-studies)
for more details.

## Example 4: Audit a new task

Before its PR, run the `qa_auditor` agent against a new or changed task. Save
this configuration as `config_qa.yaml`:

```yaml
agent:
  template: qa_auditor
  backend: claude_code      # claude_code | codex | none (deterministic checks only)

tasks:
  - triton/normalization/rms_norm/my_new_scenario

target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-run CONFIG=config_qa.yaml
```

Open the generated `qa_report.yaml` in the task workspace. The task must reach
`status: PASS`, which requires every blocking finding to be fixed. Warnings do
not block but carry a reason and a fix. See
[Audit tasks with the QA auditor](../how-to/qa-auditor.md) for the check table.

## Example 5: Run eight GPU workers in parallel

On an 8-GPU server, start one Docker worker per GPU and let the workers share the
same task queue. Save this configuration as
`config_parallel_claude_mi355x.yaml`:

```yaml
agent:
  template: claude_code

target_language: triton

tasks:
  - "*/normalization"   # every baseline language

target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-parallel-run \
  CONFIG=config_parallel_claude_mi355x.yaml \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix claude_parallel8"
```

The run directory contains normal per-task workspaces plus a scheduler queue:

```text
workspace_MI355X_claude_code/
└── run_<timestamp>_claude_parallel8/
    ├── .parallel/
    │   ├── pending/
    │   ├── running/
    │   ├── done/
    │   └── failed/
    ├── triton_normalization_rms_norm_bf16_h8192_<timestamp>/
    │   └── task_result.yaml
    └── reports/
        └── overall_report.txt
```

Each worker sees one logical GPU inside its container (`HIP_VISIBLE_DEVICES=0`,
`CUDA_VISIBLE_DEVICES=0`) while `ROCR_VISIBLE_DEVICES` selects the host GPU.
Post-processing runs once after all worker containers finish.

## Example 6: Audit tasks in parallel

The same queue works for `qa_auditor`. This is useful before merging a large
batch of tasks. Save this configuration as `config_parallel_qa_mi355x.yaml`, or
use `example_configs/qa_audit_all_mi355x.yaml`:

```yaml
agent:
  template: qa_auditor
  backend: claude_code

tasks:
  - "*/normalization"   # every baseline language

target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-parallel-run \
  CONFIG=config_parallel_qa_mi355x.yaml \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix qa_parallel8"
```

Each task workspace writes `qa_report.yaml`, and the run directory gets one
final `qa_summary.json` after all workers complete.

## Example 7: Resume an interrupted run

If a long run is interrupted, resume it without repeating completed tasks:

```bash
# Resume the most recent run
make docker-run CONFIG=example_configs/quickstart_claude_mi355x.yaml RUN_ARGS="--resume-latest"

# Or resume a specific run directory
make docker-run CONFIG=example_configs/quickstart_claude_mi355x.yaml RUN_ARGS="--resume-run run_20260617_101500"
```

For a parallel run, use the same resume arguments with `docker-parallel-run`:

```bash
make docker-parallel-run CONFIG=config_parallel_claude_mi355x.yaml GPU_IDS=0,1,2,3 RUN_ARGS="--resume-latest"
make docker-parallel-run CONFIG=config_parallel_claude_mi355x.yaml GPU_IDS=0,1,2,3 RUN_ARGS="--resume-run run_20260617_101500_parallel8"
```
