---
myst:
    html_meta:
        "description": "Run end-to-end serving tasks in AgentKernelArena to measure model throughput while allowing kernel source changes."
        "keywords": "AgentKernelArena, e2e, serving, SGLang, Qwen3, throughput, end-to-end optimization"
---

# Run end-to-end serving tasks in AgentKernelArena

Serving tasks measure model throughput while allowing only kernel source changes.
TP/EP/PP, weights, precision, serving flags, request concurrency, prompt lengths,
output lengths, and benchmark policy all remain fixed before either implementation
runs. Agents optimize kernel code; the framework measures whether the locked
workload runs faster.

Two maintained tasks are ready to run without additional setup:

- `e2e/qwen3_0.6b_sglang` — Qwen3-0.6B on SGLang; runs on a single MI355X.
- `e2e/qwen3_next_80b_sglang` — Qwen3-Next-80B FP8 on SGLang; requires four
  MI355X devices.

**When this workflow is complete:** `task_result.yaml` reports paired throughput
ratios for every measurement pair. A candidate with median speedup above 1.05×
across at least three fresh pairs is accepted and scored. Identical kernel hashes
cap the scored ratio at 1.0.

## Prerequisites

- A GPU host running the AgentKernelArena Docker runtime. Qwen3-0.6B requires
  one MI355X; Qwen3-Next-80B requires four. See
  [Install AgentKernelArena](../install/install.md).
- An authenticated agent CLI on the host — Codex, Claude Code, Cursor, or
  DeepSeek Harness. These agents support the `candidate.language: gpu` contract
  used by serving tasks. See [Configure agents and models](agents.md).
- Python with PyYAML installed on the host (outside Docker) for runtime asset
  preparation. The steps below create an isolated venv for this.

## Step 1: Prepare the serving runtime

Serving asset preparation runs on the host before the GPU container starts.
Create a small host-side Python environment and prepare the runtime for the task:

```bash
python3 -m venv ~/.local/share/aka-serving-venv
~/.local/share/aka-serving-venv/bin/python -m pip install PyYAML
export AKA_SERVING_PYTHON=~/.local/share/aka-serving-venv/bin/python
"$AKA_SERVING_PYTHON" -m src.prepare_serving_runtime \
  --config example_configs/e2e_qwen3_validator_mi355x.yaml
```

The asset cache defaults to `~/.cache/aka-serving`. Set `AKA_SERVING_CACHE` to
use a different location — for example on shared cluster storage.

## Step 2: Validate the task

Run the task validator to confirm baseline correctness before optimization:

```bash
make docker-run CONFIG=example_configs/e2e_qwen3_validator_mi355x.yaml
```

A framework-finalized PASS is required before proceeding. Resolve any failures
before starting the optimization run; do not skip this step.

## Step 3: Run the optimization

```bash
make docker-run CONFIG=example_configs/e2e_qwen3_codex_mi355x.yaml
```

The agent receives a wall-time budget (24 or 48 hours for serving tasks) and
runs in the standard worker container. Final evaluation alternates baseline and
candidate runs for at least three fresh pairs.

For Slurm, use `make slurm-run` and request enough scheduler wall time for the
task's budget.

## Review the results

After the run completes, check `task_result.yaml` for:

| Field | Meaning |
| --- | --- |
| `paired_ratios` | Raw throughput ratio for each baseline and candidate pair |
| `scored_case_ratios` | Ratios used for scoring (capped at 1.0 for identical kernel hashes) |
| `optimization_outcome` | Whether a candidate was accepted, rejected, or the run was incomplete |
| `process_sweep` | Leftover process check before final measurement — a failed sweep blocks scoring |

---

## Configuration reference

See [`example_configs/e2e_qwen3_codex_mi355x.yaml`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/example_configs/e2e_qwen3_codex_mi355x.yaml).
Key fields:

- `tasks`: selects the executable workload.
- `resources.gpu_groups`: indices into the scheduler-assigned GPU pool.
  `[[0, 1], [2, 3]]` creates two workers with two GPUs each. Group size must
  match the task's runtime lock. Overlap and unallocated devices fail. The same
  mapping applies to Docker and Slurm or Spur entrypoints.
- `budget.task_wall_time_s`: per-task wall budget (86400 or 172800 for 24 or
  48 hours), including setup and final evaluation.
- `budget.final_evaluation_pairs`: override the task's declared pair count
  (minimum 3, maximum 100). A run cannot reduce the task's declared count.
  This policy is set before the run and cannot change on resume.

## Related resources

- For the full budget mechanics, measurement policy, integrity boundaries, and
model-wide kernel bundle contract, see the
[end-to-end serving task reference](../reference/e2e-reference.md).
- To run a different model from the Magpie catalog, see
[Qualify a Magpie serving task for AgentKernelArena](e2e-magpie-qualification.md).
