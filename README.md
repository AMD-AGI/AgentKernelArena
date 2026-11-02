# AgentKernelArena: An RL-Ready Environment for GPU Kernel Agents

**Develop and evaluate kernel optimization policies with real GPU feedback.**

AgentKernelArena (AKA) provides GPU optimization tasks, controlled execution and independently evaluated reward signals for agent reinforcement learning. Tasks range from individual operators to end-to-end model workloads. Use Arena to collect task outcomes for an external RL pipeline, evaluate trained policies, and compare agents, models, prompts or tools under the same conditions.

[Quickstart](#run-your-first-task) · [Tasks](#choose-what-to-evaluate) · [Rewards and results](#understand-rewards-and-results) · [Agent comparison](#compare-agents-or-configurations) · [Documentation](#documentation)

## Choose what to evaluate

| Task scope | What the agent can change | What Arena measures |
| --- | --- | --- |
| **Operator tasks** | Implement, translate or optimize the task's declared compute region in HIP, Triton or FlyDSL. A region may contain several kernels. | Correctness against the task's reference and GPU runtime against its baseline. |
| **End-to-end tasks** | Rewrite kernel implementations and fuse computation across the model's declared source trees. | Model correctness and serving throughput under a fixed workload. |

For end-to-end tasks, model weights, precision, TP/EP/PP, serving flags and workload settings stay fixed. Gains must come from valid compute implementation changes. The supplied example kernels are starting points; they do not limit optimization to those operators. Implementations must satisfy the task's input contract; hard-coded answers and evaluator changes are outside the allowed scope. Profiling the actual workload is encouraged, and agents choose their own tools.

Browse [operator tasks](tasks/) or the [Magpie workload catalog](tasks/e2e/magpie/README.md). The catalog includes text, multimodal and AgentX workloads, with one selectable task directory per recipe. End-to-end runs require a prepared runtime binding; use the [e2e guide](docs/how-to/e2e-kernel-tasks.md) before launching them. A catalog entry alone does not establish GPU qualification.

## Run your first task

You need a Linux AMD GPU host with GPU access through Docker, Git and an installed, authenticated agent CLI. Follow the [installation guide](docs/install/install.md) for driver, CLI and authentication setup. The example below uses Claude Code on MI355X and runs one HIP GELU task.

```bash
git clone https://github.com/AMD-AGI/AgentKernelArena.git
cd AgentKernelArena

CONFIG_PATH=example_configs/quickstart_claude_mi355x.yaml

# Check the GPU/runtime and the selected agent, then run the task.
make docker-smoke
make docker-check-agents CONFIG="$CONFIG_PATH"
make docker-run CONFIG="$CONFIG_PATH"
```

Choose a quickstart matching your physical GPU:

| GPU | Claude Code quickstart |
| --- | --- |
| MI355X | [quickstart_claude_mi355x.yaml](example_configs/quickstart_claude_mi355x.yaml) |
| MI300X | [quickstart_claude_mi300x.yaml](example_configs/quickstart_claude_mi300x.yaml) |
| MI325X | [quickstart_claude_mi325x.yaml](example_configs/quickstart_claude_mi325x.yaml) |
| RDNA4 (`gfx1201`) | [quickstart_claude_rdna4.yaml](example_configs/quickstart_claude_rdna4.yaml) |

Check the [compatibility matrix](docs/reference/compatibility-matrix.md) for runtime requirements, additional hardware profiles and task-specific limits. GPU experiments run through Docker; Slurm/Spur launches that same workflow on an allocated compute node.

## Configure a run

A **run config** selects the agent, task set, GPU and search budget. Each **task's config** defines its computation, editable sources, baseline, correctness checks and measurement policy. Changing the agent does not change the task contract.

Copy an example into an untracked local configuration:

```bash
cp "$CONFIG_PATH" config_experiment.yaml
```

For example:

```yaml
agent:
  template: claude_code
  budget_mode: time
  timeout_seconds: 3600

tasks:
  - hip2hip/GELU
  - triton2triton/vllm/triton_rms_norm

target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

Then run `make docker-run CONFIG=config_experiment.yaml`.

Task selectors are relative to `tasks/`: a full path selects one task, and a parent directory selects its descendants. Each task retains its own architecture requirements and dependencies. End-to-end tasks need their additional runtime, GPU-group and evaluation-budget settings; they cannot be added to an operator run by changing only this list.

The CLI integrations include **Codex, Claude Code, Cursor and DeepSeek Harness**. **GEAK and KernelForge** have separate integration setup. Model, provider and effort settings are integration-specific; see [agents and budgets](docs/how-to/agents.md) and the [selectable template registry](src/module_registration.py). Whole-model end-to-end tasks currently support the four CLI integrations; GEAK and KernelForge adaptation is separate work.

## Understand rewards and results

For each task, Arena preserves the baseline, runs the agent in a task workspace, and independently compiles, checks and benchmarks the delivered implementation. The evaluator writes the result; an agent's completion message is not an acceptance verdict.

The correctness reference defines valid output. The performance baseline supplies the timing comparison. They can be different implementations.

```text
workspace_<gpu>_<agent>/run_<timestamp>/
├── <task workspace>/task_result.yaml  # Per-task checks, timing and outcome
├── agent_costs/                      # Usage/cost records; unavailable values are null
└── reports/                          # Run-level CSV, JSON and text summaries
```

Read these outcomes separately:

| Question | Evidence |
| --- | --- |
| Did the agent finish its invocation? | `agent_execution` |
| Does the delivered implementation build and pass the checks? | `pass_compilation`, `pass_correctness` |
| Did it satisfy the applicable acceptance gates? | `candidate_accepted`, with method, workload and review evidence |
| How fast was it, and what score did it earn? | `speedup_ratio`, timing metadata, raw measurements and `score` |

A correct implementation can receive zero reward. For candidates eligible for performance credit, the default reward is:

```text
s = evaluator-reported speedup ratio
reward = score = max(0, 1 - 1.05 / s)
```

The score is in `[0, 1]` and becomes positive only above 1.05× speedup. Compilation and correctness are prerequisites, with no separate point bonuses. Correctness, acceptance and reward remain separate fields. See the [result and scoring reference](docs/reference/api-reference.md#scoring) and [scoring implementation](src/score.py) for eligibility, case aggregation and reward parameters.

End-to-end evaluation uses fresh paired baseline/candidate serving runs and a separate read-only verifier for scope, computation and measurement evidence. It reports model throughput rather than substituting kernel-only timing. Numerical checks and semantic review have finite coverage; passing them is not a proof against every possible invalid implementation. See [e2e measurement and results](docs/how-to/e2e-kernel-tasks.md#measurement-and-results).

Arena supplies the environment and reward signals; an external system owns training and policy updates. Retain the evaluator's original result and reward alongside any trainer-specific transformations so experiments remain comparable.

## Compare agents or configurations

There are two comparisons:

- **Inside a task:** Arena compares the delivered implementation with the preserved performance baseline.
- **Across runs:** You compare agents or agent settings on the same task set and hardware.

For an agent or policy-checkpoint comparison, create two run configs and change only the setting being tested, such as the model, prompt or enabled tool. Keep tasks, hardware, runtime, evaluation policy, reward definition and search budget matched.

```bash
cp "$CONFIG_PATH" config_control.yaml
cp "$CONFIG_PATH" config_treatment.yaml
# Edit config_treatment.yaml to make the intended agent-setting change.

make docker-run CONFIG=config_control.yaml RUN_ARGS="--run-suffix control"
make docker-run CONFIG=config_treatment.yaml RUN_ARGS="--run-suffix treatment"
```

Compare the actual run directories printed by Arena:

```bash
# Replace these paths with the directories from the two runs.
CONTROL_RUN_DIR="workspace_MI355X_claude_code/run_TIMESTAMP_control"
TREATMENT_RUN_DIR="workspace_MI355X_claude_code/run_TIMESTAMP_treatment"
python3 src/tools/compare_runs.py "$CONTROL_RUN_DIR" "$TREATMENT_RUN_DIR"
```

The suffix labels the result directory; it does not enable a capability. Repeat matched runs when assessing stochastic agents, retain the full results and interpret gains alongside measurement variation.

## Validate and extend tasks

Tasks use one [shared task contract](docs/how-to/add-task.md), regardless of their source language or agent. The main task families cover existing HIP/Triton/FlyDSL optimization, PyTorch-to-kernel implementation, translation to FlyDSL, production AITER operators and image-backed sources. End-to-end tasks use the same lifecycle with serving-specific evaluation.

- **Validate a task:** [`task_validator`](docs/how-to/task-validator.md) runs the declared actions and reviews the checks. Its framework-finalized `validation_report.yaml` is separate from an optimization result.
- **Repair task quality:** [`quality_loop`](docs/how-to/quality-loop.md) can run bounded repairs with fresh validation and independent review. Its repair-only mode preserves the scored contract and does not treat repairs as optimization gains.
- **Add a task or agent:** Follow the [task authoring guide](docs/how-to/add-task.md), [agent integration guide](docs/how-to/agents.md#add-a-new-agent) and [contribution checklist](CONTRIBUTING.md).

New tasks and material harness changes require a fresh GPU validator report with `overall_status: PASS`. WARN requires explicit maintainer-approved justification. A schema-valid task or CPU test pass does not establish GPU qualification across agents, runtimes or architectures. Validate task semantics and measurement integrity before using their scores as training rewards.

Task workspaces preserve reproducibility and separate concurrent runs. The privileged worker containers can access mounted repository and authentication state; they are not security sandboxes.

## Documentation

| Goal | Guide |
| --- | --- |
| Install the runtime and authenticate an agent | [Installation](docs/install/install.md) |
| Run, select tasks or resume an experiment | [Run an experiment](docs/how-to/run-evaluation.md) |
| Configure an agent, model or time budget | [Agents and models](docs/how-to/agents.md) |
| Run model-wide kernel optimization or an AgentX workload | [End-to-end tasks](docs/how-to/e2e-kernel-tasks.md) |
| Spread tasks across GPUs or GPU groups | [Parallel execution](docs/how-to/parallel-run.md) |
| Submit from a Slurm/Spur login node | [Cluster execution](docs/how-to/slurm-run.md) |
| Understand timing or optional evaluator-side analysis | [Benchmark methodology](docs/reference/benchmark-methodology.md) · [Evaluation tools](docs/how-to/use-evaluation-tools.md) |
| Evaluate a policy's generalization on supported tasks | [Held-out evaluation](docs/how-to/held-out-evaluation.md) |
| Interpret rewards and result fields | [Scoring and results](docs/reference/api-reference.md#scoring) |
| Build tasks from a dataset bundle | [Task builder](docs/how-to/sikl-task-builder.md) |

See the [documentation index](docs/index.rst) for the full guide and [configuration reference](docs/reference/api-reference.md) for individual fields.

## Repository layout

| Path | Purpose |
| --- | --- |
| `tasks/` | Task contracts, candidate sources and task-owned evaluation |
| `agents/` | Optimization integrations, verifier and maintenance workflows |
| `src/` | Shared task lifecycle, evaluator, scoring, runtime and reporting |
| `example_configs/` | Maintained run configurations |
| `docker/` | Runtime recipes and evaluator-side tool images |
| `tests/` | Framework and task regression coverage |

## Citation and license

If you use AgentKernelArena in research, please cite:

```bibtex
@misc{younesian2026agentkernelarena,
  title = {{AgentKernelArena}: Generalization-Aware Benchmarking of {GPU} Kernel Optimization Agents},
  author = {Sharareh Younesian and Wenwen Ouyang and Sina Rafati
            and Mehdi Rezagholizadeh and Sharon Zhou and Ji Liu
            and Yue Liu and Yuchen Yang and Hao Li and Ziqiong Liu
            and Dong Li and Vikram Appia and Zhenyu Gu and Emad Barsoum},
  year = {2026},
  eprint = {2605.16819},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2605.16819}
}
```

Report the repository revision, task selection, hardware and evaluation configuration with your results. AgentKernelArena is licensed under [Apache 2.0](LICENSE).
