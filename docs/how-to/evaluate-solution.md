---
myst:
    html_meta:
        "description": "Evaluate a given kernel implementation with AgentKernelArena's standard evaluator, without running an optimizing agent."
        "keywords": "AgentKernelArena, eval_only, evaluate kernel, leaderboard, re-evaluation, GPU kernel, ROCm, Triton, HIP"
---

# Evaluate a given implementation

The `eval_only` agent evaluates an implementation you already have, such as
a kernel written by hand, a result from another system, or a candidate an
earlier run delivered, under the same conditions as an agent's. It runs no
model and needs no network access or credentials: it copies the given files
into each task workspace, and the framework then runs its usual final
evaluation of compilation, correctness, performance, integrity checks and
score.

## Prepare a solution directory

Lay out the files of each task as they appear in the task workspace, under a
directory named after the task ID:

```text
solutions/my_kernels/
└── triton/normalization/rms_norm/bf16_h8192/   # task ID
    └── solution/
        └── kernel.py                            # every file of the candidate
```

For a schema-v3 kernel task the candidate is the `solution/` directory: provide
the entry point that the task's `_aka_eval/CANDIDATE.md` names in the workspace
(`solution/kernel.hip` for HIP, otherwise `solution/kernel.py`) and every other
file it uses. Other tasks accept the
paths their configuration declares under `candidate.editable`.

The agent refuses a task's solution, and installs nothing for it, when:

- the task has no directory, or the directory holds no file;
- a file is outside the task's editable paths (for example `task.yaml`, a
  reference, a harness or a baseline file);
- an entry is a symlink or not a regular file, or a directory is one the
  evaluator ignores (`__pycache__`, `.pytest_cache`, `.git`, `build`);
- the task's entry point file would be missing after installation.

The files given under a directory declaration (`solution/` for v3 tasks)
become that directory's complete contents; the starting files there are
removed first. A file declared on its own replaces that file. An
editable path without given files keeps its starting contents.

Docker runs mount only the repository checkout, so keep the solution directory
inside it. `solutions/` is ignored by Git for this purpose.

## Run the evaluation

Select the agent and the directory in a run configuration; copy
[`example_configs/eval_only_mi355x.yaml`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/example_configs/eval_only_mi355x.yaml)
and edit the copy:

```yaml
agent:
  template: eval_only
  source_dir: solutions/my_kernels   # relative to the repository root, or absolute

target_language: triton   # the language of the given implementations

tasks:
  - triton/normalization/rms_norm/bf16_h8192

target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

| `agent` field | Default | Meaning |
| --- | --- | --- |
| `source_dir` | none (required) | Directory holding `<task_id>/<editable path>` files. |
| `timeout_seconds` | `600` | Limit for installing one task's files. |

A run uses one `target_language` (or per-task overrides, see
[Run an experiment](run-evaluation.md)), so evaluate implementations in
different languages in separate runs. Then run it as any other configuration;
no agent CLI is provisioned, and `make docker-check-agents` reports that there
is nothing to check:

```bash
make docker-run CONFIG=my_eval_only.yaml
make docker-parallel-run CONFIG=my_eval_only.yaml GPU_IDS=0,1,2,3
```

Tasks without a solution directory fail their agent step (see below), so
list only the tasks you provide solutions for.

## Read the result

Each task's `task_result.yaml` has the usual fields. Check these first:

| Field | Meaning for a given solution |
| --- | --- |
| `agent_execution.status` | `COMPLETED` when the files were installed. `FAILED` with the reason in `agent_execution.error` when the solution was refused. |
| `agent_execution.candidate_changed` | Whether the installed files differ from the task's starting candidate. `false` for a solution identical to the starting files. |
| `evaluated_candidate_sources` | SHA-256 digest of every evaluated candidate file. |
| `candidate_accepted`, `speedup_ratio`, `score` | The framework's verdict and score, as for any agent. |

When the agent fails, the framework still evaluates the task's unchanged
starting candidate, as it does after any agent failure. Whether that candidate
can be accepted depends on the [agent sandbox](agents.md#agent-sandbox):

- **Under the sandbox** (`landlock`, or `auto` where the sandbox applies), a
  refused or missing solution is never accepted. The agent fails before it
  starts its confined installer, so the launch records no attestation,
  `agent_sandbox.status` is `unattested`, and the framework rejects the
  candidate.
- **Without the sandbox** (`none`, or `auto` where it does not apply, such as
  on a host without Landlock), the
  unchanged starting candidate is judged as usual and may be accepted.

Either way that result is not the solution's: accept a result as the
solution's only when `agent_execution.status` is `COMPLETED`.

The agent also writes `.eval-only-*/manifest.json` in the task workspace,
listing each installed file with its size and SHA-256 digest, and logs the same
list in the run log. The digests match `evaluated_candidate_sources`.

## Sandbox

The agent reads and checks the solution itself, then installs the files with
a short process started like any agent CLI, through the
[agent sandbox](agents.md#agent-sandbox). Where the sandbox applies, that
process can write only the task workspace, and the result records
`agent_sandbox.status: enforced`. The installer also refuses to follow a
symlink inside the workspace.

## Limits

- End-to-end serving tasks run the candidate verifier, a model review, after
  measurement by default (see [End-to-end tasks](e2e-kernel-tasks.md)); a
  solution evaluated there needs that verifier's CLI.
- The evaluator executes the given code. Review solutions from untrusted
  sources before you run them; the worker containers are not security
  sandboxes.
