---
myst:
    html_meta:
        "description": "Audit, repair, harden, and publish AgentKernelArena tasks with the Codex-based quality_loop workflow."
        "keywords": "AgentKernelArena, quality_loop, task audit, Codex, GitHub issues, pull request, GPU kernel"
---

# Audit and harden tasks with quality_loop

`quality_loop` is a repository-level workflow for maintaining the task corpus.
It differs from a normal `agent.template`: normal agents optimize one copied task,
while `quality_loop` owns a complete multi-task campaign and a single Git branch.

For every selected, platform-compatible task it:

1. Runs the existing task validator in a fresh workspace.
2. Records WARN results without repairing them.
3. Attempts one repair for FAIL results, revalidates from a fresh copy, and files
   a fingerprinted GitHub issue if the task remains invalid.
4. Runs exactly one Codex optimization iteration and the centralized evaluator.
5. Starts a separate, read-only Codex session to review correctness evidence and
   case coverage.
6. Promotes a first-iteration candidate only when three measurements have median
   speedup at least 5x and all correctness/method/case-count gates pass.
7. Adds targeted cases only when both the original kernel and candidate pass them.
8. Commits accepted task changes to one isolated branch and creates one draft PR.

## Prerequisites

Install and authenticate Codex and GitHub CLI on the host. The GitHub identity
must have write permission to this repository:

```bash
codex --version
gh auth status -h github.com
gh api repos/AMD-AGI/AgentKernelArena --jq '.permissions.push'
```

The Docker launcher performs GitHub preflight and creates the audit worktree on
the host. It mounts Codex state, but never mounts GitHub credentials into the GPU
container. The main checkout is mounted read-only, while only the current run's
artifact and isolated worktree directories are writable. After the task campaign
exits, a host-side deterministic publisher verifies the recorded diff, commits
accepted task changes, creates issues, pushes the branch, and opens the draft PR.

## Inspect a campaign

Planning is offline and does not create a branch or require GPU access:

```bash
python3 -m agents.quality_loop \
  --config example_configs/quality_loop_mi300.yaml \
  --plan
```

The output lists runnable and platform-deferred tasks. A task with
`platform_support.required_arch` is run only on the matching architecture.

## Run and resume

```bash
make docker-quality-loop \
  QUALITY_LOOP_CONFIG=example_configs/quality_loop_mi300.yaml
```

Select `example_configs/quality_loop_mi355x.yaml` on an MI355X host.

For a bounded smoke campaign:

```bash
make docker-quality-loop \
  QUALITY_LOOP_CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--tasks hip2hip/gpumode/GELU triton2triton/vllm/triton_rms_norm"
```

Resume with the run ID printed in `quality_loop_runs/`:

```bash
make docker-quality-loop \
  QUALITY_LOOP_CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--resume <run-id>"
```

The crash-safe `state.yaml` skips terminal tasks. `audit_report.yaml` records every
warning, failure, issue URL, speedup confirmation, accepted file change, and commit.

## Safety boundaries

- GitHub authentication and write permission are hard preflights. Failure happens
  before branch creation or task mutation.
- GitHub credentials never enter the agent container, and Codex state is copied
  into an ephemeral writable home.
- The host refuses to commit or publish when the worktree contains a path that is
  not in the accepted per-task change manifest.
- Optimizers cannot edit task harness files; the existing harness digest guard is
  checked before evaluation.
- Reviewer output is schema checked, and modifications beyond its one YAML result
  file invalidate the review.
- External repository/image worktrees and generated benchmark helpers are never
  copied into a task commit.
- Translation/authoring tasks do not promote a generated 5x solution as their new
  baseline because that would change the task category.
- If equivalence cannot be established against a committed original kernel, the
  baseline or case change is rejected.

See `agents/quality_loop/README.md` and
`agents/quality_loop/agent_config.yaml` for the complete configuration contract.
