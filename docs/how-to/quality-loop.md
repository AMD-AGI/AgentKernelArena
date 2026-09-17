---
myst:
    html_meta:
        "description": "Audit, repair, harden, and publish AgentKernelArena tasks with the Codex-based quality_loop workflow."
        "keywords": "AgentKernelArena, quality_loop, task audit, Codex, draft pull request, GPU kernel"
---

# Audit and harden tasks with quality_loop

`quality_loop` is a repository-level workflow for maintaining the task corpus.
It differs from a normal `agent.template`: normal agents optimize one copied task,
while `quality_loop` owns a complete multi-task campaign and a single Git branch.

For every selected, platform-compatible task it:

1. Runs the existing task validator in a fresh workspace.
2. Records WARN results without repairing them.
3. Attempts one repair for FAIL results, revalidates from a fresh copy, and records
   the unresolved failure locally if the task remains invalid.
4. Runs exactly one Codex optimization iteration and the centralized evaluator.
5. Starts a separate, read-only Codex session to review correctness evidence and
   case coverage.
6. Promotes a first-iteration candidate only for an implemented initial-candidate
   baseline, when three measurements have median speedup at least 5x and all
   correctness/method/case-count gates pass. Provided baselines remain independent.
7. Proposes targeted cases when enabled and requested by the accepting reviewer.
   Adopt them only when their file changes are eligible, the original baseline
   and actual optimized candidate pass them, and fresh task validation passes.
8. Requires fresh framework-finalized validator PASS evidence for material task
   changes before applying them, then commits accepted changes to one isolated
   branch and creates at most one draft PR. The workflow never creates GitHub issues.

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
accepted task changes, pushes the branch, and opens the draft PR.

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
  CONFIG=example_configs/quality_loop_mi300.yaml
```

Select `example_configs/quality_loop_mi355x.yaml` on an MI355X host.

For a bounded smoke campaign:

```bash
make docker-quality-loop \
  CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--tasks hip2hip/gpumode/GELU triton2triton/vllm/triton_rms_norm"
```

Resume with the run ID printed in `quality_loop_runs/`:

```bash
make docker-quality-loop \
  CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--resume <run-id>"
```

The crash-safe `state.yaml` skips terminal tasks only while their validation
evidence and task fingerprints still match. `audit_report.yaml` records every
warning, unresolved failure, speedup confirmation, accepted file change, and commit.
If a run has no accepted changes, it does not open an empty pull request.

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
- Materialized image source trees and generated benchmark helpers are never
  copied into a task commit.
- Runtime reports, ROCmBench `*_py.pt` outputs, compiled objects/libraries, and
  ELF executables are filtered from task changes. The host independently rejects
  generated outputs before a commit, including paths accepted by an older run's
  manifest. This leaves local experiment artifacts intact; ordinary tensor/input
  fixtures are not rejected merely for being binary files.
- The top-level `tasks` selectors define the complete audit scope. Baseline
  promotion is attempted without a task-type allowlist and fails closed when the
  selected task has no promotable committed source baseline.
- Case proposals use the filename allowlist in
  [`is_case_path`](../../agents/quality_loop/filesystem.py), plus the declared
  `evaluation.workloads` file. They cannot edit candidate scopes, materialized
  source destinations or generated benchmark helpers. A file outside this
  allowlist rejects the whole proposal before the correctness gates. Input
  generators under other directory names are not automatically eligible.
- If the declared original baseline or actual optimized candidate fails the
  proposed cases, the change is rejected. A generation task's original empty
  candidate is not used as an executable baseline.

See the [agent guide](../../agents/quality_loop/README.md) and
[configuration](../../agents/quality_loop/agent_config.yaml) for the complete
contract, role receipts and external reviewer evidence index.

## Recorded GPU smoke (2026-09-15)

Job `141055`, run `20260915_200935`, completed the ordinary Docker workflow with
`--no-publish` on one MI355X in 18 minutes 24 seconds (scheduler exit `0:0`). The
controller was `27f2846189fd8fb1e35e4e92aa669eb7eb0ea7d3`; the native host-created
task worktree started at `e8ec5d6b4bc9d62b38af59a66a1797da42d3f30f`. The selected
task files matched the sealed controller packages before execution. This records
those revisions, not GPU qualification of later changes.

Both tasks retained their original 11 cases. Actual Codex calls used
`gpt-5.6-terra` with `medium` effort; optimizer/repair/case-enhancer, reviewer and
validator limits were 1800, 900 and 1200 seconds respectively. The run used the
ROCm image digest
`sha256:106a7adbeec5554b6e66a4bda0b3694af442717b9fe92754a9885520077b6f93`.
These are historical run settings, not new defaults.

| Task | Initial validator | Independent review | Measured candidate outcome |
| --- | --- | --- | --- |
| `hip2hip/gpumode/GELU` | Framework-finalized PASS | Accepted | Original candidate retained; 1.001x |
| `torch2hip/gpumode/14539_GELU` | Framework-finalized PASS | Accepted | New HIP implementation; 0.893x |

The audit reparsed 28 task-action stdout envelopes, verified four session contexts
and 62 frozen files, and checked two external reviewer indices. Both reviewers'
completed native tool commands referenced their index. Five backend roles used
anonymous stdin and retained matching, complete raw streams and terminal events;
the seven observed native processes also included two initial validator calls.
Role usage totals exclude those separately launched validators and are not a
full-run token count or a dollar-cost estimate.

Neither candidate reached the configured 5x threshold, so extra confirmation
measurements and baseline promotion were not triggered. The Torch-to-HIP reviewer
requested additional boundary coverage, which triggered one case-enhancer call.
Its proposal retained the original 11 rows and appended four correctness-only
cases, but also changed a README, evaluator and input generators outside the case
allowlist. The controller rejected the proposal before its dual correctness gate
or fresh validator ran. No enhanced task was accepted; no task changes, commits,
push, PR or matrix-count increment resulted. This exercises rejection of an
ineligible proposal, not successful case enhancement or baseline promotion.

The preserved evidence bundle contains `HANDOFF.json`, `final-verification.json`,
`review-index-verification.json`, the raw receipts, and the rejected proposal.
The handoff SHA256 is
`49aad01b200d6388d231430d784fa679936ddc54c831045c630624929e7e3d7a`.
Earlier smoke rejections remain separate evidence and were not rewritten.
