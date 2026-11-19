---
myst:
    html_meta:
        "description": "Audit, repair, harden, and publish AgentKernelArena tasks with the Codex-based quality_loop workflow."
        "keywords": "AgentKernelArena, quality_loop, task audit, Codex, draft pull request, GPU kernel"
---

# Audit and harden AgentKernelArena tasks with quality_loop

`quality_loop` is a repository-level workflow for maintaining the task corpus.
Unlike a normal optimization agent that works on one copied task, `quality_loop`
runs a complete multi-task campaign: it validates every selected task, repairs
failures, optionally runs one optimization iteration per task, and commits
accepted changes to an isolated branch with a draft pull request.

Use this workflow to bring multiple tasks to a known-good state and surface
optimization wins across the corpus in one campaign.

**When this workflow is complete:** `audit_report.yaml` lists the status for
every task (warnings, unresolved failures, speedup confirmations, accepted file
changes, and commits). Accepted changes are committed to one isolated branch and
a draft PR is open — or no PR is created when no changes were accepted.

## Prerequisites

- **Codex CLI** installed and authenticated on the host. The workflow uses Codex
  as the generation, repair, and review agent. See
  [Install agent CLIs](../install/install.md#install-agent-clis).
- **GitHub CLI** (`gh`) installed and authenticated with write permission to this
  repository. The publisher uses `gh` to push the branch and open the draft PR.
  See [GitHub CLI documentation](https://cli.github.com/manual/).
- **A GPU host** running the AgentKernelArena Docker runtime. GPU access is
  required for validation, optimization, and evaluation.

Verify all three before starting:

```bash
codex --version
gh auth status -h github.com
gh api repos/AMD-AGI/AgentKernelArena --jq '.permissions.push'
```

## Step 1: Inspect the campaign

Preview which tasks will run before committing GPU time. The plan step is
offline and does not create a branch, require GPU access, or run any agent:

```bash
python3 -m agents.quality_loop \
  --config example_configs/quality_loop_mi300.yaml \
  --plan
```

The output lists runnable and platform-deferred tasks. Tasks with a
`platform_support.required_arch` declaration run only on the matching
architecture. Fix any configuration issues before continuing.

## Step 2: Run the campaign

Run the campaign on compatible ROCm hardware using the Docker runtime:

```bash
make docker-quality-loop \
  CONFIG=example_configs/quality_loop_mi300.yaml
```

Select `example_configs/quality_loop_mi355x.yaml` on an MI355X host.

To run a bounded smoke campaign on specific tasks before committing to the full
corpus:

```bash
make docker-quality-loop \
  CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--tasks hip2hip/GELU triton2triton/vllm/triton_rms_norm"
```

## Step 3: Review the audit report

After the campaign completes, check `quality_loop_runs/<run-id>/audit_report.yaml`.
It records every warning, unresolved failure, speedup confirmation, accepted file
change, and commit for the run.

If no tasks produced accepted changes, the workflow does not open an empty pull
request. Each task's validation logs and agent workspaces are under
`quality_loop_runs/<run-id>/` for review.

## Step 4: Customize the configuration (optional)

The default mode runs validation, optional repair, and one optimization iteration
per task. For maintenance-only runs that skip optimization and baseline promotion,
use `repair_only` mode:

```yaml
quality_loop:
  mode: repair_only
  max_repair_attempts: 2
  repair_warnings: [correctness_implementation_review, benchmark_integrity]
```

Key configuration options:

- `mode`: `optimize` (default) or `repair_only`. `repair_only` stops after
  validation and repair and never promotes a baseline or reports a repair as a
  speedup.
- `max_repair_attempts`: Repair budget per failing task (default: 1, maximum: 3).
  Each attempt needs changed source, fresh validation, and an independent review
  closing all retained findings.
- `repair_warnings`: List of warning checks to attempt to repair. Warnings are
  recorded but not repaired by default.

Existing scored cases cannot be removed, changed, or expanded during automatic
repair. Additional correctness-only controls are permitted.

See [`agents/quality_loop/agent_config.yaml`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/quality_loop/agent_config.yaml)
for the full configuration contract.

## Step 5: Resume an interrupted run (optional)

If the campaign is interrupted, resume it using the run ID printed in
`quality_loop_runs/`:

```bash
make docker-quality-loop \
  CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--resume <run-id>"
```

The crash-safe `state.yaml` skips terminal tasks only while their validation
evidence and task fingerprints still match the saved state. Start a new campaign
after changing configuration, task sources, or the environment.

---

## How quality_loop works

For each selected, platform-compatible task, `quality_loop`:

1. Runs the existing task validator in a fresh workspace.
2. Records WARN results; optionally repairs explicitly selected warning checks.
3. Attempts bounded repairs for FAIL results (one by default, at most three),
   retaining every finding and proposal. Acceptance requires an independent
   read-only repair review closing all retained findings.
4. In `optimize` mode, runs exactly one Codex optimization iteration and the
   centralized evaluator. `repair_only` stops after validation and repair.
5. Starts a separate, read-only Codex session to review correctness evidence and
   case coverage.
6. Promotes a first-iteration candidate only when: the task has an implemented
   initial-candidate baseline, three measurements have median speedup at least
   5x, and all correctness, method, and case-count gates pass.
7. Proposes targeted cases when enabled and requested by the accepting reviewer.
   Adopts them only when file changes are eligible, the original baseline and
   optimized candidate pass them, and fresh task validation passes.
8. Requires fresh framework-finalized validator PASS evidence for material task
   changes, then commits accepted changes to one isolated branch and creates at
   most one draft PR.

## Safety boundaries

- GitHub authentication and write permission are hard preflights. Failure happens
  before branch creation or task mutation.
- GitHub credentials never enter the agent container. Codex state is copied into
  an ephemeral writable home.
- The host refuses to commit or publish when the worktree contains a path not in
  the accepted per-task change manifest.
- Optimizers cannot edit task harness files; the existing harness digest guard is
  checked before evaluation.
- Reviewer output is schema checked; modifications beyond its one YAML result
  file invalidate the review.
- Materialized image source trees and generated benchmark helpers are never
  copied into a task commit.
- Runtime reports, ROCmBench `*_py.pt` outputs, compiled objects and libraries,
  and ELF executables are filtered from task changes. The host independently
  rejects generated outputs before a commit, including paths accepted by an
  older run's manifest.
- Case proposals use the filename allowlist in
  [`is_case_path`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/quality_loop/filesystem.py)
  plus the declared `evaluation.workloads` file. A file outside this allowlist
  rejects the whole proposal before any correctness gate.
- If the declared original baseline or actual optimized candidate fails the
  proposed cases, the change is rejected.

See the [agent guide](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/quality_loop/README.md)
for the complete contract, role receipts, and external reviewer evidence index.


<!-- This kind of content does not belong in our user-facing docs >
## Recorded GPU smoke (2026-09-15)

Job `141055`, run `20260915_200935`, completed the ordinary Docker workflow with
`--no-publish` on one MI355X in 18 minutes 24 seconds (scheduler exit `0:0`). The
controller was `27f2846189fd8fb1e35e4e92aa669eb7eb0ea7d3`; the native host-created
task worktree started at `e8ec5d6b4bc9d62b38af59a66a1797da42d3f30f`. The selected
task files matched the sealed controller packages before execution. This records
those revisions, not GPU qualification of later changes.

Both tasks retained their original 11 cases. Actual Codex calls used
`gpt-5.6-terra` with `medium` effort; optimizer, repair, case-enhancer, reviewer,
and validator limits were 1800, 900, and 1200 seconds respectively. The run used
the ROCm image digest
`sha256:106a7adbeec5554b6e66a4bda0b3694af442717b9fe92754a9885520077b6f93`.
These are historical run settings, not new defaults.

| Task | Initial validator | Independent review | Measured candidate outcome |
| --- | --- | --- | --- |
| `hip2hip/GELU` | Framework-finalized PASS | Accepted | Original candidate retained; 1.001x |
| `torch2hip/gpumode/14539_GELU` | Framework-finalized PASS | Accepted | New HIP implementation; 0.893x |

The audit reparsed 28 task-action stdout envelopes, verified four session
contexts and 62 frozen files, and checked two external reviewer indices. Both
reviewers' completed native tool commands referenced their index. Five backend
roles used anonymous stdin and retained matching, complete raw streams and
terminal events; the seven observed native processes also included two initial
validator calls.

Neither candidate reached the configured 5x threshold, so extra confirmation
measurements and baseline promotion were not triggered. The Torch-to-HIP reviewer
requested additional boundary coverage, which triggered one case-enhancer call.
Its proposal retained the original 11 rows and appended four correctness-only
cases, but also changed a README, evaluator, and input generators outside the
case allowlist. The controller rejected the proposal before its dual correctness
gate or fresh validator ran. No enhanced task was accepted; no task changes,
commits, push, PR, or matrix-count increment resulted. This exercises rejection
of an ineligible proposal, not successful case enhancement or baseline promotion.

The preserved evidence bundle contains `HANDOFF.json`, `final-verification.json`,
`review-index-verification.json`, the raw receipts, and the rejected proposal.
The handoff SHA256 is
`49aad01b200d6388d231430d784fa679936ddc54c831045c630624929e7e3d7a`.
Earlier smoke rejections remain separate evidence and were not rewritten.
