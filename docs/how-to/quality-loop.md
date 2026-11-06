---
myst:
    html_meta:
        "description": "Audit, repair, harden, and publish schema-v3 AgentKernelArena tasks with the Codex-based quality_loop workflow and the QA auditor."
        "keywords": "AgentKernelArena, quality_loop, task audit, Codex, draft pull request, GPU kernel"
---

# Audit and harden tasks with quality_loop

`quality_loop` is a repository-level workflow for maintaining the task corpus.
It differs from a normal `agent.template`: normal agents optimize one copied task,
while `quality_loop` owns a complete multi-task campaign and a single Git branch.

It works only on schema-v3 kernel task packages (`task.yaml`, `workloads.json`,
`reference.py`, a vendored `baseline/`; see
[Task definition, schema, and authoring](add-task.md)). Discovery finds
`tasks/**/task.yaml` and ignores schema-v2 packages such as the e2e serving
tasks. It audits through the [QA auditor](qa-auditor.md); the audit rubric,
check ids and `qa_report.yaml` format are defined there.

For every selected, platform-compatible task it:

1. Audits the package in a fresh workspace with the QA auditor and accepts only
   a framework-finalized `qa_report.yaml` (with its `.qa_complete` marker) for
   the exact package digest and current rubric.
2. Records warnings; optionally repairs explicitly selected warning check ids
   (`repair_warnings`, for example `QA-README`).
3. Attempts bounded repairs for blocking findings (one attempt by default, at
   most three), retaining every finding and proposal. Each attempt needs changed
   source, a fresh QA PASS and preservation of the original task, cases and
   timing policy. Acceptance also requires an independent read-only repair
   review closing all retained findings. Unresolved failures remain in the local
   report.
4. In the default `optimize` mode, runs exactly one Codex optimization iteration
   in `solution/` and the centralized evaluator. `repair_only` stops after
   audit/repair and never optimizes, promotes a baseline or reports a repair as
   a speedup.
5. Starts a separate, read-only Codex session to review correctness evidence and
   case coverage.
6. Promotes a first-iteration candidate only when it is a same-language copy of
   the baseline, three measurements have median speedup at least 5x and all
   correctness, method and case-count gates pass. A cross-language candidate is
   never promoted.
7. Proposes targeted cases when enabled and requested by the accepting reviewer.
   Only `workloads.json` and `inputs.py` may change, existing cases stay
   identical and new cases are correctness-only. The original baseline and the
   actual optimized candidate must pass them, and the package needs a fresh QA
   PASS.
8. Requires fresh QA PASS evidence for material task changes before applying
   them, then commits accepted changes to one isolated branch and creates at
   most one draft PR. The workflow never creates GitHub issues.

## Optimization language

The run configuration has a top-level `target_language` (`triton`, `gluon`,
`hip` or `flydsl`); `null` optimizes each task in its audit language (the
baseline's language when it is a target language). It is not part of the
`quality_loop:` block. Each audit materializes the package for its audit
language, so the candidate starts as a copy of the baseline.

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
  QUALITY_LOOP_ARGS="--tasks triton/normalization/rms_norm"
```

Resume with the run ID printed in `quality_loop_runs/`:

```bash
make docker-quality-loop \
  CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--resume <run-id>"
```

The crash-safe `state.yaml` skips terminal tasks only while their files and
completed QA evidence (report digest, package digest and rubric version) still
match. `audit_report.yaml` records every warning, unresolved failure, speedup
confirmation, accepted file change, and commit. If a run has no accepted
changes, it does not open an empty pull request. `--no-publish` still requires
the GitHub preflight but suppresses push and PR creation.

For repair-only maintenance, set the following in a run config:

```yaml
quality_loop:
  mode: repair_only
  max_repair_attempts: 2
  repair_warnings: [QA-README, QA-BOUND-OVERRIDE]
```

Repairs keep scoring unchanged (`agents/quality_loop/repair_policy.py`). A
repair may add correctness-only cases, edit `inputs.py`, `reference.py`,
`README.md` and the `task.yaml` description, add a `baseline/LICENSE` and
tighten an error bound. It may not change the `task.yaml` interface, the
reference or baseline declarations, operator, platform or timeout, the vendored
baseline sources, any original workload case, or add a performance case.

## Configure roles

Optimizer/repair/case-enhancement (`backend`), reviewer and auditor have
independently configurable model, effort and timeout budgets in the
`quality_loop:` block. The `auditor` block configures the QA auditor's semantic
review (`codex`, `claude_code`, or `none` for its deterministic checks only).
Defaults are in
[`agent_config.yaml`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/quality_loop/agent_config.yaml); dated model
evidence is in [CLI defaults](../reference/agent-model-defaults.md).

## Safety boundaries

- GitHub authentication and write permission are hard preflights. Failure happens
  before branch creation or task mutation.
- GitHub credentials never enter the agent container, and Codex state is copied
  into an ephemeral writable home.
- The host refuses to commit or publish when the worktree contains a path that is
  not in the accepted per-task change manifest.
- Optimizers can edit only `solution/`; the harness guard is checked before
  evaluation. The framework-written `config.yaml`, `_aka_eval/` and `solution/`
  are never proposed as task changes.
- Reviewer output is schema checked, and modifications beyond its one YAML result
  file invalidate the review.
- QA and evaluation reports, compiled objects/libraries and ELF executables are
  filtered from task changes, and the host independently rejects generated
  outputs before a commit. Ordinary tensor/input fixtures are not rejected merely
  for being binary files.
- The top-level `tasks` selectors define the complete audit scope.
- If the original baseline or the actual optimized candidate fails proposed
  cases, the change is rejected.

See the [agent guide](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/quality_loop/README.md) for the complete
contract, role receipts and the external reviewer evidence index.

## Recorded GPU smoke (2026-09-15, historical)

This record predates schema v3 and the QA auditor: it used the former task
validator and the former task identities `hip2hip/GELU` and
`torch2hip/gpumode/14539_GELU`, which no longer exist as selectable tasks. It is
kept unchanged as dated evidence, not as current behavior.

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
| `hip2hip/GELU` | Framework-finalized PASS | Accepted | Original candidate retained; 1.001x |
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
