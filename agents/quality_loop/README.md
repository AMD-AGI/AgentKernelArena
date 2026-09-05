# quality_loop agent

`quality_loop` is a repository-level task curator. It audits every selected task,
attempts one repair for blocking validator failures, runs exactly one Codex
optimization iteration, sends the result to an independent Codex reviewer, and
optionally hardens an easy baseline or task cases behind fail-closed correctness
gates. It records unrepairable task defects locally and bundles all accepted task
changes into at most one draft pull request. It never creates GitHub issues.

Unlike normal Arena agents, `quality_loop` is not registered in `AgentType`.
Normal launchers operate once inside one copied task workspace; this workflow owns
the full repository campaign, isolated git worktree, resume manifest, and final PR.

## Hard preflight

A real run stops before creating a branch or modifying a task unless all of these
pass:

- `gh auth status -h github.com`
- the authenticated account has repository write permission
- `git`, `gh`, and `codex` are installed
- Git has a usable author identity for task commits
- the source worktree is clean
- the configured GPU/runtime is available through the Docker runner

Only the host-side deterministic publisher uses `gh`. The Docker runner performs
GitHub preflight and creates the audit worktree on the host, runs Codex/GPU work
without mounting GitHub credentials, then returns to the host to commit accepted
task changes, push, and open the draft PR. The main checkout is
read-only inside that container; only this run's artifact and isolated worktree
directories are writable. Codex login state is copied into an ephemeral container
home instead of being writable in place.

## Run

Inspect task selection without credentials, GPU work, or mutations:

```bash
python3 -m agents.quality_loop \
  --config example_configs/quality_loop_mi300.yaml \
  --plan
```

Run through the supported Docker environment:

```bash
make docker-quality-loop CONFIG=example_configs/quality_loop_mi300.yaml
```

Use `example_configs/quality_loop_mi355x.yaml` on MI355X (`gfx950`).

Limit a smoke run to several tasks:

```bash
make docker-quality-loop \
  CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--tasks hip2hip/gpumode/GELU triton2triton/vllm/triton_rms_norm"
```

Resume after interruption:

```bash
make docker-quality-loop \
  CONFIG=example_configs/quality_loop_mi300.yaml \
  QUALITY_LOOP_ARGS="--resume 20260803_120000"
```

`--no-publish` still requires the GitHub login/write preflight, but suppresses
push and PR creation. `--plan` is the only intentionally offline mode.

## Per-task gates

1. Run the existing 10-check validator in a fresh workspace.
2. Record WARN findings without repairing them.
3. For FAIL, allow one task-local repair and re-run the full validator in another
   fresh workspace. Record an unresolved failure locally if it still fails.
4. Measure the baseline, run one Codex optimization candidate, protect the
   harness, and use the centralized compile/correctness/performance evaluator.
5. Run an independent read-only Codex review. Deterministic evaluator failures
   always override an agent acceptance.
6. Treat the task as easy only when three measurements of the single candidate
   have median speedup at least 5x, use consistent benchmark methods and case
   counts, and the reviewer accepts logic equivalence.
7. Attempt promotion for every task selected by the run config. Capability checks,
   rather than task-type names, reject candidates without a committed source
   baseline that can pass fresh validation and the dual correctness gate.
8. Case changes may touch only test/harness paths and are accepted only when both
   the pre-audit kernel and candidate pass the updated cases.
9. Before any host commit, the complete worktree diff must exactly match the
   accepted per-task paths recorded in `state.yaml`; unexpected edits abort
   publication.

Run artifacts are written under `quality_loop_runs/<run-id>/`; the isolated audit
branch lives under `.quality_loop_worktrees/<run-id>/`. Both are ignored by Git.
Tasks pinned to another GPU architecture are reported as `platform_deferred`; run
the matching MI300/MI355X campaign to audit those tasks. If a run produces no
accepted file changes, it writes the local report without opening an empty PR.
