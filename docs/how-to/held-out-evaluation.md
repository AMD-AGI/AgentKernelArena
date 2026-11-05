---
myst:
    html_meta:
        "description": "Re-run a finished AgentKernelArena schema-v3 candidate on new seeds and optional extra cases to measure kernel generalization."
        "keywords": "AgentKernelArena, held-out evaluation, generalization, GPU kernel, private tests, unseen inputs"
---

# Evaluate kernels on held-out inputs

The optional `src.held_out` module evaluates whether a finished, accepted
candidate generalizes to inputs it did not see. During an optimization run the
evaluator derives two seeds per workload case from the task and case IDs, and an
agent can run the evaluator as often as it likes, so those inputs are
observable. A held-out evaluation re-runs the same candidate on the same task
with:

- **new seeds**: every workload case gets two new seeds derived from a salt that
  the framework draws for this evaluation (`secrets.token_hex`); `--salt`
  reproduces one;
- optional **extra in-domain cases** from your own files (below).

This is a manually invoked post-run workflow; normal `docker-run` and
`docker-parallel-run` commands do not execute it. It applies to schema-v3 kernel
tasks (see [Task definition, schema, and authoring](add-task.md)); workspaces of
other task kinds are not discovered. It runs no evaluation tools and makes no
sanitizer claims.

## Enter the runtime

Run it inside the scoring runtime of the original run, because the current
scoring runtime must be identical to the one captured then:

```bash
make docker-shell
```

Run the remaining commands inside that shell from the mounted repository root.

## Run the evaluation

```bash
python3 -m src.held_out.run_heldout_eval \
  --run-dir workspace_MI355X_codex/run_<timestamp> \
  [--cases-dir held_out_cases/] [--salt HEX] \
  [--tasks TASK_ID ...] [--output-suffix _heldout]
```

| Option | Meaning |
| --- | --- |
| `--run-dir` | Completed run directory (required). |
| `--cases-dir` | Optional directory of extra cases: `<dir>/<task id>/held_out_cases.json`. |
| `--salt` | 8 to 64 lowercase hexadecimal digits, to reproduce an evaluation. Default: a fresh random salt. |
| `--tasks` | Task IDs to evaluate. Default: every framework-completed schema-v3 workspace in the run. |
| `--output-suffix` | Suffix of the output directory beside `--run-dir` (default `_heldout`). An existing output directory is never reused. |

Record the salt (it is written to the summary and the log) so the evaluation can
be reproduced. The process exits nonzero if no task was evaluated or any task
ended in `evaluation_error`.

## Extra cases

`held_out_cases.json` is `{"cases": [...]}` in the `workloads.json` case format:
each case has `id`, `axes` and `checks` (`["correctness"]` or
`["correctness", "performance"]`), with optional `scalars`, `source` and
`description`. Each case must name exactly the task's `var` axes, within their
declared `[min, max]` and `multiple_of`; use a new `id` matching
`[A-Za-z0-9_.-]+`; and differ in axes and scalars from every existing case.
Seeds are never supplied by the file. Keep these files private; they are not
part of the repository.

## What it does for each task

1. Requires the original run to be framework-complete (its completion record
   matches its report and candidate sources), its candidate accepted, and the
   scoring runtime identical to the captured one. Nothing in the original run is
   written.
2. Builds `<output>/<workspace>/workspace` from the session's frozen initial
   workspace (evaluator, contract, baseline, reference and task files exactly as
   materialized) with the evaluated candidate files (`solution/`) copied in; their
   digests must equal the report's `evaluated_candidate_sources`.
3. Changes only `_aka_eval/contract.json`: held-out seeds for every case plus any
   extra cases.
4. Runs the trusted evaluator: `validate-task` (its case list must equal the
   held-out contract's), then baseline and candidate `compile` and `correctness`,
   then `performance` in alternating fresh processes (baseline, candidate,
   baseline, candidate), averaged per case as in the final evaluation. After
   every action the harness guard and the candidate digests are rechecked. Action
   evidence is kept under `actions/`.

## Outcomes

`generalization_status` combines baseline and candidate correctness on the
held-out inputs:

| Baseline | Candidate | Status |
| --- | --- | --- |
| Pass | Pass | `both_pass` |
| Pass | Fail | `candidate_regression` |
| Fail | Fail | `both_fail` |
| Fail | Pass | `candidate_improvement` |

Missing evidence, a violated protocol or an execution error is reported as
`evaluation_error`, separately from these correctness outcomes. Only `both_pass`
with consistent timing methods and workloads receives a held-out speedup
(geometric mean over cases) and score, by the normal scoring rules.

## Outputs

Results go to `<run-dir><suffix>/`:

- `<workspace>/heldout_task_result.yaml` per task, with the status, salt, extra
  case IDs, compile/correctness flags, timings, held-out and original speedup and
  score, and any error;
- `heldout_summary.yaml`: quadrant counts, conditional correctness
  (`P(candidate correct | baseline correct)` on the held-out inputs), held-out
  speedup and original-run speedup statistics over `both_pass` tasks, and a
  per-task table;
- `heldout_eval.log`.

Keep these outputs with the run's artifacts; do not commit them. Run the unit
tests with `make check-held-out`.
