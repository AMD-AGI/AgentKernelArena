---
myst:
    html_meta:
        "description": "Use the AgentKernelArena task_validator agent to run 12 deterministic and review-based quality checks before using GPU kernel tasks in shared experiments."
        "keywords": "AgentKernelArena, task validator, GPU kernel, quality checks, ROCm, HIP, Triton, validation report"
---

# Validate tasks in AgentKernelArena

The `task_validator` agent checks that tasks are correctly configured,
reproducible, and functional. It doesn't optimize kernels — it audits them.
Use it to validate new tasks before merging and to audit existing tasks before
using them in controlled comparisons or RL data collection.

Before adding or modifying a task, read
[Task definition, schema, and authoring](add-task.md). That guide owns the task
schema and authoring rules. Schema-v2 tasks use the shared TaskSession lifecycle
and validator report version 4. Legacy tasks retain report version 3 while
migration is in progress. A task config version is separate from a validator
report schema version.

## Run the validator

Save a run configuration such as `config_validator.yaml` with the validator as
the agent and the tasks to check:

```yaml
agent:
  template: task_validator
tasks:
  - hip2hip/gpumode/GELU
  - triton2triton/vllm/triton_rms_norm
  # - all                     # validate every task
target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace
```

Then run:

```bash
make docker-run CONFIG=config_validator.yaml
```

Each task workspace receives a `validation_report.yaml` with per-check results,
and a `validation_summary.yaml` with aggregated statistics is written to the
workspace root. Tasks skipped by `platform_support.status: skip` or by a
non-matching `platform_support.required_arch` are filtered before workspace
creation, so they do not produce a validation report or appear in the summary
counts.

For large validation batches on a multi-GPU server, use the parallel Docker
runner. It starts one validator worker container per GPU and writes the same
reports:

```bash
make docker-parallel-run \
  CONFIG=config_validator.yaml \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix validator_parallel8"
```

Parallel resume skips only validator tasks with a framework-finalized supported
report and matching completion digest. A partial, obsolete, or manually copied
`validation_report.yaml` is rerun.

## Validator configuration

The validator's own backend and limits are set in
`agents/task_validator/agent_config.yaml`. This backend-neutral example leaves
the model unset so the selected CLI uses its default:

```yaml
backend: claude_code          # claude_code | codex
timeout_seconds: 1200         # v2 model review limit; task actions have separate budgets (0 disables)
python_path: null             # null uses the framework/container Python

# Optional model settings for the active backend.
# claude_code: passed as `claude --model` and `claude --effort`
# codex: passed as `codex exec --model` and `model_reasoning_effort`
model: null                   # null uses the selected CLI's default
effort: max

compile_timeout: 600
correctness_timeout: 600
performance_timeout: 600
```

For v2, the shared executor runs task actions with their declared
`evaluation.*.timeout_s` budgets before the model reviews the captured evidence.
The backend timeout covers semantic review. Legacy command timeout overrides
and the corresponding expanded outer budget remain supported during migration.

## `task_validator` checks

For v2 tasks, the framework first validates task data and the initial state,
then builds, checks and times the separately preserved baseline. A confirmed empty
candidate is allowed only in this initial phase. An existing candidate is also
checked, reusing the frozen baseline evidence when they are the same initial
implementation. The model reviews source semantics and the captured command
evidence; it does not supply authoritative command verdicts.

For `baseline.kind: initial_candidate`, matching initial candidate and baseline
source is expected: the framework freezes the original implementation before
optimization. This does not itself violate candidate independence. The reviewer
must separately trace the correctness oracle and any prohibited runtime access
from the candidate to protected reference/baseline code. An independent PyTorch,
analytical, or known-answer oracle can validate the shared initial implementation;
comparing that implementation with itself alone cannot. A valid unchanged candidate
does not need an optimization gain to pass correctness.

The final report contains the following checks.

| # | Check | What it verifies |
| --- | --- | --- |
| 1 | `config_schema` | All required fields exist with correct types |
| 2 | `source_files_exist` | Declared initial implementation files exist; confirmed unimplemented candidates receive a framework-owned lifecycle skip |
| 3 | `target_symbols_found` | Declared initial interfaces exist, with the same empty-candidate rule |
| 4 | `compilation` | Baseline compilation completed within its action budget |
| 5 | `correctness` | Baseline outputs satisfy the task-owned reference comparison; diagnostic policy preserves an actual numerical FAIL |
| 6 | `performance` | Baseline measurement completed with full manifest coverage and scoreable timing |
| 7 | `correctness_implementation_review` | The correctness check is meaningful, not trivially passing |
| 8 | `self_contained` | No missing headers/imports; tasks avoid undeclared external paths and declare required runtime dependencies |
| 9 | `gpu_hang_check` | No command hangs or times out |
| 10 | `result_template_compatibility` | Command and per-case output signals can be consumed by the centralized evaluator |
| 11 | `benchmark_integrity` | Every case has scoreable device timing/method metadata, stable identity, and fair state/allocation boundaries; missing exact replay validation is WARN |
| 12 | `harness_integrity` | Harness logic stays protected while co-located target and Triton-JIT implementation nodes remain editable |

## Overall status

- **PASS:** all applicable checks passed; a contract-approved `SKIP` does not
  prevent PASS.
- **WARN:** no failures, but at least one warning (for example, a questionable
  correctness implementation). Acceptable with justification.
- **FAIL:** a check failed/timed out, the backend failed, or the report contract is incomplete; the task must be fixed before merging.

The framework normalizes every report and recomputes `overall_status`; it does
not trust the agent's claimed aggregate. Passing actions must exit zero and emit
valid structured results. Only an explicitly declared baseline diagnostic policy
can accept a complete numerical-mismatch result; its numerical status remains
FAIL. Final candidates have no such exception. Stale output cannot override a failure. The final
CLI exits nonzero when any task validation fails. WARN is non-failing but requires
review.

Version 4 distinguishes a failed task from invalid evaluation evidence:

- `task_evidence_valid` records whether the captured execution history is
  internally consistent; `task_validation_failures` lists actual task failures.
- `framework_status: PASS` means the evidence and semantic report are valid,
  not that the task passed. `initial_validation_gate` and `overall_status` stay
  FAIL when a required task action fails. Quality loop may repair such a task.
- Actions after the first failed action are `NOT_RUN`. They are neither PASS
  nor an allowlisted lifecycle SKIP. Missing actions without a recorded failure,
  altered command evidence, and an incomplete model review are framework errors.
- Only a verified unimplemented initial candidate can receive
  `SKIP/candidate_unimplemented`. Its baseline still requires validation.

Reports are bound to a framework request ID, captured evidence digest, and
completion digest. A complete FAIL report remains useful diagnostic output;
only `overall_status: PASS` satisfies the clean task-validation gate.

For performance, `cuda_graph` and `cuda_event_fallback` are the only scoreable
methods. Each case must use the same method for baseline and candidate; different
cases may use different methods. CPU/host timing, missing or unknown methods,
aggregate `mixed:*` values, mismatched method pairs, candidate-triggered fallback,
invalid/partial cases, missing state restore, or demonstrably asymmetric timed work
fail `benchmark_integrity`. Missing exact output validation from the captured Graph is
WARN by itself; an observed incorrect/stale replay or a demonstrated unsafe state/reset
interaction remains FAIL. The 10-warmup/100-sample pattern is a recommended default
rather than a hard scoring rule.

The validator receives trusted framework facts for the protected harness boundary and
the pre/post scoring lifecycle. Baseline and candidate are separate invocations of the
same protected performance entrypoint, so a task runner does not need an in-process
reference timing path. Judgment-heavy WARN/FAIL results include source-line or runtime
case evidence; genuinely unavailable evidence is reported as WARN rather than inferred
as a failure.

## Result template

A validated task's **compile → correctness → performance** flow must produce results
that populate the standard template:

```yaml
task_name: "<full path relative to tasks/>"
pass_compilation: true/false
compilation_error_message: null
pass_correctness: true/false
correctness_error_message: null
base_execution_time: 0.0          # ms
best_optimized_execution_time: 0.0
speedup_ratio: 0.0
baseline_benchmark_methods: []
optimized_benchmark_methods: []
benchmark_method_consistent: true/false
valid_baseline_cases: 0
valid_optimized_cases: 0
speedup_calculation_error_message: null
optimization_summary: "Framework-generated evaluator summary"
score: 0.0
```

For the full author checklist and self-containedness rules, see
`agents/task_validator/README.md` in the repository and
[Add a task](add-task.md).
