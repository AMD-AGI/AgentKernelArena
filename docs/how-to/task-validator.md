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

Parallel resume skips only validator tasks with a framework-finalized schema-v3
report and matching completion digest. New `main.py` runs also require the
completed post-validator integrity receipt described below. A partial or
unfinalized report, or a report whose required integrity check is pending, is
rerun.

## Validator configuration

The validator's own backend and limits are set in
`agents/task_validator/agent_config.yaml`. This backend-neutral example leaves
the model unset so the selected CLI uses its default:

```yaml
backend: claude_code          # claude_code | codex
timeout_seconds: 1200         # minimum outer limit; auto-raised for command budgets (0 disables)
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

Per-task timeout values in `config.yaml` override these defaults. The outer
backend timeout is expanded to cover the three command budgets plus review time.

## `task_validator` checks

The `task_validator` runs the following checks in order.

| # | Check | What it verifies |
| --- | --- | --- |
| 1 | `config_schema` | All required fields exist with correct types |
| 2 | `source_files_exist` | Every file in `source_file_path` exists |
| 3 | `target_symbols_found` | Every `target_kernel_functions` symbol is defined in source |
| 4 | `compilation` | `compile_command` succeeds within `compile_timeout` |
| 5 | `correctness` | `correctness_command` succeeds within `correctness_timeout` |
| 6 | `performance` | `performance_command` succeeds within `performance_timeout`, if present |
| 7 | `correctness_implementation_review` | The correctness check is meaningful, not trivially passing |
| 8 | `self_contained` | No missing headers/imports; isolated tasks avoid undeclared external repos/paths, and repository tasks declare their upstream in `repo_url` |
| 9 | `gpu_hang_check` | No command hangs or times out |
| 10 | `result_template_compatibility` | Command and per-case output signals can be consumed by the centralized evaluator |
| 11 | `benchmark_integrity` | Every case has scoreable device timing/method metadata, stable identity, and fair state/allocation boundaries; missing exact replay validation is WARN |
| 12 | `harness_integrity` | Harness logic stays protected while co-located target and Triton-JIT implementation nodes remain editable |

Check 7 judges completeness against the **declared current task contract**: required
case identities, configured phases, runnable declared targets, and the documented
current workload scope. A missing required current case or an unchecked runnable
declared target is a failure, even if the available cases pass. Weak but substantive
current comparisons or tolerances can still warrant a warning; historical scope
notes do not excuse defects in the current checks.

Historical or legacy scenarios explicitly outside that contract remain advisory.
Their absence alone must not cause `WARN` or `FAIL`. Keep unresolved historical
flags and describe their unclaimed coverage in the review details, including in a
`PASS` report when all current correctness criteria are met. Do not add synthetic
cases, broaden the task's qualification scope, or imply that historical scenarios
were validated. Explicit current requirements take precedence over broad coverage
disclaimers. The existing rule for a concrete backend-specific unavailable branch
applies to an in-scope branch, not to unrelated historical scenarios.

This is guidance for a fresh review. It does not rewrite existing reports, approve
their warnings, or change schema normalization and overall-status aggregation.

## Overall status

- **PASS:** all applicable checks passed; a contract-approved `SKIP` does not
  prevent PASS.
- **WARN:** no failures, but at least one warning (for example, a questionable
  correctness implementation). Acceptable with justification.
- **FAIL:** a check failed/timed out, the backend failed, or the report contract is incomplete; the task must be fixed before merging.

The framework normalizes every report and recomputes `overall_status`; it does
not trust the agent's claimed aggregate. Compile/correctness/performance commands
must exit zero, and stale JSON/YAML output cannot override a failure. The final
CLI exits nonzero when any task validation fails. WARN is non-failing but requires
review.

For performance, `cuda_graph` and `cuda_event_fallback` are the only scoreable
methods. CPU/host timing, missing or mixed methods, candidate-triggered fallback,
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

## Post-validator input integrity

`main.py` snapshots the protected workspace inputs before launching the validator
and compares them again after the launcher returns, before accepting completion.
The comparison therefore follows the complete compile, correctness, performance,
and report-finalization flow, including any whole-case workers. A launcher error
still triggers this comparison. A report or `.validation_complete` marker written
before the comparison finishes cannot complete or resume the new audit.

Validation rejects changed or deleted protected inputs, changed path types or
symlink targets, and newly added protected files. Added files remain in place as
failure evidence: removing them after the commands ran would not establish which
inputs the commands used. Ordinary optimization retains its existing behavior of
discarding newly added protected files before centralized evaluation.

Tasks with `trusted_evaluation` or
`harness_protection.freeze_task_inputs: true` additionally freeze the original
bytes of their declared submission sources for this audit, using the existing
repository-aware source-path resolver. This includes kernel function bodies that
are editable during optimization. Other tasks retain their existing masked-source
boundary. Complete-input protection retains the configured runtime-output
exclusions: build outputs (including `build/private` worker data), declared
`harness_protection.runtime_outputs`, validator audit directories, and bytecode
caches do not become new protected inputs merely because the commands create them.

The framework retains a separate attempt directory outside the task workspace:

```text
<run-directory>/.validator-integrity/<workspace-name>/<attempt>/
  before.json
  sources/manifest.json       # present when raw source freezing applies
  sources/files/...
  result.json
```

`before.json` records the actual pre-launch digests, path/link observations, and
source fingerprint. `result.json` records the observed comparison, any errors,
and the report digest. Every harness digest states its scope: full file bytes,
an AST with editable Python bodies masked, or the existing co-located harness
AST mask. The separate declared-source fingerprint covers full source bytes,
presence, and source alias identity. A masked digest must not be described as a
full-byte source check. If changed topology prevents a safe after-scan, that
observation is unavailable (`null`), not a fabricated matching hash.

The workspace receives the same receipt as `validation_integrity.json`.
`.validation_complete` binds its hash, attempt identity, integrity status, and
the finalized report. Archive **the report, completion marker, and integrity
receipt together**; they allow the completed audit to be checked after copying
the workspace. Preserve the outside attempt directory as well when retaining
original source copies and incomplete attempts. Pending checks, missing or
altered required receipts, and failed receipt publication do not qualify as
complete. A finalized integrity failure is recorded as a failed audit and can
never qualify as a PASS.

`framework_guard_enforced` in the report describes the configured optimization
guard. The post-validator receipt supplies the observed enforcement evidence for
this particular audit. Existing schema-v3 reports created outside this new
`main.py` lifecycle retain their prior compatibility; they do not retroactively
prove the new post-run checks. These receipts establish input integrity, not a
filesystem/process sandbox, GPU-worker retirement, or global device idleness.

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
