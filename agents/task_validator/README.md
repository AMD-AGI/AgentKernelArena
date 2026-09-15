# Task Validator Agent

## What This Agent Does

The **task_validator** agent validates that tasks in AgentKernelArena are correctly configured, self-contained, functional, benchmark-fair, and compatible with the protected harness boundary. It does **not** optimize kernels. It runs 12 checks and produces a framework-finalized, schema-versioned `validation_report.yaml`.

Use it to:
- Audit existing tasks before controlled comparisons or RL data collection.
- Validate new tasks before merging them into the task suite.
- Identify broken tasks (missing files, external dependencies, trivially-passing correctness checks, GPU hangs).

## How to Use

### 1. Create a run configuration

Save the following as `config_task_validator.yaml`:

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

### 2. Run

```bash
make docker-run CONFIG=config_task_validator.yaml
```

### 3. Read Results

Each task workspace contains `validation_report.yaml` plus a framework completion marker. A `validation_summary.yaml` is written to the workspace root with aggregated statistics. Resume accepts only reports with a valid schema-v3 marker and digest; the presence of an arbitrary or partial YAML file is not completion evidence.

Tasks filtered by `platform_support.status: skip` or a non-matching
`platform_support.required_arch` are skipped before workspace creation and are
not included in the validation summary counts.

### Agent Configuration

Edit `agents/task_validator/agent_config.yaml`. This portable example leaves the
model unset so the selected CLI uses its default:

```yaml
backend: claude_code          # claude_code | codex
timeout_seconds: 1200         # minimum outer limit; auto-raised for command budgets (0 disables)
python_path: null             # null -> auto-use framework-detected interpreter (recommended)

# Optional model settings for the active backend.
# claude_code: passed as `claude --model` and `claude --effort`
# codex: passed as `codex exec --model` and `model_reasoning_effort`
model: null                   # null uses the selected CLI's default
effort: max

compile_timeout: 600
correctness_timeout: 600
performance_timeout: 600
```

Task-level `compile_timeout`, `correctness_timeout`, and `performance_timeout`
override these defaults. The validator backend timeout is automatically raised
enough to cover those commands plus static review.

## Validation Checks

| # | Check | What It Verifies |
|---|-------|-----------------|
| 1 | **config_schema** | All required fields exist in `config.yaml` with correct types |
| 2 | **source_files_exist** | Every file in `source_file_path` exists in the workspace |
| 3 | **target_symbols_found** | Every function in `target_kernel_functions` is defined in source files |
| 4 | **compilation** | `compile_command` succeeds within the configured `compile_timeout` |
| 5 | **correctness** | `correctness_command` succeeds within the configured `correctness_timeout` |
| 6 | **performance** | `performance_command` succeeds within the configured `performance_timeout`, if present |
| 7 | **correctness_implementation_review** | The correctness check is meaningful (not trivially passing) |
| 8 | **self_contained** | No missing headers/imports; tasks avoid undeclared external paths and declare required runtime dependencies |
| 9 | **gpu_hang_check** | No command hangs or times out |
| 10 | **result_template_compatibility** | Command and per-case output signals can be consumed by the centralized evaluator |
| 11 | **benchmark_integrity** | Device timing, case identity, Graph/Event policy, state reset, and timed workload boundaries are scoreable and fair; missing exact replay validation is reported as WARN |
| 12 | **harness_integrity** | Protected harness logic remains protected while co-located target and Triton-JIT implementation nodes remain editable |

### Overall Status

- **PASS** — all applicable checks passed; a contract-approved `SKIP` does not prevent PASS
- **WARN** — no failures, but at least one warning (e.g., questionable correctness implementation)
- **FAIL** — at least one check failed or timed out, the report is malformed, or the validator backend failed

A verified zero-byte `torch2hip` generation placeholder uses
`SKIP/generation_placeholder` for candidate compilation and correctness. Its
performance command still runs once with `--baseline_only` to validate reference
timing before candidate generation.

`overall_status` is recomputed by the framework from normalized checks. The
validator agent's self-reported value cannot override a failed command, timeout,
missing check, invalid benchmark method, or malformed report. A validation FAIL
also makes the final CLI/post-processing gate exit nonzero.

The framework supplies the validator with authoritative scoring-lifecycle and
harness-guard facts. Baseline and candidate are measured in separate pre/post
invocations of the same protected performance entrypoint; a task-local performance
command does not need to time both implementations at once. Judgment-heavy WARN/FAIL
findings should include source-line or runtime-case evidence. Insufficient evidence is
WARN rather than an inferred failure.

---

## Task Authoring Contract

Before adding or modifying a task, read
[Task definition, schema, and authoring](../../docs/how-to/add-task.md).
It is the canonical reference and replaces the duplicated task schemas,
authoring rules, and checklist previously maintained here.

The guide covers the task definition, selected unified v2 schema, command and
result protocols, baseline/reference/candidate roles, edit boundaries,
sanitizers, examples, and the authoring workflow. Its implementation-status and
migration sections distinguish the v2 target from the current runtime.

The checks and generation-placeholder behavior above describe the current
validator. The v2 general initial-state policy and result envelope require
runtime/report-normalizer changes before use; documentation does not enable
new skip reasons or make v2 configs pass the current validator.

New tasks and material task-contract/harness changes require a fresh
framework-finalized validator report on compatible GPU hardware before PR
submission. Require `overall_status: PASS`; WARN needs an explicit
maintainer-approved justification and is not a clean pass. See
[Validate tasks](../../docs/how-to/task-validator.md) for execution and reports.
