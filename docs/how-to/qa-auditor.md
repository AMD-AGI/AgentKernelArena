---
myst:
    html_meta:
        "description": "Use the AgentKernelArena qa_auditor agent to audit GPU kernel tasks: itemized blocking and warning findings, reproduced counterexamples, and a framework-finalized qa_report.yaml."
        "keywords": "AgentKernelArena, QA auditor, task quality, GPU kernel, ROCm, qa_report, counterexample"
---

# Audit tasks with the QA auditor

The `qa_auditor` agent audits tasks; it does not optimize kernels. Use it to
qualify a new or changed task before its PR and to audit existing tasks before
using them in a controlled comparison or in RL data collection.

Before adding or modifying a task, read
[Task definition, schema, and authoring](add-task.md). That guide owns the task
contract; this page owns the audit rubric.

## How an audit works

1. The framework materializes the task into a workspace and runs its **initial
   validation**: `validate-task`, then baseline compile, correctness and
   performance, and a copy of the baseline as a candidate when the baseline is
   in a target language. The captured evidence is trusted framework state.
2. The auditor runs its **deterministic checks** over the task package, that
   evidence and a GPU **control suite** (known-wrong implementations the
   correctness check must reject). A perturbing control that never changes a
   checked tensor (for example fp8 rounding of integer outputs, or zeroing half
   of a single-element output) is reported as not applicable instead. No model
   is involved, and the checks run before any model sees the workspace. For a
   [multi-GPU task](add-task.md#multi-gpu-tasks) the suite runs on every rank,
   and a control that fails on any rank is rejected.
3. A **model review** reads the task for problems the deterministic checks
   cannot see (semantic mismatches, unexercised input domains,
   unrepresentative workloads, false README or provenance claims). Its
   findings are warnings unless it supplies a counterexample the framework
   reproduces (below).
4. The framework verifies that the audit changed neither the initial candidate
   nor the captured evidence, then writes `qa_report.yaml`.

Kernel tasks are audited as the task itself: the audit does not need
`target_language`. The candidate is the baseline when the baseline's language is
a target language. The e2e serving tasks under `tasks/e2e/` are also audited
(see [e2e kernel tasks](e2e-kernel-tasks.md)).

## Run the auditor

Use `example_configs/qa_audit_all_mi355x.yaml` to audit every task, or save a
local config such as `config_qa.yaml` (the repository ignores `config_*.yaml`;
keep PR-specific task lists out of `example_configs/`):

```yaml
agent:
  template: qa_auditor
  backend: claude_code        # claude_code | codex | none
tasks:
  - "*/normalization/rms_norm"   # every baseline language
  # - all                     # audit every task
target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-run CONFIG=config_qa.yaml
```

For a large batch on a multi-GPU server, run one auditor worker per GPU:

```bash
make docker-parallel-run CONFIG=config_qa.yaml GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix qa_round1"
```

Resume skips only tasks whose `qa_report.yaml` is finalized (below). A partial,
edited or copied report is rerun. Tasks excluded by `platform_support` are
filtered before workspace creation and produce no report.

Qualify a task on MI355X (gfx950), where the Triton, Gluon, HIP and FlyDSL
languages are verified. Claims for other architectures are reported as
`QA-UNVERIFIED-ARCH` warnings.

## Configure the review backend

The review's defaults live in `agents/qa_auditor/agent_config.yaml`:

```yaml
review:
  backend: claude_code        # claude_code | codex | none
  model: null                 # null uses the backend CLI's default
  effort: high
  timeout_seconds: 3600       # model review
  counterexample_timeout_seconds: 1800   # each reproduction run
```

A run configuration overrides `backend`, `model`, `effort` and
`timeout_seconds` under `agent:`. `controls_timeout_seconds` (default 7200)
limits the control suite. Selecting another backend without a `model` or
`effort` uses that CLI's defaults rather than inheriting the previous backend's.

Set `backend: none` to run only the deterministic checks. The report then
carries a `QA-REVIEW-UNAVAILABLE` warning ("model review disabled by
configuration"); every blocking check still applies. Backend CLI installation
and authentication are described in [Configure agents](agents.md).

## Findings and severity

Each finding names one check, its severity, a detail and a fix. **Blocking**
findings come only from deterministic checks or from a counterexample the
framework reproduced. **Warnings** never block, and each carries the reason and
a suggested fix; read them, since a warning can be a real defect the audit could
not prove.

| Check | Severity | Meaning | Fix |
| --- | --- | --- | --- |
| `QA-AUDIT` | blocking | The audit did not run to completion (framework or environment failure, a missing or altered report, or the audit changed protected task state). | Fix the reported failure and rerun. |
| `QA-SPEC` | blocking | `task.yaml` or `workloads.json` does not load. | Fix them until the framework loads them. |
| `QA-PACKAGE` | blocking | A task file imports a production package (`aiter`, `vllm`, `sglang`, ...), uses a process, network, dynamic-import or file API, names an absolute path, includes a vendor library header in a HIP baseline, or ships a binary. | Vendor the code into the task; use only the task and allowed libraries. |
| `QA-REFERENCE` | blocking | The reference imports anything but PyTorch, standard math helpers and task-root helpers, or returns a bfloat16, float16 or float8 result. | Write `reference.py` in float32/float64 PyTorch only. |
| `QA-PROVENANCE` | blocking | The baseline's revision is not a fixed commit or release, a provenance field is a placeholder, or a non-Apache-2.0 baseline has no `baseline/LICENSE`. | Fill `baseline.provenance` with the real source, revision, path and license. |
| `QA-INITIAL` | blocking | Framework initial validation did not accept the task. | Make `validate-task` and baseline compile, correctness and performance pass; read the captured evidence. |
| `QA-POSITIVE` | blocking | A copy of the baseline fails as a candidate. | Make the baseline satisfy its language's candidate rules (kernels only, no PyTorch computation, capturable launches). |
| `QA-TIMING-METHOD` | blocking | A performance case was not timed as a cold-cache CUDA graph. | Make the baseline graph-capturable on the current stream. |
| `QA-TIMING-FLOOR` | blocking | A performance case takes the baseline under 10 us. | Use larger production shapes; keep small shapes correctness-only. |
| `QA-SPAN-EVIDENCE` | blocking | A task with the [production-range span exemption](add-task.md#production-range-span-exemption) has a `perf_min` or `perf_max` endpoint case that the baseline run did not measure or that takes under 10 us. | Remeasure and correct the evidence's perf range; keep faster shapes correctness-only. |
| `QA-CONTROLS` | blocking | A known-wrong control implementation passes correctness. | Strengthen inputs, cases or the error bound. |
| `QA-CONTROLS-RUN` | blocking | The control suite timed out, wrote no report or did not complete. | Fix the input, reference or baseline failure it reports. |
| `QA-INPUTS` | blocking | An input draw is degenerate (constant, or exactly representable in a lower precision than declared). | Use a distribution whose values vary. |
| `QA-COUNTEREXAMPLE` | blocking | A wrong implementation that the task accepts fails an in-domain case, reproduced by the framework. | Add the counterexample's input domain to the workloads or tighten the bound. |
| `QA-POSITIVE-UNAVAILABLE` | warning | The baseline language (TileLang) cannot be copied as a candidate, so no positive control ran. | Consider a Triton baseline. |
| `QA-BOUND-OVERRIDE` | warning | `compare` departs from `baseline_relative` with `k: 2`, or sets `baseline_min_sqnr_db`, a violation fraction or a `decode` hook. | Justify the override in README.md or return to the default. |
| `QA-BOUND-CALIBRATION` | warning | The reference evaluated with float32 instead of float64 (a correct implementation with float32 accumulation and library reduction orders) fails a non-exact bound (`baseline_relative`, `sqnr`, `tolerance` or `output_rounding`). | Unless the baseline is known to be more accurate than float32 accumulation, loosen the bound (`sqnr` `min_db`, `tolerance`, or `k` and `max_violation_fraction`) and justify it in README.md. |
| `QA-NONDETERMINISTIC` | warning | The baseline is not bitwise deterministic across repeated runs. | Informational: `baseline_relative` compares against one baseline run per draw. |
| `QA-SPAN-EXEMPTION` | warning | The performance cases use the production-range span exemption instead of the 100× span rule; the finding shows the observed range, the perf range, the share of observed calls it covers and the evidence source. | Informational: confirm the evidence and that no context or sequence axis could meet the 100× rule. |
| `QA-README` | warning | The task has no README.md. | Describe the operator, production source and workload origin. |
| `QA-UNVERIFIED-ARCH` | warning | `platform_support.required_arch` names an architecture the audit did not run on. | Audit on each claimed architecture or document it as unverified. |
| `QA-REVIEW` | warning | A model-review finding without a reproduced counterexample. | Review it; supply a counterexample to make it blocking. |
| `QA-REVIEW-UNAVAILABLE` | warning | The review is disabled, failed, timed out or wrote an unusable draft. | Rerun with a working backend, or accept a deterministic-only audit. |

The authoritative table is `agents/qa_auditor/rubric.py`; the report records its
`rubric_version`.

## The counterexample rule

A model opinion alone never fails a task. To make a review finding blocking, the
reviewer supplies a complete wrong candidate solution in the baseline's language
and one additional workload case inside the declared interface domain (every
`var` axis within its `[min, max]` and `multiple_of`). In a fresh copy of the
workspace the framework runs the task's own correctness check twice: the
solution must **pass** the task as written, and must then **fail** only the added
case. If both reproductions succeed, the finding becomes `QA-COUNTEREXAMPLE`
(blocking); otherwise it stays a `QA-REVIEW` warning with the reason the
reproduction failed. At most three counterexamples are verified per audit.

## Report format

Each workspace receives `qa_report.yaml` and a `.qa_complete` marker. The marker
holds the report's SHA-256 and status; a report without a matching marker, an
edited report, a symlink, or a status inconsistent with the findings is treated
as absent. Key fields:

```yaml
qa_report_version: 1
rubric_version: "2026-11-05.3"
task_id: triton/normalization/rms_norm/bf16_h8192
task_package_sha256: ...        # the exact task package audited
target_language: triton         # the candidate language used for the positive control
audit_request_id: ...
status: PASS                    # PASS | FAIL
blocking: 0
warnings: 2
checks: {QA-SPEC: PASS, QA-README: WARN, ...}   # PASS | WARN | FAIL per check
findings:
  - id: QA-BOUND-OVERRIDE
    severity: warning
    title: ...
    detail: ...
    evidence: {}
    fix: ...
evidence: {...}                 # raw control and action evidence
```

A report qualifies only the exact `task_package_sha256` and the current
`rubric_version`. Any change to the task package, or a new rubric, requires a new
audit.

## PASS rule

`status: PASS` requires zero blocking findings: every blocking finding must be
fixed and the task re-audited. Warnings never block. A `FAIL`, a timed-out or
partial audit, a stale report (different package hash or rubric), or a skipped
task does not satisfy the pre-PR gate in [CONTRIBUTING.md](https://github.com/AMD-AGI/AgentKernelArena/blob/main/CONTRIBUTING.md).

## Run summary

After the run, `qa_summary.json` is written next to the task workspaces:

```json
{"tasks": 12, "pass": 10, "fail": 1, "invalid_reports": ["<workspace>"],
 "blocking_by_check": {"QA-CONTROLS": 1},
 "warnings_by_check": {"QA-README": 3},
 "results": [{"task_id": "...", "status": "PASS", "blocking": 0, "warnings": 1,
              "workspace": "...", "blocking_findings": []}]}
```

The audit run succeeds only when every task has a valid report and passed.
Keep reports and summaries in their run directories or an external artifact
archive with the source, config and runtime identity. Do not copy them into
`tasks/` or commit them; describe the outcome in the PR. Follow the
[commit checklist](https://github.com/AMD-AGI/AgentKernelArena/blob/main/CONTRIBUTING.md#repository-cleanliness-before-committing)
before staging task changes.
