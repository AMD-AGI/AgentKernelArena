---
myst:
    html_meta:
        "description": "The unified AgentKernelArena task definition, config.yaml v2 design, command and evaluation contracts, migration guide, and task authoring workflow."
        "keywords": "AgentKernelArena, task schema, task definition, task authoring, baseline, candidate, sanitizer"
---

# Task definition, schema, and authoring

Read this document before adding or modifying a task, its configuration,
reference, input generator, harness, or benchmark. It is the canonical task
contract and replaces the previous task-family-specific configuration guidance.

## Status and scope

**All 438 retained tasks use schema v2; GPU qualification is in progress.**
The shared declaration parser, action/result protocol, materialization,
baseline session, evaluator and validator are connected through
`src/task_run.py`. Normal runs and quality_loop use the same final evaluation
and initial validation entrypoints. CPU process fixtures cover their integration;
this does not certify GPU task quality or an agent's optimization capability.
The migrated task families include SIKL, HIP, Triton, FlyDSL, generation, and
image-backed tasks. Legacy execution/report compatibility remains for old
workspaces and external configs; new or changed repository tasks must use v2.
Do not replace a config without implementing its task-owned actions.
See [Migration](#migration) before changing
an executable task. Task config version 2, runner protocol 1 and validator report
version 4 are separate contracts.

The design uses one task `config.yaml`, task-owned evaluation scripts, and
optional workload data. It does not require a second `definition.yaml`, an
agent-specific driver, or a Python callback API. All task families use the same
schema, including existing optimization, generation, and image-backed
tasks and the new SIKL operator-to-FlyDSL tasks.

Use the existing task directory as the package boundary. A typical layout is:

```text
tasks/<suite>/<task>/
  config.yaml
  README.md                 # optional extended task instructions
  source/                   # editable implementation files
  scripts/                  # protected evaluation/reference/input code
  workload.json             # optional case data
```

Other layouts, including a root-level `kernel.py` or a combined source/harness
file, are supported when their paths and edit boundaries are explicit.

## What a task defines

A task is a reproducible optimization problem for an operator or implementation
region. An operator may launch multiple GPU kernels. One task may cover multiple
shapes and input distributions; it must not silently score only a convenient
subset.

| Concept | Task responsibility |
| --- | --- |
| Semantics | State what is computed, the input/output interface, layouts, dtypes, supported cases, and allowed implementation dependencies. |
| Candidate | Identify the implementation to produce or improve, its language, editable boundary, entrypoints, and initial state. |
| Baseline | Identify the implementation used for performance comparison. Preserve it independently of candidate edits. |
| Reference | Supply the mathematical or statistical oracle and the task's comparison rule in protected evaluation code. |
| Workloads | Supply reproducible case identities, parameters, seeds, and input generation or bundled input data. |
| Evaluation | Supply commands that build, check, and time the declared role and emit machine-readable results. |
| Optional analysis/export | Supply task-specific sanitizer or export commands when the corresponding common tooling needs them. |

The numerical reference and performance baseline are distinct roles, even when
one implementation serves both. A production AITER operator can be the baseline
while a task-local FP32 calculation supplies the reference. GEMM element-wise
checks, MoE SQNR checks, and sampling distribution checks need not share a
numerical rule.

The run configuration chooses the agent, GPU, budget, and evaluation-tool
policy. Agents decide how to search. The centralized evaluator owns acceptance,
timing aggregation, scores, and final reports. An agent's success message is not
an evaluation result.

## Minimal configuration

An existing HIP implementation can use:

```yaml
schema_version: 2
description: Optimize the HIP implementation while preserving its semantics.
candidate:
  language: hip
  editable: [source/kernel.hip]
evaluation:
  runner: [python3, scripts/evaluate.py]
```

Defaults keep common tasks short:

- The task ID is its full directory path relative to `tasks/`, retained when the
  task is copied into a run workspace. Directory names do not select behavior.
- `candidate.initial_state` defaults to `implemented`.
- For an implemented candidate, `candidate.initial_language` defaults to
  `candidate.language`; declare it when converting an existing implementation.
- `baseline.kind` defaults to `initial_candidate` for an implemented candidate,
  and to `provided` for an unimplemented candidate. The latter still requires a
  working task-provided baseline command path.
- `baseline.correctness_policy` defaults to `required`.
- `evaluation.timeout_s` defaults to `3600` per action.
- Evaluation tools and exports are disabled/absent unless configured.
- A task-local `README.md`, when present, provides additional instructions.

Only `schema_version`, `candidate`, and `evaluation` are always required.
`candidate.language` and a nonempty `candidate.editable` are required.
Evaluation must provide a runner or explicit commands for every action below.
Descriptions and explicit entrypoints are recommended; a file-level C++ task
need not invent a single function entrypoint when its harness builds a library.

## Schema reference

### Task metadata and candidate

| Field | Type/default | Meaning |
| --- | --- | --- |
| `schema_version` | Integer, required: `2` | Selects this task contract and its command/result protocol. Unknown versions must be rejected. |
| `description` | Optional string | Agent-independent objective and semantic summary. |
| `instructions` | Optional list of relative paths | Additional protected instruction/contract files. No provider prompt syntax is required. |
| `kernel_identity` | Optional mapping | `logical_operator` and `source_owner` identify the operator and upstream owner for reports/exports. They never implicitly determine an entrypoint symbol. |
| `candidate.language` | Required string | Required final implementation backend, such as `hip`, `triton`, or `flydsl`. It is not inferred from a `.py` extension. |
| `candidate.initial_state` | `implemented` or `unimplemented` | Whether the starting candidate is an existing implementation or a generation target. |
| `candidate.initial_language` | Optional string | Starting implementation backend. Omit for an unimplemented candidate; it describes the candidate, not the baseline. |
| `candidate.editable` | Nonempty list | Relative implementation paths or scoped edit declarations; see below. |
| `candidate.entrypoints` | Optional list | Objects with `file`, `kind`, and, where applicable, `symbol`. Kinds are `function`, `builder`, `class`, or `executable`. |

Unknown core fields and invalid types must fail schema validation. A backend
name does not promise that every agent or analysis tool supports it; capability
checks must report unsupported combinations explicitly.

A string in `candidate.editable` means an entire implementation file. Use a
scoped declaration for files containing both implementation and harness code:

```yaml
candidate:
  language: triton
  editable:
    - path: benchmark.py
      scope: symbols
      symbols: [compute_kernel]
      allow_new_helpers: true
  entrypoints:
    - {file: benchmark.py, kind: function, symbol: compute_kernel}
```

A mapping's `scope` is `file`, `symbols`, or `tree`. `symbols` requires a nonempty
symbol list; `allow_new_helpers` defaults to `false` and applies only to
implementation helpers under a symbol-scoped boundary. `tree` permits an
implementation subtree, not its tests or build/evaluation policy. Broad tree
access must not override a protected harness path. Generated build artifacts
are not editable source declarations.

Entrypoint files must lie within the candidate boundary. Declared symbols must
exist in the final candidate. Initial validation verifies the starting
implementation's interface; a translation may intentionally introduce a new
final entrypoint, and a verified unimplemented candidate has none yet. A builder
is a host entrypoint that prepares a case-specific launch; its tile/split
choices may vary by case. Its
signature and the returned launch interface belong in the task's instructions
and harness. A builder symbol is not necessarily a GPU kernel symbol.

### Baseline and reference

| Field | Type/default | Meaning |
| --- | --- | --- |
| `baseline.kind` | `initial_candidate` or `provided` | Use a frozen starting implementation, or a separately provided baseline selected by the task runner. |
| `baseline.language` | Optional string | Baseline implementation backend when useful; production dispatch may involve multiple backends. |
| `baseline.source_files` | Optional list of relative paths | Read-only production/reference material for the agent. Merely copying these files does not establish what the baseline command executes. |
| `baseline.correctness_policy` | `required` or `diagnostic` | Whether a baseline numerical mismatch rejects initial task validation. Candidate correctness is always required. |
| `baseline.diagnostic_reason` | Required for `diagnostic` | Explain why this implementation remains a useful performance reference despite its known numerical mismatch. |

For `initial_candidate`, the framework freezes the original implementation and
runs baseline actions in a separate workspace containing that snapshot.
Candidate actions run in the working candidate workspace. Both expose the same
relative task layout; the runner must not look outside its assigned workspace
to find the other role. `initial_candidate` is invalid for an unimplemented
candidate.

For `provided`, the runner explicitly invokes the declared baseline, including
its required runtime dependencies. For example, SIKL baseline scripts call the
installed AITER package; the materialized `aiter_source/` tree is explanatory
source. Record the actual runtime version and dispatch used. Avoid a source
copy that shadows the installed package on Python's import path.

Reference code, input generators, tolerances, and comparison functions remain
protected task files. They are not editable candidate dependencies. State the
allowed dependency policy in the task instructions and enforce it in the
harness; do not put a family-specific numerical tolerance in the common schema.

### Evaluation commands and timeouts

| Field | Type/default | Meaning |
| --- | --- | --- |
| `evaluation.runner` | Nonempty argv list | Common command prefix; the framework appends the role/action arguments in the next section. |
| `evaluation.workloads` | Optional relative file path | Case data consumed by the runner. Existing JSON/JSONL data can stay in its current format. Omit when the runner constructs its cases. |
| `evaluation.timeout_s` | Positive integer, default `3600` | Total deadline for each action, including all commands in that action. |
| `evaluation.task` | Optional action mapping | Overrides for `validate-task`: `commands` and/or `timeout_s`. |
| `evaluation.baseline.<action>` | Optional action mapping | Overrides for baseline `compile`, `correctness`, or `performance`. |
| `evaluation.candidate.<action>` | Optional action mapping | Overrides for candidate `compile`, `correctness`, or `performance`. |

Each `commands` value is a nonempty list of argv lists. An override replaces the
runner invocation for that action; role/action arguments are not appended to an
override. Commands execute sequentially and stop on the first failure. An
omitted timeout inherits `evaluation.timeout_s`. A timeout-only override still
uses the runner. Without a runner, all seven actions require explicit commands.

```yaml
evaluation:
  runner: [python3, scripts/evaluate.py]
  timeout_s: 3600
  candidate:
    compile:
      commands:
        - [python3, scripts/build_candidate.py]
      timeout_s: 600
```

Commands run with the workspace root as their working directory. V2 argv lists
are not shell command strings: pipes, redirects, and environment setup belong in
an explicit task-local wrapper when needed. A zero exit code means that action
succeeded; failures must exit nonzero. Missing required commands, timeouts,
malformed results, and stale result files are errors, not implicit skips.

### Workspace, platform, tools, and exports

These optional fields use the same schema for small isolated tasks and larger
image-backed tasks. Declared pinned Git sources remain an acquisition option;
the unmaintained `tasks/repository` suite has been removed.

| Field | Meaning |
| --- | --- |
| `workspace.sources` | List of declared source acquisitions. An image source has `kind: image`, `image_path`, `destination`, and optional `exclude`. A Git source has `kind: git`, `url`, immutable `revision`, and `destination`. |
| `workspace.setup` | Ordered list of argv lists, run after materialization and before baseline capture. Setup must be repeatable and must not be silently delegated to an agent. |
| `workspace.timeout_s` | Positive integer, default `3600`, bounding source materialization and setup together. A timeout aborts setup before baseline capture. |
| `platform_support` | `required_arch` accepts one exact architecture string or a nonempty list of alternatives; `status` is `active \| skip`, with optional `skip_reason`. Omission declares no architecture restriction; it does not prove sanitizer support. |
| `evaluation_profile` | Optional analysis-tool profile overrides when inference from candidate language, paths, and artifact kind is insufficient. See the tool guide for supported keys. |
| `evaluation_tools` | Optional task-side tool commands/options. Only the run config enables tools and sets their policy/runtime. |
| `exports` | Optional list of `{format, output, command, timeout_s}` objects. `command` is an argv list, `output` is a relative artifact path, and `timeout_s` defaults to `60`. |

For example, `required_arch: [gfx942, gfx950]` permits either architecture;
`required_arch: gfx950` remains valid for a single architecture. Empty lists,
duplicate names, and wildcards are invalid. Main scheduling, quality-loop
planning, and actual runtime binding use the same rule. Supporting another
architecture requires real task validation there with the full workload and
numerical gates; merely adding its name is not qualification. Record the tested
source, image, and GPU separately for each architecture. An existing baseline
session cannot resume on a different GPU architecture or runtime, even when both
architectures are allowed by the task.

A Git revision must be a pinned commit, not a floating branch. Image source
paths name locations in the selected runtime image; the framework records that
image's immutable identity and materialized source evidence. An image source
path may be absolute **inside the image**. This is an acquisition exception,
not permission to use absolute checkout/host paths in evaluation commands.
Each source `exclude` is a path relative to that source root, including its
descendants. It is not a basename pattern: `jit` does not exclude `aiter/jit`.
Exclude disposable runtime caches explicitly while retaining required source.

All other task paths are relative to the task workspace root after setup.
Reject absolute paths, traversal, and symlink resolutions escaping that root.
Keep directory components when installing generated candidates: a declaration
of `source/kernel.py` must not be installed as root-level `kernel.py`.
An invalid submitted path must still produce a failed evaluation report. The
framework records `candidate_source_error` instead of following an escaping
path to hash it; that submission cannot receive acceptance or a speedup.
Repository destinations also count toward paths: `upstream/src/kernel.hip`
always refers to that path from the workspace root, never an implicitly changed
repository working directory. Source destinations must not overwrite task
configuration or harness files.

Exports run through common framework post-processing after final evaluation.
They read the accepted candidate and framework-finalized results; they do not
supply correctness or scores. Each exporter documents the status it accepts,
its required inputs, and output format. Failed candidates must not be exported
as accepted solutions. Preserve failure reports as diagnostics. Exporters must
honor the configured candidate/workload paths instead of assuming `kernel.py`
and `workload.json`. Export failures are reported separately from numerical
correctness. An SIKL solution export must work for any supported optimizing
agent, not only Forge.

## Command and result protocol

The common runner is a CLI contract, not a required Python module interface.
Task scripts may delegate to pytest, CMake, native executables, or existing
harnesses. No particular filename is required.

The framework sets `ARENA_EVAL_PHASE` to `task_validation` or
`candidate_evaluation`. During task validation, implemented candidates are
checked in their declared initial language/interface; final evaluation enforces
the requested target language/interface. This phase is framework-controlled,
not inferred from an agent's success message or whether a target file exists.
An agent invoking checks during optimization uses `candidate_evaluation`.

| Arguments appended to `evaluation.runner` | Responsibility |
| --- | --- |
| `validate-task` | Check task data, references, dependency availability, candidate initial state, and enumerate the complete case manifest. It does not certify an optimized candidate. |
| `baseline compile` | Build/syntax-check the selected baseline and verify the executable dependency path. |
| `baseline correctness` | Compare baseline outputs against the task's reference using the task's rule. |
| `baseline performance` | Measure the selected baseline over the declared cases. |
| `candidate compile` | Actually build or syntax-check candidate targets; prepare required JIT/build validation. |
| `candidate correctness` | Execute the candidate and compare against the task's reference for all declared correctness cases. |
| `candidate performance` | Measure the candidate's actual timed execution over the declared performance cases. |

The same input definitions and timing policy apply to baseline and candidate.
Correctness and performance can have different documented case sets, but every
performance case must have associated correctness coverage. Baseline and
candidate performance manifests must match exactly by ID, shape, dtype, and
semantic parameters. The framework must obtain the manifest independently of
the candidate's claimed performance rows.

### Structured command results

Alongside human-readable logs, each invocation must emit exactly one stdout
line beginning with `ARENA_EVAL_RESULT=` followed by a JSON object. This envelope
is parsed by the v2 executor; legacy parsers do not accept it. Its fields are:

| Field | Meaning |
| --- | --- |
| `protocol` | Required string `arena-eval-v1`; distinct from task/report schema versions. |
| `role` | `task`, `baseline`, or `candidate`; must match the invocation. |
| `action` | `validate-task`, `compile`, `correctness`, or `performance`; must match the invocation. |
| `status` | `PASS` or `FAIL`. Only the framework decides lifecycle skips and diagnostic acceptance. |
| `cases` | Array of case records. `validate-task` enumerates the manifest with each case's `checks` list (`correctness`, `performance`, or both). Correctness/performance records cover their declared manifest. A whole-build compilation check may use an empty array. |
| `reason` | Required explanation for a failure; optional on success. |
| `failure_kind` | Optional failure classification. A diagnostic baseline numerical failure must use `numerical_mismatch` at the report level and on every failed case. Missing or other classifications cannot use that exception. |
| `metadata` | Optional object with task-specific diagnostic evidence; it cannot override status, identity, coverage, or timing checks. |

Each case uses a stable `test_case_id` and declared `shape`, `dtype`, and
`params` as applicable. Correctness records include `status` and task-defined
`metrics`; no universal error formula is imposed. Performance records include
`status`, finite positive `execution_time_ms`, and `benchmark_method` using the
shared methodology's supported device-timing modes. Additional task metrics
belong in `metrics`/`metadata`, not new top-level acceptance switches.

For example, after the prefix a candidate performance command could emit this
object on a single line:

```json
{
  "protocol": "arena-eval-v1",
  "role": "candidate",
  "action": "performance",
  "status": "PASS",
  "cases": [{
    "test_case_id": "m16_n32_k6144",
    "shape": [16, 32, 6144],
    "dtype": "bfloat16",
    "params": {"transpose_b": true},
    "status": "PASS",
    "execution_time_ms": 0.012,
    "benchmark_method": "cuda_graph"
  }]
}
```

This is an illustrative result, not a measured latency. Its outer status cannot
override a failed case, missing/duplicate case, nonzero exit, timeout, malformed
JSON, or invalid timing. For a multi-command action, each command emits its own
matching envelope; the framework requires all commands to pass and merges
case records, rejecting duplicates. Wrappers around legacy harnesses translate
fresh legacy output to this envelope and preserve failure exit codes.

The framework records config/source identities, actual environment, action,
command, exit status, and tool evidence. Commands never author or overwrite
`task_result.yaml` or `validation_report.yaml`. Scores and speedup aggregation
remain in the centralized evaluator; defining v2 must not silently change the
existing scoring formula. See the [result reference](../reference/api-reference.md#result-schema-task_resultyaml)
and [benchmark methodology](../reference/benchmark-methodology.md).

The `validate-task` report must include `metadata.candidate_state` with
`implemented` or `unimplemented`, determined by checking the actual initial
implementation files and their interface. The framework rejects missing or
conflicting state evidence and a mismatch with the configuration declaration.
This is task-check evidence, not an agent-controlled permission to skip final
evaluation. The validator also audits the state check itself.

During final evaluation, successful compilation and correctness evidence bind
to the same candidate source content subsequently measured. A source change
invalidates earlier checks and requires compilation/correctness again. Command
logs and failure evidence are retained outside the candidate workspace.

## Initial task validation and final candidate evaluation

These are separate lifecycle stages for every task family.

| Stage | Implemented initial candidate | Unimplemented initial candidate |
| --- | --- | --- |
| Initial task validation | Validate the existing candidate as its declared initial language, plus baseline, reference, inputs, and harness. | Verify the declared generation state; validate baseline, reference, inputs, and harness. Candidate symbols/build/correctness may be deferred by the framework. |
| Final candidate evaluation | Require the requested final language, real entrypoints, compilation, correctness, and performance. | The same requirements; no generation-state exemption remains. |

`initial_state` is an author declaration, not a perpetual bypass. The validator
must confirm it against the actual starting files and entrypoints. A missing
candidate may be a declared generation target; missing baseline/reference
files are not. Once a candidate is submitted, its commands must execute that
candidate and must fail if it is absent or incomplete. They must not silently
fall back to the baseline.

With `baseline.correctness_policy: diagnostic`, a baseline correctness command
still reports the real numerical mismatch and exits nonzero. The framework
may accept that specifically identified mismatch during initial validation and
must preserve it in the finalized report. Compilation failures, crashes,
missing cases, nonfinite outputs where finite outputs are required, and
unavailable dependencies are not covered by this policy. Candidate correctness
always uses the full task rule. Establish evidence that a valid candidate can
satisfy the task; baseline executability alone does not establish feasibility.

The deterministic validator/report normalizer derives lifecycle decisions from
the captured session evidence. A verified empty starting candidate uses
`candidate_unimplemented` in the version-4 report. The model reviews task
semantics and writes `validation_report.draft.yaml`; the framework verifies its
request/evidence identity and writes the official report. A new prompt-only
`SKIP` reason is insufficient.

Session state, the original harness boundary and action evidence are retained
outside the candidate workspace. Resume verifies that state and preserves the
original baseline; it does not capture a modified candidate as a fresh baseline.
Optimization completion records bind the final report to the delivered candidate.
An agent CLI failure is recorded independently, and any retained candidate still
receives the ordinary final checks. Export failures remain separate from numeric
scores and make the delivery incomplete.

## Benchmark and edit-boundary contracts

- Inputs, references, expected outputs, tolerances, case sets, and timing policy
  are protected from optimization agents. Implementation changes cannot alter
  the acceptance rule.
- Preserve equivalent work, output allocation, scratch preparation, state
  restoration, and synchronization boundaries for baseline and candidate.
  Document what runs inside the timed invocation and what is prepared once.
- Verify the actual timed/replayed path according to the task's numerical
  contract. Output finiteness or a changed output alone does not establish that
  the operator was computed correctly. A production baseline diagnostic must
  not relax candidate replay correctness.
- Keep sanitized builds and their timing separate from official scoring builds.
  Associate each tool result with the actual candidate and covered cases/kernels.
- Use canonical timing helpers from `src/tools/perf/`; do not hand-edit committed
  helper stubs or `AKA-GENERATED` regions. See the
  [performance helper instructions](../../src/tools/perf/README.md).
- Use a shared path/edit-boundary interpretation across the validator, prompt
  builder, harness guard, and agent installation path. A policy about whether a
  boundary violation stops a run is separate from declaring that boundary;
  schema migration does not silently change that run policy.

## Agent contract

Tasks must not import `agents/`, depend on one agent's prompt format, require
agent credentials, or special-case a model/provider. Task-local evaluation code
must remain usable after materialization without importing Arena's `src/` tree;
framework helpers are materialized through the existing shared mechanism.

Agents receive the configured workspace, editable paths, instructions, target
language/entrypoints, evaluation commands, and read-only baseline material.
They submit changes under the declared boundary. The framework independently
validates and scores the resulting files, then performs configured exports.
Agent logs and search histories are supplementary artifacts.

Scratch repositories, generated drivers, search budgets, and PORT/OPTIMIZE
phases are agent-internal choices. An agent that uses scratch must install all
candidate files back to their configured relative destinations before final
evaluation. It must not require tasks to ship `scripts/forge_driver.py` or
infer builder names from operator identity. A unified Forge integration can
choose a workflow from actual candidate state and language/backend capabilities:
an empty candidate needs generation even when its target language matches the
baseline's; an existing candidate may be optimized or translated. The current
[Forge integration](../../agents/forge/README.md) exposes one agent identity and
adapts its internal workflows to this contract. Its supported conversions and
initialization paths require their own runtime qualification; schema support
alone does not establish an engine capability.

## Optional sanitizers and evaluation tools

After the agent finishes, the framework fixes the candidate version for final
evaluation. Tools run after ordinary candidate compilation/correctness and
before official performance measurement, using isolated tool environments and
the same candidate identity. Their instrumented builds never supply official
performance timings.

Run config decides what is enabled:

```yaml
evaluation_tools:
  enabled: [gpu_asan]
  policy: advisory
  timeout_s: 600
```

Task config supplies an optional dedicated invocation:

```yaml
evaluation_tools:
  tools:
    gpu_asan:
      timeout_s: 300
      options:
        command: [python3, scripts/check_memory.py]
```

The command must actually build/instrument and exercise the candidate and emit
the selected tool's required evidence. Tool commands use the existing
plugin-specific protocol, not the ordinary `ARENA_EVAL_RESULT` envelope. A
successful script exit alone does not prove instrumentation or coverage.

| Concern | Required distinction |
| --- | --- |
| Global memory | Actual memory-access checking requires a supported instrumented artifact/runtime. Loading a library is not evidence that its kernels were checked. |
| Concurrency | Record which kernels, launches, and cases were analyzed. One checked dispatch is not full coverage of a multi-kernel operator. |
| Floating point | Tool-specific semantic checks supplement the task's numerical correctness rule; they do not replace it with a universal tolerance. |

Under `advisory`, findings and incomplete checks remain visible while performance
can continue. Under `required`, every selected applicable tool must be ready,
complete, and clean to permit performance. `not_applicable`, unsupported,
unavailable runtime, missing adapter, and a detected bug remain distinguishable.
A tool's policy does not rewrite ordinary numerical correctness.

A task may register adapters for multiple known tools; only the run-enabled
subset executes. Disabled known adapters remain dormant. Unknown tool names
or malformed options must still be rejected. Task config cannot enable tools,
change run policy/runtime identities, weaken evidence requirements, or increase
the run-level timeout. The common merger validates registered adapters even
when disabled and includes only the run-enabled subset in the execution plan.

Infer the common profile from the candidate declaration. Keep explicit
`evaluation_profile` overrides only where necessary, and verify capabilities
against actual builds/runtime evidence. No schema can create missing FlyDSL
instrumentation support. Consult the maintained
[evaluation-tool guide](use-evaluation-tools.md) for current supported
language/GPU paths, attestation formats, runtime isolation, and limitations.

## More task examples

These examples show v2 authoring patterns, not executable migrations. Entrypoint
names and script paths must be adapted to the real task. Optional blocks appear
only when the task needs them.

### Generate HIP from a PyTorch reference

```yaml
schema_version: 2
description: Implement the supplied PyTorch GELU semantics in HIP.
instructions: [docs/operator.md]
candidate:
  language: hip
  initial_state: unimplemented
  editable: [hip/kernel.hip]
baseline:
  kind: provided
  language: pytorch
  source_files: [pytorch_code_module/reference.py]
evaluation:
  runner: [python3, scripts/evaluate.py]
```

`baseline` actions execute the provided PyTorch implementation; `candidate`
actions execute the HIP implementation. The runner may reuse existing compile,
correctness, and timing scripts behind these role-specific commands.

### Translate an existing Triton implementation to FlyDSL

```yaml
schema_version: 2
description: Replace the Triton implementation with an equivalent FlyDSL implementation.
candidate:
  language: flydsl
  initial_language: triton
  editable: [source/kernel.py]
baseline:
  kind: initial_candidate
evaluation:
  runner: [python3, scripts/evaluate.py]
```

The frozen baseline remains Triton. Initial validation must not require the
unchanged Triton source to be FlyDSL; final candidate validation must enforce
the target language. If the target is instead a separate empty file, use
`unimplemented` plus a `provided` baseline.

### Reimplement a production SIKL GEMM

```yaml
schema_version: 2
description: Implement BF16 C = A @ B.T in FlyDSL for every declared M case.
kernel_identity:
  logical_operator: gemm_a16w16_nt_n32_k6144
  source_owner: aiter
workspace:
  sources:
    - kind: image
      image_path: /sgl-workspace/aiter/aiter
      destination: aiter_source/aiter
      exclude: [jit/build, jit/flydsl_cache, __pycache__]
platform_support:
  required_arch: gfx950
  status: active
candidate:
  language: flydsl
  initial_state: unimplemented
  editable: [kernel.py]
  entrypoints:
    - {file: kernel.py, kind: builder, symbol: build_gemm_a16w16_nt_n32_k6144_module}
baseline:
  kind: provided
  source_files: [aiter_source/aiter/tuned_gemm.py]
evaluation:
  runner: [python3, scripts/evaluate.py]
  workloads: workload.json
```

The copied AITER package provides source context. Baseline actions use the
installed package in the pinned runtime; they do not import this copy. Preserve
the package's Python JIT sources while excluding generated build/cache trees.
Copying the whole upstream repository is unnecessary for this task contract.

The existing SIKL suite contains 17 BF16 GEMM tasks and four MXFP4 MoE tasks,
each covering its declared case set. The same schema fits both families. A MoE
runner keeps its routing, quantization, reference, comparison, and multi-kernel
timing logic in task files. It does not acquire a separate task config schema.

If the production baseline has a documented numerical mismatch, explicitly add
`correctness_policy: diagnostic` and `diagnostic_reason` under `baseline` after
reviewing that evidence. This example does not silently enable that exception.
An optional common export declaration can be added to either family:

```yaml
exports:
  - format: sikl-solution
    output: artifacts/solution.json
    command: [python3, scripts/export_solution.py]
```

## How to add or modify a task

1. **Read the contract.** Read the status and migration sections. Use the shared
   v2 loader, evaluator, and validator, with a task-owned runner implementing
   every required action. A config declaration alone is not an implementation.
2. **Define the problem.** Describe semantics, input/output interface, allowed
   dependencies, target language, editable boundary, and whether a candidate
   already exists. Inspect a nearby task's implementation, not just its config.
3. **Prepare the task files.** Use one `config.yaml`; keep reference, input,
   comparison, and harness logic in protected task-local scripts. An existing
   layout is acceptable. Bundle data or declare pinned source/setup steps.
4. **Declare baseline and cases.** Identify what baseline actions actually
   execute. Establish stable case IDs, representative inputs, and a reproducible
   manifest. Keep numerical reference and performance baseline roles explicit.
5. **Implement the commands.** Use a common runner or explicit action overrides.
   Existing harnesses can be wrapped. Verify failures propagate and outputs
   satisfy the declared protocol. Do not replace real checks with text searches.
6. **Verify evaluation quality.** Exercise a known-correct candidate and a
   deliberately incorrect implementation in a disposable workspace. Verify the
   actual timed path, state reset, complete case coverage, and output parsing.
   For initial stubs, keep candidate feasibility evidence separately rather than
   installing a solution into the committed generation target.
7. **Add optional tooling/export.** Only declare commands that exist and support
   their documented scope. Check them on the intended GPU/runtime. Do not turn
   unsupported tools into clean results.
8. **Run task validation.** Every new task or material contract/harness change
   requires a fresh framework-finalized validator report on compatible GPU
   hardware before submitting a PR. A CPU-only check is not GPU qualification.
9. **Review the change.** Include relevant docs and focused regression coverage
   for behavioral changes. Preserve user-owned workspaces/logs, and do not commit
   generated artifacts or cloned runtime dependencies.

Example validator run config (replace the selector and choose matching hardware):

```yaml
agent:
  template: task_validator
tasks:
  - <task-path-relative-to-tasks>
target_gpu_model: MI355X
log_directory: logs
workspace_directory_prefix: workspace
```

```bash
make docker-run CONFIG=<validator-config>
```

Require `validation_report.yaml` finalized by the framework with
`overall_status: PASS`. WARN requires an explicit maintainer-approved
justification and is not a clean pass. FAIL, timeout, partial/stale reports, and
architecture skips do not satisfy the gate. See the
[validator guide](task-validator.md) for current execution and report details.
Documentation-only changes that do not alter task code/configuration are checked
as documentation; they do not qualify any task on GPU.

## Migration

### Current runtime versus the selected contract

Orchestration selects the v2 path by `schema_version: 2`. That path uses
`TaskSpec`, bounded materialization, `TaskSession`, `ARENA_EVAL_RESULT`, the
shared evaluator and captured-evidence validator. All retained repository tasks
have migrated. Compatibility paths for old workspaces and external legacy
configs still understand `compile_command`/`correctness_command` and their
existing timing formats; they are not a second schema for new tasks.
Integration tests do not qualify Forge/GEAK workflows, SIKL GPU numerical
behavior, or optional GPU tools; those require their own runtime evidence.

When converting an external legacy task, inspect its working commands and
implementation. Legacy command fields are lists of shell command **strings**,
unlike v2 lists of argv lists. Do not mechanically split a shell string on
whitespace; wrap existing shell behavior or translate it deliberately. The old
validator's `torch2hip` placeholder exception belongs only to legacy reports.
V2 uses the same initial-state lifecycle for every task family.

### Mapping existing tasks into v2

| Existing field/behavior | V2 destination or migration action |
| --- | --- |
| `task_type`, `repository_language` | Explicit candidate language, initial state/language, and workspace source declarations. Existing directory names/selectors may remain organizational labels. |
| `source_file_path`, `target_file_path`, `editable_sources` | Explicit `candidate.editable` and, where applicable, `candidate.entrypoints`. Inspect roles: a PyTorch source file can be a reference while the HIP target file is the actual candidate. |
| `target_kernel_functions` | Explicit entrypoint symbols and/or symbol-scoped edit boundaries. Preserve existing harness protection. |
| `rewrite_source_file` | Read-only `baseline.source_files`; the baseline runner defines actual execution. |
| `kernel_identity` | Retain operator/owner metadata; remove implicit builder-name derivation. |
| `compile_command`, `correctness_command`, `performance_command` | A task-local CLI wrapper or role/action overrides with argv lists and v2 results. Preserve the original checks and timing policy. |
| `compile_timeout`, `correctness_timeout`, `performance_timeout` | Action-specific `timeout_s` overrides; preserve intentional budgets. |
| `repo_url`, `repo_subdir`, `image_repo_path`, `image_repo_exclude` | `workspace.sources`, with pinned revisions/image identity and workspace-relative destination-prefixed paths. |
| `post_clone_install`, `post_clone_install_mode` | Explicit repeatable `workspace.setup`; preserve needed setup during fresh/resumed materialization. |
| `prompt.instructions`, `prompt.cheatsheet`, `prompt.source_code` | Agent-independent description/instruction files and declared readable sources. Inspect custom source overrides; do not drop task semantics. |
| `task_result_template` | No v2 task-authored result template. The centralized result schema remains authoritative. |
| Forge-only driver/export behavior | Agent-internal CLI adaptation; task-neutral export commands in framework post-processing. |
| `evaluation_profile`, `evaluation_tools` | Reuse tool contracts; infer candidate metadata from v2 and support dormant known task adapters. |
| `platform_support` | Retain its architecture/skip meaning; a skipped task is not a validator pass. |

### Runtime migration acceptance

Before declaring the migration complete, verify the following together:

- One shared schema/path normalization used by discovery, workspace setup,
  prompting, edit-boundary enforcement, evaluator, validator, and agents.
  Reject ambiguous mixed v1/v2 configs and unsupported schema versions.
- The seven command actions, phase context, result parser, timeout semantics,
  manifest pairing, baseline snapshots, and non-fallback candidate execution.
- Deterministic initial-state/diagnostic policies and report normalization,
  including target-symbol handling. Do not rely on agent prompts for exceptions.
- Common installation/export paths that retain relative directories and do not
  impose Forge-specific filenames or naming conventions.
- Tool profile mapping and disabled-adapter selection with existing runtime,
  evidence, and policy protections intact.
- Representative coverage of an existing HIP task, PyTorch-to-HIP generation,
  Triton-to-FlyDSL conversion, a symbol-scoped task, image tasks,
  and both SIKL operator families. Compare before/after cases, rules, timing,
  and scores on compatible hardware; test with more than one agent integration.

Keep one authored schema for all tasks. Compatibility with historical reports
does not justify new legacy configs or an SIKL-specific schema. Do not silently
change task semantics or benchmark thresholds as part of field migration.
