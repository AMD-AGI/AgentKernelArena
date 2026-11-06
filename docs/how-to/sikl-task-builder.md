# Build Arena tasks from a SIKL bundle

`sikl_task_builder` turns a local SIKL dataset folder into standard, self-contained
schema-v3 Arena tasks under `tasks/SIKL-task/`. It combines deterministic import
tools, a Codex agent that reviews generation (and repairs input distributions),
and an independent audit by the [QA auditor](qa-auditor.md). Only accepted tasks
reach the output directory. Future bundles using the supported schema can use the
same workflow without a new handwritten task adapter per JSONL.

This is an agent workflow **before task discovery**, invoked with its own Make
command. It is not an `agent.template` for `main.py`: its input is a dataset
folder and its output is multiple tasks. The resulting tasks use the
[schema-v3 task contract](add-task.md): `task.yaml`, `workloads.json`,
`reference.py`, an optional `inputs.py`, `README.md` and a vendored `baseline/`.
They name no candidate language; the optimization run selects it with its
top-level `target_language`. It does not require KernelForge or a task-specific
SIKL integration.

The builder's configurable `output_dir` defaults to the legacy-named
`tasks/SIKL-task/`. Set `output_dir` in the
[builder configuration](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/sikl_task_builder/agent_config.yaml)
to choose another directory below `tasks/`. Generated task IDs follow that
directory, and existing curated task packages must not be overwritten.

## Input and task boundaries

Pass an extracted directory, not a tar file:

```text
bundle/
├── LICENSE                     # needed when the baseline is not Apache-2.0
├── definitions/**/*.json
├── workloads/**/*.jsonl
└── solutions/
    ├── baseline/**/*.json
    └── reference/**/*.json     # legacy schema only
```

A JSONL contains **cases**, not kernel source. The builder joins its `definition`
name to a definition and one baseline solution. A definition describes
axes, shapes, dtypes and outputs; a solution contains its entry point and source
files. One JSONL for one definition becomes one task. Rows with identical axes
and scalars are merged into one case; the rest are kept. For example, 21 JSONLs
of 13 rows each become 21 tasks, rather than 273 independent kernels.
`solutions/kernel-forges` submissions are not performance baselines or numerical
references and are ignored.

The builder derives the workload file deterministically: every distinct row is
a case, a size-ranked selection of those cases are performance cases (at least
the contract's minimum, spanning its minimum size ratio) and the rest are
correctness-only. When the contract's boundary rule (minimum, non-power-of-two
and unaligned cases) is not met, it adds the fewest in-domain correctness-only
boundary controls. A bundle that cannot satisfy the contract fails as
`needs_spec`.

For **input schema v2**, `definition.reference` contains Python source exporting
`run`, and optional `definition.initialize` and `definition.compare` contain
separate Python modules exporting `run` too:

- `reference` becomes `reference.py` unchanged. It must be one self-contained
  file importing only PyTorch and the standard library.
- `initialize(inputs, seed=...)` is preserved unchanged and wrapped by a
  generated `inputs.py`; it owns the input distribution and the generator agent
  cannot edit it.
- `compare` is **not** carried over. Outputs use the task contract's bounds
  (`exact` for non-floating outputs, `baseline_relative` otherwise), and the
  generated README says so.
- Solution `spec.target` lists `{arch, hardware_id}` objects. The declared GPU
  architecture must match the selected runtime. Legacy `target_hardware` lists
  remain supported; mixing both forms is rejected.

An absent version or explicit version 1 uses the legacy reference solution.
Unknown versions and ambiguous legacy/embedded references fail.

The baseline is vendored unchanged as one flat package under `baseline/`,
beside a generated `baseline/kernel.py` that adapts its return-value entry point
to the contract's in-place `run()` interface. Its language (Triton, Gluon or
FlyDSL) is resolved from the solution declaration and its imports. The builder
rejects a baseline that imports a production package (such as `aiter`, `vllm` or
`sglang`), uses a forbidden process, network, dynamic-import or file API, or
records no license. `baseline.provenance` records the bundle's source, a fixed
revision and the SPDX license. A baseline license other than Apache-2.0 needs
the bundle's `LICENSE` text, which becomes `baseline/LICENSE`; when the solution
records carry no license, set `license` in the builder configuration to the
bundle's actual SPDX identifier.

Supported solutions use positive integer constant/variable dimensions, shape
entries that are dimensions or axis names, scalar literals, and `random` inputs.
Dimension constraints support comparisons of axis names and integer literals.
Ambiguous solutions require explicit selection:

```yaml
selections:
  example_definition:
    baseline: selected_baseline_solution_name
    reference: selected_reference_solution_name  # legacy schema only
```

Duplicate UUIDs/keys, mixed definitions within one JSONL, multiple JSONLs for the
same definition, path traversal, symlinks, missing references, opaque dimension
expressions, unsupported input descriptors, destination-passing interfaces,
nested baseline layouts and absolute imports of local solution modules fail
explicitly.

The conversion preserves metadata and source text. **It cannot reconstruct the
original tensors from `{"type": "random"}`.** Without a supplied initializer,
tensor inputs get a default synthesis distribution (normal for floating dtypes,
bounded integers otherwise) that the generator agent may repair in
`inputs.py` and `task.yaml`; review it against the operator's semantics. An
unresolved baseline/reference disagreement cannot be repaired by rewriting the
reference or relaxing the error bound.

## Run

Inspect metadata on the host; this does not execute solution code, import torch,
create tensors, call an agent or use a GPU:

```bash
python3 -m agents.sikl_task_builder inspect --input-dir '/path/to/extracted bundle'
```

Run on compatible ROCm hardware using the standard Docker runtime:

```bash
make docker-sikl-task-builder INPUT='/path/to/extracted bundle'
```

The default configuration is
[`agents/sikl_task_builder/agent_config.yaml`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/sikl_task_builder/agent_config.yaml).
It selects MI355X and the existing Docker image policy. Copy it to change the
hardware, generator/validator settings, task selection, `license` or budgets,
then pass `CONFIG=path/to/config.yaml`. Relative paths resolve from the
repository. `output_dir` must be below `tasks/`; `artifact_root` must be inside the
repository and outside `tasks/`. Input/output/artifact trees must not overlap.

The generator currently uses Codex. The `validator` section selects the QA
auditor's model review: `codex`, `claude_code`, or `none` (deterministic checks
only). A null model uses that backend's configured default. The Docker launcher
checks and mounts the required CLIs and login state. All source dependencies must
exist in the selected runtime; the agent does not get an automatic
dependency-installation or source-rewriting escape hatch.

There is no `target_language` option and no seeds, timing or tolerance setting:
those belong to the framework's task contract and the optimization run's
configuration.

## Workflow and tools

1. Inspect and join source records; snapshot the bundle, configuration and runtime
   identity, and compute fingerprints.
2. Emit a deterministic draft outside `tasks/`, with the vendored baseline,
   `reference.py`, workloads and generated `README.md`.
3. Invoke the generation agent. `describe_task` exposes the authoring boundary:
   `README.md`, correctness-only workload cases, and (when no initializer is
   supplied) `inputs.py` plus the input distributions and generator declaration
   in `task.yaml`. Everything else is the controller's emission, and
   `check_contract` rejects any other change.
4. In a fresh GPU workspace, the shared `TaskSession` runs the initial actions
   over every case, and `audit_task_session` runs the QA auditor on that captured
   evidence. The draft is materialized for the baseline's language, so the
   candidate path gets an exact copy of the baseline as a positive control.
5. Accept the draft only when the framework-finalized `qa_report.yaml` is current
   for this validation request and task, has status `PASS` (no blocking findings;
   warnings do not block), the audit left the task files unchanged, and the task
   digest matches. A timeout, partial report, or model-written `PASS` without
   finalization is insufficient.
6. Feed blocking findings back for up to three repairs by default. Every repair
   requires a fresh audit. Install only the exact accepted draft by atomic
   directory rename; never replace a differing existing task.

The agent can invoke these tools from the provided prompt:

```bash
python3 -m agents.sikl_task_builder.tools \
  --run-dir sikl_task_builder_runs/<run-id> --task-id <definition> describe_task
```

| Tool | Result |
| --- | --- |
| `inspect_bundle` | Definitions, task IDs, case counts and provenance |
| `describe_task` | Full linked contract and editable boundary |
| `materialize_task` | Deterministic skeleton; refuses to overwrite an existing draft |
| `check_contract` | File, source, case and syntax diagnostics |
| `check_task --mode <mode>` | One evaluator action (`validate-task`, `source-check` (default; baseline correctness), `compile`, `correctness`, `performance`) against the task's own case manifest |
| `validate_task` | Fresh initial actions plus QA audit and evidence ID |
| `read_validation --validation-id <id>` | Evidence corresponding to an agent-invoked validation |

Agent-invoked validation is diagnostic. The controller runs its own audit
before installation and does not accept agent-supplied report paths or aggregates.
The reference, baseline, performance selection and error bounds are fixed
during a run.

## Outputs, repair limits and resume

```text
tasks/SIKL-task/<definition>/
├── task.yaml
├── workloads.json
├── reference.py
├── inputs.py              # generator-owned inputs, or the initialize wrapper
├── sikl_initialize.py     # supplied initialize callback, when present
├── README.md
└── baseline/              # vendored original code, kernel.py adapter, LICENSE when required
```

Generated tasks do not import this builder, Arena's `src/`, or the original input
folder. The framework writes the evaluator and timing helper into each run
workspace; no benchmark code is committed with the task.

Run state, source snapshot, generator logs, audit workspaces and content-bound
`evidence.json` files remain under `sikl_task_builder_runs/<run-id>/`. Review
`summary.json` for installed, failed, `needs_spec` (an unresolved source-check
failure or a bundle that cannot meet the task contract) or `platform_deferred`
(declared hardware mismatch) tasks. A partial campaign returns a nonzero exit
code. A malformed bundle fails at inspection before any agent runs.

```bash
make docker-sikl-task-builder INPUT='/path/to/extracted bundle' \
  SIKL_BUILDER_ARGS='resume --run-id <run-id>'
```

Resume checks source/configuration/builder/runtime identity, saved snapshots, and
installed task hashes. It skips completed terminal tasks and resumes interrupted
work with the original attempt count and absolute per-task deadline. Time spent
paused counts against that deadline. Start a new campaign after changing input,
policy, implementation or environment, or to retry a terminal failure.

The Docker wrapper mounts source data, configuration and framework code read-only,
with writable artifact/output subdirectories and isolated agent home state.
This preserves workflow boundaries, not a hostile-code security sandbox; source
code is executed during GPU validation in the repository's privileged runtime.
Use trusted bundles and review any generated input adapter and its evidence.

This workflow does not commit tasks or create PRs. Review generated tasks and
retain their finalized QA audit evidence when contributing them, as required by
the [task authoring guide](add-task.md) and the [QA auditor guide](qa-auditor.md).
