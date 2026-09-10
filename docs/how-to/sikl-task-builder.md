# Build Arena tasks from a SIKL bundle

`sikl_task_builder` turns a local SIKL dataset folder into standard, self-contained
Arena tasks under `tasks/SIKL-task/`. It combines deterministic import tools, a
Codex agent that repairs input generation, and an independent `task_validator`.
Only accepted tasks reach the output directory. Future bundles using the supported
schema can use the same workflow without a new handwritten task adapter per JSONL.

This is an agent workflow **before task discovery**, invoked with its own Make
command. It is not an `agent.template` for `main.py`: its input is a dataset
folder and its output is multiple tasks. The resulting tasks use the existing
`instruction2triton` task type and can be selected by ordinary optimization runs.
It does not require KernelForge or a task-specific SIKL integration.

## Input and task boundaries

Pass an extracted directory, not a tar file:

```text
bundle/
├── definitions/**/*.json
├── workloads/**/*.jsonl
└── solutions/
    ├── baseline/**/*.json
    └── reference/**/*.json
```

A JSONL contains **cases**, not kernel source. The builder joins its `definition`
name to a definition and one baseline/reference solution. A definition describes
axes, shapes, dtypes and outputs; a solution contains its entry point and source
files. One JSONL for one definition becomes one task, with every row retained.
For example, 21 JSONLs of 13 rows each become 21 tasks covering 273 cases, rather
than 273 independent kernels. Empty `solutions/kernel-forges` slots are ignored.

The initial schema supports Python return-value solutions, positive integer
constant/variable dimensions, shape entries that are dimensions or axis names,
scalar literals, and `random` inputs. Dimension constraints support comparisons
of axis names and integer literals. Package-relative multi-file Python imports
are preserved. Ambiguous solutions require explicit selection:

```yaml
selections:
  example_definition:
    baseline: selected_baseline_solution_name
    reference: selected_reference_solution_name
```

Duplicate UUIDs/keys, mixed definitions within one JSONL, multiple JSONLs for the
same definition, path traversal, symlinks, missing references, opaque dimension
expressions, unsupported input descriptors, destination-passing interfaces, and
absolute imports of local solution modules fail explicitly. Unknown descriptive
fields are preserved; they do not automatically acquire runtime meaning.

The conversion preserves metadata and source text. **It cannot reconstruct the
original tensors from `{"type": "random"}`.** Inputs use an explicit versioned
synthesis policy. Built-in policies cover floating-point GEMM and declared
per-1x32 MXFP4 MoE: quantized/shuffled weights, block scales, and normalized unique
top-k routing. Other policies need an input adapter supported by the definition
and original code. An unresolved baseline/reference disagreement cannot be
repaired by rewriting the reference or relaxing tolerances.

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
[`agents/sikl_task_builder/agent_config.yaml`](../../agents/sikl_task_builder/agent_config.yaml).
It selects MI355X and the existing Docker image policy. Copy it to change the
hardware, generator/validator settings, numerical policy, task selection or
budgets, then pass `CONFIG=path/to/config.yaml`. Relative paths resolve from the
repository. `output_dir` must be below `tasks/`; `artifact_root` must be inside the
repository and outside `tasks/`. Input/output/artifact trees must not overlap.

The generator currently uses Codex. The validator supports Codex or Claude Code.
A null model uses that backend's configured default. The Docker launcher checks
and mounts the required CLIs and login state. All source dependencies (including
AITER for the example bundle) must exist in the selected runtime; the agent does
not get an automatic dependency-installation or source-rewriting escape hatch.
The first release supports `target_language: triton` only.

## Workflow and tools

1. Inspect and join source records; snapshot the bundle, configuration and runtime
   identity, and compute fingerprints.
2. Emit a deterministic draft outside `tasks/`. Start `source/kernel.py` from the
   selected production baseline. This is the initial benchmark implementation,
   not an optimized Triton solution.
3. Invoke the generation agent. Its only editable draft file is
   `scripts/task_inputs.py`; it can call JSON-returning tools to inspect the
   contract, run checks, and diagnose validator feedback.
4. Check file/source fidelity. In a fresh GPU workspace, independently run
   baseline-vs-reference, compilation, correctness, and performance checks over
   **every** case. Candidate exceptions never fall back to the baseline.
5. Launch the official `task_validator` in that workspace and finalize its report
   with the existing framework. A complete `PASS`, successful independent commands,
   unchanged harness, and matching task digest are all required. `WARN`, timeout,
   partial report, or a model-written `PASS` without finalization is insufficient.
6. Feed failures back for up to three repairs by default. Every repair requires
   fresh validation. Install only the exact accepted draft by atomic directory
   rename; never replace a differing existing task.

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
| `check_task --mode source-check` | Baseline vs original reference; also accepts compile/correctness/performance |
| `validate_task` | Fresh independent checks plus formal validation and evidence ID |
| `read_validation --validation-id <id>` | Evidence corresponding to an agent-invoked validation |

Agent-invoked validation is diagnostic. The controller runs its own validation
before installation and does not accept agent-supplied report paths or aggregates.
Numerical policy, cases, original sources and timing code are fixed during a run.
The runner checks tensor/scalar contracts, finite outputs, input non-mutation, and
the output of the exact CUDA graph replay used for timing, after poisoning its
captured output buffers to detect stale results. Canonical performance
helpers are materialized by the normal workspace machinery.

## Outputs, repair limits and resume

```text
tasks/SIKL-task/<definition>/
├── config.yaml
├── source/kernel.py
├── source/implementation/       # initial baseline source
└── scripts/
    ├── task_runner.py
    ├── task_api.py
    ├── task_inputs.py
    ├── baseline/               # unchanged original code
    ├── reference/              # unchanged original code
    ├── workload.json           # all original cases and fixed policy
    └── provenance.json
```

Generated tasks do not import this builder, Arena's `src/`, or the original input
folder. During later optimization, Arena protects `config.yaml` and `scripts/`;
only the configured files under `source/` are editable.

Run state, source snapshot, generator logs, validator workspaces and content-bound
`evidence.json` files remain under `sikl_task_builder_runs/<run-id>/`. Review
`summary.json` for installed, failed, `needs_spec` (unresolved source-check failure)
or `platform_deferred` (declared hardware mismatch) tasks. A partial campaign
returns a nonzero exit code. A malformed bundle fails at inspection before any
agent runs. Runtime/dependency errors retain their diagnostics.

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
retain their finalized GPU validation evidence when contributing them, as required
by the [task authoring guide](add-task.md).
