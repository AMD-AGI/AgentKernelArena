---
myst:
    html_meta:
        "description": "Learn how to use sikl_task_builder to convert a SIKL dataset folder into validated, self-contained Arena tasks ready for optimization runs."
        "keywords": "SIKL, task builder, AgentKernelArena, task creation, schema v2, Codex, task validator"
---

# Build AgentKernelArena tasks from a SIKL bundle

`sikl_task_builder` converts an extracted SIKL dataset folder into validated,
self-contained Arena tasks ready for optimization runs. It joins definitions to
workloads, runs a Codex generation agent to produce each task, and independently
validates every result with `task_validator`. Only tasks that pass all checks
reach the output directory.

Use this workflow to convert any SIKL bundle using the supported schema — no
custom adapter per bundle is required.

**When this workflow is complete:** all accepted tasks appear in
`tasks/SIKL-task/` (or your configured output directory), are listed as
`installed` in `summary.json`, and can be selected immediately by ordinary
`make docker-run` optimization runs.

## Prerequisites

- An extracted SIKL bundle directory (not a tar archive). See
  [Bundle format and schema](#bundle-format-and-schema) for the required layout
  and supported schema versions.
- A GPU host running the AgentKernelArena Docker runtime. The default
  configuration targets MI355X (`gfx950`).
- Codex CLI installed and authenticated on the host — the builder uses Codex as
  the generation agent. The validator supports Codex or Claude Code. See
  [Install agent CLIs](../install/install.md#install-agent-clis).
- All source dependencies referenced by the bundle must exist in the selected
  runtime (including AITER for example bundles). The builder does not install
  missing dependencies.

## Step 1: Inspect the bundle

Before running the builder, verify the bundle is well-formed and preview how
many tasks will be generated. This step does not execute solution code, import
PyTorch, create tensors, or use a GPU:

```bash
python3 -m agents.sikl_task_builder inspect --input-dir '/path/to/extracted bundle'
```

The output reports definitions, task IDs, case counts, and provenance. A
malformed bundle fails here before any agent runs — fix reported errors before
continuing.

## Step 2: Run the builder

Run the builder on compatible ROCm hardware using the Docker runtime:

```bash
make docker-sikl-task-builder INPUT='/path/to/extracted bundle'
```

The builder runs the full pipeline: joining source records, generating task
drafts through Codex, running independent GPU checks for every case, and
validating each task. Depending on the number of definitions and repair
iterations, this might take significant time. Each task gets up to three repair
attempts by default before being marked `failed`.

## Step 3: Review the results

After the run completes, check `sikl_task_builder_runs/<run-id>/summary.json`:

| Status | Meaning |
| --- | --- |
| `installed` | Task passed all checks and was written to the output directory |
| `failed` | Task could not be repaired within the repair budget |
| `needs_spec` | Unresolved source-check failure — requires investigation |
| `platform_deferred` | Declared hardware does not match the selected runtime |

A partial campaign exits with a nonzero code. Each task's generator logs and
validator workspaces are under `sikl_task_builder_runs/<run-id>/` for review.

This workflow does not commit tasks or create pull requests. Review generated
tasks and retain their finalized GPU validation evidence when contributing them,
as required by the [task authoring guide](add-task.md).

## Step 4: Customize the configuration (optional)

To change hardware, numerical policy, generator or validator settings, task
selection, or repair budgets, copy the default configuration and pass it to the
builder:

```bash
make docker-sikl-task-builder INPUT='/path/to/extracted bundle' CONFIG=path/to/config.yaml
```

The default is
[`agents/sikl_task_builder/agent_config.yaml`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/agents/sikl_task_builder/agent_config.yaml).
Key fields:

- `output_dir`: must be below `tasks/`. Generated task IDs follow this
  directory. Do not set it to an existing curated task directory.
- `artifact_root`: must be inside the repository and outside `tasks/`.
- `target_language`: `triton` (default) or `flydsl`.

Relative paths resolve from the repository root. Input, output, and artifact
trees must not overlap.

If a definition has multiple candidate baselines or references, add an explicit
selection to avoid an error:

```yaml
selections:
  example_definition:
    baseline: selected_baseline_solution_name
```

## Step 5: Resume an interrupted run (optional)

If the builder is interrupted, resume the existing campaign rather than
starting from scratch:

```bash
make docker-sikl-task-builder INPUT='/path/to/extracted bundle' \
  SIKL_BUILDER_ARGS='resume --run-id <run-id>'
```

Resume verifies source, configuration, builder, and runtime identity against
the saved snapshots, skips completed terminal tasks, and continues interrupted
work with the original attempt count and per-task deadline. Time spent paused
counts against that deadline.

Start a new campaign (run the builder without `SIKL_BUILDER_ARGS`) after
changing the input bundle, policy, implementation, or environment, or to retry
a terminal failure.

---

## Bundle format and schema

Pass an extracted directory, not a tar file:

```text
bundle/
├── definitions/**/*.json
├── workloads/**/*.jsonl
└── solutions/
    ├── baseline/**/*.json
    └── reference/**/*.json     # legacy schema only
```

One JSONL for one definition produces one task, with every workload row
retained as a benchmark case. For example, 21 JSONLs of 13 rows each produce
21 tasks covering 273 cases. `solutions/kernel-forges` submissions are not
performance baselines and are ignored.

**Input schema v2** — `definition.reference` contains Python source exporting
`run`. Optional `definition.initialize` and `definition.compare` provide
separate Python modules:

- `initialize(inputs, seed=...)` fills preallocated buffers in place and
  returns the same dictionary. Seeds are deterministic per UUID.
- `compare(actual, expected)` returns `None` on success or raises
  `AssertionError`. It owns the numerical rule; with a comparison callback,
  `rtol`/`atol` defaults do not apply.

An absent version or explicit version 1 uses the legacy reference solution and
input policy. Unknown versions and ambiguous legacy/embedded references fail.

Supported entry points include Python, Triton, and FlyDSL Python-hosted
return-value entrypoints. The builder cannot reconstruct original tensor values
from `{"type": "random"}` descriptors — inputs use an explicit versioned
synthesis policy when no initializer is supplied.

Failure modes that are explicitly rejected: duplicate UUIDs or keys, mixed
definitions within one JSONL, multiple JSONLs for the same definition, path
traversal, symlinks, missing references, opaque dimension expressions,
destination-passing interfaces, and absolute imports of local solution modules.

## Output directory structure

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
    ├── initialize/             # supplied callback, when present
    ├── compare/                # supplied callback, when present
    ├── workload.json           # all original cases and fixed policy
    └── provenance.json
```

Generated tasks do not import the builder, Arena's `src/`, or the original
input folder. During optimization runs, Arena protects `config.yaml` and
`scripts/`; only the configured files under `source/` are editable.

Use `scripts/task_runner.py validate-task` to emit the complete UUID manifest,
or `scripts/task_runner.py <baseline|candidate> <compile|correctness|performance>`
to run individual checks.

## Builder tools reference

Invoke tools directly for diagnostics:

```bash
python3 -m agents.sikl_task_builder.tools \
  --run-dir sikl_task_builder_runs/<run-id> --task-id <definition> describe_task
```

| Tool | Result |
| --- | --- |
| `inspect_bundle` | Definitions, task IDs, case counts, and provenance |
| `describe_task` | Full linked contract and editable boundary |
| `materialize_task` | Deterministic skeleton; refuses to overwrite an existing draft |
| `check_contract` | File, source, case, and syntax diagnostics |
| `check_task --mode source-check` | Baseline vs. original reference; also accepts `compile`, `correctness`, `performance` |
| `validate_task` | Fresh independent checks plus formal validation and evidence ID |
| `read_validation --validation-id <id>` | Evidence from an agent-invoked validation |

The builder's controller runs its own independent validation before installing
any task. It does not accept agent-supplied report paths or aggregated results.
