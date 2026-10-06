# Canonical performance-benchmark helpers

This directory is the single source of truth for graph-first device timing.
Committed task sources keep small imports, stubs, or generated marker regions;
`setup_workspace()` materializes the canonical code into each run workspace.
The resulting task remains self-contained and does not import `src`.

## Files

- `aka_benchmark.py` — self-contained Python implementation (standard library
  plus PyTorch). It is copied as `_aka_benchmark.py` beside importing
  performance entrypoints. Its optional `prepare_fn` supports stateful kernels
  by restoring inputs before, and outside, each measured replay/event interval.
  It also contains the conservative HIP-source current-stream preflight used by
  compiled-extension tasks.
- `performance_utils_pytest.py` — thin ROCmBench adapter. Task sources keep a
  stub; run workspaces receive this file and a sibling `_aka_benchmark.py`.
- `vllm_cuda_graph_block.py` — thin compatibility functions injected between
  the vLLM/image AKA-GENERATED markers. A sibling `_aka_benchmark.py` supplies
  their implementation.
- `native_hip_graph_benchmark.hpp` — self-contained HIP runtime implementation,
  materialized as `scripts/native/hip_graph_benchmark.hpp` when requested by a
  native benchmark driver.

## Workflow

1. Edit canonical helper code here.
2. Run `make check-perf-helpers`. The check validates stubs and audits every
   configured task performance entrypoint.
3. If marker or stub structure changes, run `make sync-perf-helpers`.

## Task-local protocols

Some tasks ship protected timing implementations instead of requesting generated
helpers. `custom_protocols.json` registers the reviewed entrypoint hashes and
their timing-helper hashes. The audit requires the exact performance command
and regular files within the task; a task name or a matching filename alone does
not qualify. Portable case protocols additionally require byte parity with
`src/task_contract.py` and a valid `cases.json`. The registry covers direct
`checked_replays` callers, the `FreshCallbacks` adapter, legacy head-kernel
event/GEAK timing, and isolated native graph workers. A known entrypoint that
always emits a blocked report and exits nonzero is counted separately as
`blocked_entrypoint`; recognizing it does not make the task scoreable.

Review changes to these task-local runners and timing helpers before updating
their registry hashes, and run their focused contract tests. The sync command
never refreshes those hashes or rewrites those runners. Recognition is a static
implementation check, not GPU correctness, runtime source binding, or task
qualification. Existing legacy protocols keep their original methodology and
validation status. The task-validator and measurement gates still apply.

Do not commit `_aka_benchmark.py` or `hip_graph_benchmark.hpp` copies under
`tasks/`, and do not hand-edit generated helper regions. Use
`make materialize-perf-workspace WORKSPACE=...` for an existing copied
workspace, or `make materialize-perf-task TASK=tasks/...` for local inspection.

See `docs/reference/benchmark-methodology.md` for measurement and fairness
rules, including paired graph/Event baseline selection.
