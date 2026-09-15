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

Do not commit `_aka_benchmark.py` or `hip_graph_benchmark.hpp` copies under
`tasks/`, and do not hand-edit generated helper regions. Use
`make materialize-perf-workspace WORKSPACE=...` for an existing copied
workspace, or `make materialize-perf-task TASK=tasks/...` for local inspection.

See `docs/reference/benchmark-methodology.md` for measurement and fairness
rules, including paired graph/Event baseline selection.

## Observing measured outputs

Pass `TimedRun()` as `timed_run` to retain the actual measured output and a
callable for subsequent numerical validation. Graph timing binds the captured
output buffers and replays the measured graph. Explicit Event timing
(`use_cuda_graph=False`, including the paired forced-Event setting) retains the
return value from the last measured sample; `rerun()` calls the same eager
callable again and can return new buffers. Metadata distinguishes these as
`benchmark_timed_run_kind: captured_graph` or `eager_callable`. Eager re-invocation
is not captured graph replay and cannot freeze Python dispatch decisions.

Preparation runs before the start Event, outside the measured interval, and
before each eager re-invocation. The collector never makes a post-timing call
to obtain its initial output. A failed benchmark clears any previous binding.
Automatic graph-to-Event fallback with a collector still fails closed; Event
collection must be explicitly selected. Both methods require actual GPU timing.
Task harnesses must check the last measured value before re-invoking, then check
the returned replay value with the task's reference and numerical policy.
