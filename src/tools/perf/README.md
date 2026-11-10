# Canonical performance-benchmark helpers

This directory is the single source of truth for benchmark code that runs inside
task workspaces. Task sources never carry copies of it: the framework writes the
canonical code into each run workspace, so a workspace stays self-contained and
does not import `src`.

## Files

- `aka_benchmark.py` — graph-first device timing (standard library plus
  PyTorch). Schema-v3 kernel tasks receive it as `_aka_eval/_aka_benchmark.py`
  when `src/task_contract.py::render_workspace` writes the evaluator; the
  evaluator times every performance case with
  `benchmark_cuda_graph_cold_cache_samples` and `ReadOnlyCacheFlush`, and every
  case of a multi-GPU task with `benchmark_distributed_cold_cache_samples` on
  each rank (synchronized release, maximum over ranks). The caller passes the
  ranks' communication object, so the module needs no `torch.distributed` setup
  of its own; `tests/test_multi_rank_timing.py` covers the aggregation.
- `serving.py`, `serving_bundle.py`, `serving_backends.py`,
  `serving_scenarios.py`, `serving_request_plan.py` — the end-to-end serving
  client, compute-bundle installation and evidence, and scenario composition.
  `src/perf_helper_materialization.py` copies them (as `_aka_serving*.py`, with
  `src/measurement.py` as `_aka_measurement.py`) beside a serving task's
  `scripts/evaluate.py`. Serving tasks declare a separate wall-clock
  measurement contract; it does not change device-timing rules for kernel tasks.
- `catalog_serving.py`, `catalog_checks.py`, `catalog_instructions.md` — the
  runner, checks and instructions of prepared serving recipe packages
  (`src/magpie_catalog.py`, `src/serving_task_preparation.py`).

## Workflow

1. Edit canonical helper code here.
2. Run `make check-perf-helpers`. It audits that every committed schema-v2 task
   is a serving recipe or serving client and that no generated helper copy is
   committed under `tasks/`.
3. Run the relevant tests (`tests/test_aka_benchmark.py`, the serving tests).

Do not commit `_aka_benchmark.py`, `_aka_serving*.py` or `_aka_eval/` copies
under `tasks/`, and do not hand-edit generated copies in workspaces.

See `docs/reference/benchmark-methodology.md` for measurement and fairness rules.

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
A caller must check the last measured value before re-invoking, then check the
returned replay value with the task's reference and numerical policy.

For evolving state, set `timed_run.before_sample` to a callback accepting the
number of logical calls in that sample's replay. The helper invokes it on the
measurement stream, after `prepare_fn` (when present) and before the start event
of each reported sample. It is not called during warmup, capture, estimation,
graph priming, or `rerun()`. The callback must only observe state; it must not
reset inputs or run the reference. It does not change graph batching.
Apply the same observation and state policy to baseline and candidate. An
observer exception rejects the measurement and leaves the collector unbound.

To check the results of the reported samples themselves, set
`timed_run.after_sample` to a callback accepting the sample's outputs (the
captured outputs under CUDA-graph timing, the return value under Event timing).
The helper invokes it after each reported sample's end event has completed,
outside the measured interval, and never during warmup, capture, estimation,
graph priming, `rerun()` or `rerun_ms()`. It must only read the outputs. The
check happens after the sample has run, so nothing the implementation observes
while running distinguishes a checked sample from an unchecked one -- unlike a
separate post-timing invocation whose inputs or outputs were prepared for it.

`timed_run.rerun_ms()` executes the timed unit once more through the reported
sample path (preparation, start event, invocation, end event) and returns its
device time, under both timing methods.
