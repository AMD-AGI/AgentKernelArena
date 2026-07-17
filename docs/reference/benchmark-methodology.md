---
myst:
    html_meta:
        "description": "Learn how AgentKernelArena measures baseline and optimized kernel performance, records timing methods, and computes speedup signals."
        "keywords": "AgentKernelArena, benchmark, CUDA graph, CUDA event, HIP, Triton, kernel timing, warmup, speedup, performance measurement"
---

# Performance measurement methodology in AgentKernelArena

Performance timing is defined by each task's `performance_command`. The
framework runs that command on the original implementation and again on the
agent-modified implementation, then parses structured or textual timing output.
Task suites can use different warmup counts, sample counts, and timing APIs.

This page documents two common timing families. The bundled vLLM and ROCmBench
suites use shared helpers materialized from `src/tools/perf/`; gpumode uses
task-local `cal_kernel_perf.py` copies. FlyDSL and repository-level tasks use
additional task-specific harnesses; inspect their `config.yaml`, runner, and
recorded timing metadata before comparing results across suites.

The shared Triton timing helpers are maintained in `src/tools/perf/`; committed
task sources contain stubs and markers, and `setup_workspace()` materializes the
canonical helpers into each run workspace. See the
[`src/tools/perf/README.md`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/src/tools/perf/README.md)
helper workflow for maintenance details.

## CUDA-graph timing (Triton tasks)

Used by:

- `triton2triton/vllm/*/scripts/task_runner.py` — `_benchmark_cuda_graph_or_events()`
- `*/rocmbench/*/performance_utils_pytest.py` (`instruction2triton` and
  `triton2triton`) — `_measure_times()`

Method:

- 10 warmup iterations.
- A CUDA-graph is captured that replays the kernel `n_repeat` times, where `n_repeat`
  is chosen adaptively so one graph replay takes about `target_ms` (≈1 ms), capped at
  `max_graph_repeats`. This amortizes/eliminates per-launch host overhead.
- The graph replay is timed `repetition` (100) times; each sample is
  `elapsed_time / n_repeat` (per-call time). The reported runtime is the average of
  the 100 samples. Metadata records `benchmark_method: cuda_graph` and
  `benchmark_samples`.
- If CUDA-graph capture fails, it falls back to per-call CUDA-event timing
  (`benchmark_method: cuda_event_fallback`).

## CUDA-event timing (HIP tasks)

Used by:

- `hip2hip/gpumode/*/eval_tools/cal_kernel_perf.py` — `cal_hip_latency()`

Method:

- 10 warmup iterations, then 100 measured iterations timed with a single
  start/stop CUDA-event pair; reported runtime is `elapsed_time / 100`.
- `hip2hip/others/ball_query` uses its own `scripts/task_runner.py` event timing
  (correctness was strengthened separately; timing is unchanged).

## Cross-suite comparison notes

The following notes apply when comparing results across timing implementations.

- The shared CUDA-graph and gpumode paths use 10 warmups and 100 averaged
  measurements. Other task-specific harnesses may use different methodology;
  the validator can report a warning when a task does not follow or document the
  expected method.
- `src/evaluator.py` matches baseline and optimized test cases and prefers the
  average of their per-case speedup ratios. It falls back to aggregate
  `base_execution_time / best_optimized_execution_time` only when an explicit
  ratio is unavailable.
- `task_result.yaml` records baseline and optimized timing-method metadata and a
  `benchmark_method_consistent` flag. Treat a mixed-method speedup as suspect.
- Even when method names match, cross-suite comparisons require checking shapes,
  warmups, sample counts, synchronization, and aggregation rules.
