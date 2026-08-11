---
myst:
    html_meta:
        "description": "How AgentKernelArena measures GPU kernels with CUDA or HIP Graph replay, records event fallbacks, and compares matched benchmark cases."
        "keywords": "AgentKernelArena, benchmark, CUDA Graph, HIP Graph, CUDA event, HIP event, Triton, FlyDSL, kernel timing, speedup"
---

# Performance measurement methodology in AgentKernelArena

AgentKernelArena uses graph-first device timing for Python-callable GPU kernels,
regardless of whether the kernel is implemented in Triton, HIP, FlyDSL, or
PyTorch.  The canonical implementation is
`src/tools/perf/aka_benchmark.py`.  PyTorch exposes CUDA and ROCm through the
same `torch.cuda` namespace, so `benchmark_method: cuda_graph` means CUDA Graph
on a CUDA build and HIP Graph on a ROCm build.

Native HIP benchmark drivers use the equivalent implementation in
`src/tools/perf/native_hip_graph_benchmark.hpp`.

## Canonical Python API

The common mean-time interface is:

```python
mean_ms, metadata = benchmark_cuda_graph_or_events(
    fn,
    prepare_fn=None,
    warmup=10,
    repetition=100,
    target_ms=1.0,
    estimate_reps=5,
    max_graph_repeats=1000,
)
```

`mean_ms` is device milliseconds per invocation of `fn`.  Suites whose scoring
contract uses a median or percentile call
`benchmark_cuda_graph_or_events_samples()` instead.  It returns the same
metadata and a list of per-invocation millisecond samples.

The callable must launch already-compiled device work. JIT compilation, module
construction, and scratch/workspace sizing belong before the benchmark call.
If the public operation returns newly allocated outputs, both sides must retain
that same contract; otherwise both sides use caller-owned output buffers.

Stateful in-place kernels may provide `prepare_fn`. It runs on the active
benchmark stream before each warmup and measured sample, but before the start
event, so input restoration is ordered correctly and excluded from the kernel
time. In this mode each graph replay contains one logical `fn` call; this
prevents a later captured call from consuming state mutated by an earlier one.
Output resets follow the same rule: they belong in `prepare_fn`, not in the
captured/timed callable.

## Graph-first measurement

For each fixed test case, the helper:

1. Refuses to score when `torch.cuda.is_available()` is false. CPU wall-clock
   time is never substituted for kernel device time.
2. Runs the requested warmup calls and synchronizes them.
3. Captures and primes a small estimate graph, then measures one replay with
   GPU events.
4. Rejects PyTorch's explicit empty-graph warning and independently rejects an
   effectively empty replay rather than reporting a fabricated near-zero time.
5. Selects a bounded number of calls per graph so one replay targets roughly
   `target_ms`. This amortizes host launch overhead for short kernels.
6. Captures and primes the final graph outside the reported sample set, replays
   it `repetition` times, and divides each replay's event time by the calls
   captured in that graph.
7. For tasks that request a timed-run handle, exposes the exact captured output
   buffers and an additional replay of the same graph executable. Correctness
   checks poison or perturb those buffers, replay, and compare them with the
   eager/reference result. Such tasks fail closed rather than falling back to an
   unobservable Event invocation.

The start event, graph replay, and end event all run on the same side stream.
This ordering is important: recording events on one stream while replaying on
another can appear to time an empty stream.

Successful samples record at least:

```yaml
benchmark_method: cuda_graph
benchmark_samples: 100
benchmark_effective_repeats: 80
benchmark_target_ms: 1.0
benchmark_max_repeats: 1000
benchmark_warmup: 10
```

## GPU-event fallback

Some callables cannot be captured, for example because they synchronize the
device, read a device scalar on the host, allocate capture-unsafe memory, or
launch work on an incompatible stream.  Capture failure, empty capture, and
invalid replay timing fall back explicitly to eager GPU-event measurement:

```yaml
benchmark_method: cuda_event_fallback
benchmark_fallback_reason: "cuda_graph_failed: RuntimeError: ..."
benchmark_effective_repeats: 1
```

Tasks may disable graph capture for a known incompatibility by passing
`use_cuda_graph=False` with a stable `fallback_reason`.  A result is never
silently changed to event timing merely because graph timing is slower.

Before accepting fallback, task harnesses should hoist host-side shape work,
pointer-table creation, temporary allocation, and JIT setup out of `fn`, then
directly launch the underlying kernel where that is part of the task's stable
API. A legacy HIP extension that hard-codes stream `0` is a legitimate example
of an incompatible launch: CUDA/HIP Graph capture begins on a non-default side
stream, while capturable PyTorch allocation/fill nodes can otherwise make a
graph look non-empty even when the target kernel escaped to stream zero. HIP
task harnesses therefore run a conservative source preflight. Every visible
launch must use a stream obtained from PyTorch's current CUDA/HIP stream; a
literal legacy stream, capture-unsafe synchronization/allocation, or an
unverifiable launch construct classifies the baseline case as Event-only. For
hip2hip comparisons, the reference source fixes this policy; candidate source
cannot downgrade a graph-capable reference to Events.

Fallback still uses device events; it does not use Python, subprocess, or CPU
wall-clock timing.  Host and wall-time-only fields are rejected by the central
performance parser.

## Independent task workspaces

Task sources do not import AgentKernelArena's `src` package and do not carry
copies of the full canonical helper. During `setup_workspace()`:

- `_aka_benchmark.py` is copied beside each importing performance entrypoint,
  including root `test_kernel_harness.py`, `scripts/task_runner.py`, and
  `eval_tools/cal_kernel_perf.py`;
- ROCmBench's `performance_utils_pytest.py` stub is replaced with a thin adapter;
- vLLM's marked inline block is replaced with a thin adapter and receives a
  sibling `_aka_benchmark.py`;
- a native driver that includes `hip_graph_benchmark.hpp` receives the canonical
  header as `scripts/native/hip_graph_benchmark.hpp`.

The materialized workspace is self-contained. The Python timing helper depends
only on the standard library and PyTorch; the native helper depends only on the
HIP runtime already required by its task.

Generated helpers, marked adapters, native benchmark drivers, and the source
files directly named by `performance_command` are covered by the harness
integrity guard. If a ROCmBench task intentionally colocates its editable kernel
and benchmark in one Python file, the guard masks only the declared target
function bodies; the surrounding imports, decorators, signatures, helpers, and
tests remain protected.

## Baseline and optimized fairness

The evaluator matches baseline and optimized cases by unique explicit ID first,
then by unique parameters or shape. Benchmark methods are then compared **per
matched case**:

- graph versus graph is comparable;
- event fallback versus event fallback is comparable;
- graph versus event fallback is retained as timing data but produces no
  speedup;
- different shapes may use different methods, provided each matched
  baseline/optimized pair uses the same exact method.

The baseline result fixes the timing policy before candidate measurement. If a
baseline case succeeds with Graph but the candidate falls back to Events, the
candidate timing remains visible for diagnosis but earns no performance score.
Only a case that the baseline itself classified as Event-only can be scored as
Event versus Event. The evaluator never collects or selects a candidate-driven
alternate Event baseline.

Missing or unknown method metadata is not comparable and produces no speedup;
every scored case must explicitly identify `cuda_graph` or
`cuda_event_fallback`.

An aggregate `mixed:...` method string is never scored, even when the baseline
and optimized strings are identical. It records only a set of methods and loses
the shape-to-method mapping, so equality cannot prove that the same shapes used
the same timer. The timing is retained with an
`ambiguous_mixed_aggregate` mismatch reason.

This means a task can legitimately use graph timing for one shape and event
fallback for another. The evaluator records method sets, per-case mismatch
details, and `benchmark_method_consistent` in `task_result.yaml`. All
`benchmark_*` metadata survives the baseline/optimized YAML round trip.

## Maintenance checks

Run:

```shell
make check-perf-helpers
```

The check verifies committed ROCmBench stubs and marked adapters, then audits
every task config. Each configured performance path must be recognized as a
canonical Python importer, vLLM adapter, ROCmBench adapter, or native graph
driver. An unrecognized benchmark entrypoint fails the check instead of quietly
using an undocumented timer.

Pull requests also run the CPU/mock unit suite and Python compilation audit in
CI. `benchmark-gpu-smoke` exercises a stateful Graph workload, observable timed
replay, forced-Event rejection, and native matrix/MLA replay validation with
nonzero inputs. It retains the logs as artifacts. Repository administrators can
set `AKA_GPU_CI_ENABLED=true` after registering an isolated self-hosted runner
with the `rocm` and `gfx950` labels, then make this workflow a required check.
