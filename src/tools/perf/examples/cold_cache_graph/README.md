# Cold-cache graph replay: task-authoring reference

New and materially revised kernel benchmarks should use this measurement pattern.
The runnable example uses the original five geometries, Triton entrypoint, FP16
reference and `atol=rtol=0.01` gate from
`tasks/triton2triton/vllm/triton_chunked_prefill_paged_decode`.
The same PyTorch FP32 oracle formula runs on CPU and its FP16 result is copied
back before cache eviction, outside timing. This keeps oracle execution
independent of the GPU BLAS runtime used by a baseline; candidate computation,
cache eviction and Graph/Event measurement still execute on the GPU. The report
records `oracle_device: cpu`. The original task's reference code is unchanged.
It compares the original source with itself by default (A/A); pass a separate
candidate source for A/B. It is an example, not an Arena task or a score writer.

## Measurement boundary

```text
same GPU stream:
  fresh input + pristine oracle + output poison
  -> read-only 1 GiB cache eviction sweep
  -> start GPU Event
  -> replay graph containing ONE complete operator invocation
  -> end GPU Event, wait
  -> validate the actual measured output and read-only inputs
```

`ReadOnlyCacheFlush` allocates and initializes its buffer once outside timing.
The sweep reduces every int32 input into small preallocated scratch; the large
buffer is read, not overwritten. A large `zero_()` sweep would leave dirty cache
lines and can charge their writeback to the following operator. Never use
`torch.cuda.empty_cache()` as a substitute: it manages allocator reservations,
not GPU data-cache residency.

Input preparation precedes eviction, so preparation/reference checks cannot warm
the operands after the sweep. The GPU stream orders eviction before the start
Event. No flush-time subtraction or profiler instrumentation is needed. The
metric is the complete **GPU Event graph span**, including internal kernel gaps;
it is not a profiler sum of kernel durations or end-to-end host latency.

Every replay contains one invocation. Flushing once and then replaying several
operators in one timed interval would mostly measure warm-cache work. There is
no `target_ms` batching and no automatic Event/CPU fallback. `TimedRun.rerun()`
and `rerun_ms()` retain the same preparation/eviction policy and captured graph.

The method is `cuda_graph_cold_cache`. Baseline and candidate must use the same
method, eviction byte count, device, work, input distribution and output/scratch
contract. Never compare it with `cuda_graph` or `cuda_graph_cold_weights` results.
Use the same initialized eviction object for both arms. One GiB is the reference
default; explicitly qualify a larger sweep if the device's relevant caches need
it. An eviction sweep approximates cold **data caches**; it is not hardware cache
invalidation, cold compilation, a fresh process or a guarantee of zero residency.

## Run inside the GPU Docker runtime

From the repository root, in the project's selected GPU runtime container:

```bash
python3 src/tools/perf/examples/cold_cache_graph/paged_attention.py \
  --output "$AKA_EXPERIMENT_DIR/results/cold-cache-aa.json"
```

`AKA_EXPERIMENT_DIR` is an existing user-owned experiment directory mounted into
the container, outside the source snapshot. Use the repository's Docker/Slurm
launch workflow; do not start a host-side scoring run. For a short protocol smoke
(not statistically useful performance evidence):

```bash
python3 src/tools/perf/examples/cold_cache_graph/paged_attention.py \
  --case 1 --rounds 4 --samples 5 --warmup 2 \
  --output "$AKA_EXPERIMENT_DIR/results/cold-cache-smoke.json"
```

To compare a preserved original with a candidate:

```bash
python3 src/tools/perf/examples/cold_cache_graph/paged_attention.py \
  --baseline-source "$AKA_EXPERIMENT_DIR/baseline/triton_chunked_prefill_paged_decode.py" \
  --candidate-source "$AKA_EXPERIMENT_DIR/candidate/triton_chunked_prefill_paged_decode.py" \
  --output "$AKA_EXPERIMENT_DIR/results/cold-cache-ab.json"
```

The default is 16 paired rounds, ten warmups and 100 samples per arm/round.
Balanced AB/BA order is shuffled with a recorded seed. Each sample consumes fresh
query/K/V values; both arms reset to the same generation sequence in each round.
Full pristine CPU reference, unchanged-input and genuine Triton-dispatch checks run
outside timing. Changed-input replay outputs are checked as well.

The JSON retains every sample, round order, source/helper hashes and runtime
versions. Per-round medians produce paired ratios; the report gives their
geometric mean and a 95% paired-round percentile-bootstrap interval, with no
outlier removal/clipping. This interval assumes representative, approximately
independent rounds; inspect raw variation and device/thermal state. It cannot
repair noisy/shared hardware. A/A is a sanity check, not a speedup claim. A short
smoke or a confidence interval crossing 1 does not establish an optimization win.
A confidence interval excluding 1 is also insufficient by itself: A/A can show
tiny apparent differences, and testing many cases increases false discoveries.
Predeclare a practically meaningful gain and a multiple-comparison policy rather
than selecting a favorable case, retry or interval after seeing the data.

## Reuse in a self-contained task

Arena already materializes `aka_benchmark.py` into the workspace as
`_aka_benchmark.py`. Import `ReadOnlyCacheFlush`, `TimedRun` and
`benchmark_cuda_graph_cold_cache_samples` from that module; allocate the sweep
once outside capture. Preserve all task-owned correctness/replay/input-mutation
checks. Put pre-call snapshots in `prepare_fn`; `before_sample` observers are
rejected because GPU reads there could rewarm operands after eviction. In a
performance case, emit `benchmark_method` and `execution_time_ms` on the row and
the returned timing metadata under `metadata`, so cache identity survives the
protocol and baseline/candidate comparison.
Use `benchmark_paired_cuda_graph_cold_cache` for a paired reference or
extend the trusted evaluator to coordinate arms; a single-role performance action
must never invent a baseline or write framework-owned scores.

This example does not silently migrate existing tasks or invalidate their old
scores in place. A task migration must update its accepted method, docs and
benchmark identity, remeasure both arms, and pass the GPU task validator. Historical
warm-cache scores must not be relabelled as cold-cache measurements.

## Related implementation

The inspected SIKL `sikl/core/utils/testing.py::flush_l2_cache` uses a read-only
1 GiB sweep. Its `bench_cudagraph` captures flush-before-each-call and excludes
flush kernels using profiler records. This reference uses the same eviction
principle with Event boundaries, and therefore has a different timing metric.
The existing `cuda_graph_cold_weights` helper rotates independent weight buffers;
it is a different policy and does not replace this per-call eviction protocol.
