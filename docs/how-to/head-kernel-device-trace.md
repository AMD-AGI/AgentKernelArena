# Inspect baseline GPU dispatch for head kernels

The optional `trace` action records actual profiler GPU kernel events for
representative **scored baseline cases**. It helps inspect dispatch under the
current public runtime. It does not change the benchmark, optimize a candidate,
write a score, run full task validation, or certify equivalence to an original
serving scenario.

Start with one task on a GPU node whose declared public image is already pulled:

```bash
CONFIG_PATH=example_configs/top5_validator_glm_bf16_public_mi355x.yaml
python3 src/scripts/top5_head_kernels.py trace --config "$CONFIG_PATH" -- \
  --max-cases 3 --replays 1 --timeout 600
```

This uses the normal Docker image/config identity checks, cache isolation, GPU
selection, and credential-free direct-execution container. No LLM backend or
agent authentication is needed. The command itself performs no GPU allocation.

The defaults sample at most three cases per selected task. Selection takes the
first case of each regime and spreads remaining selections over the scored
case list. The report includes the complete scored-case identities, selected
indexes, and all untraced identities. Optional `--case-index` arguments select
exact positions in that same scored list:

```bash
python3 src/scripts/top5_head_kernels.py trace --config "$CONFIG_PATH" -- \
  --case-index 0 --case-index 4 --max-cases 2 --timeout 600
```

`--max-cases` is limited to 1–16, `--replays` to 1–5, and `--timeout` to at most
3600 seconds across the selected tasks. The remaining budget is passed to each
worker; timed-out worker descendants are stopped. Raw traces above 64 MiB or
4096 GPU kernel events are rejected as oversized diagnostic evidence. Reduce
the selected cases/replays rather than interpreting truncated traces.

## Relationship to the scored benchmark

The tool copies the task to a fresh `workspace_device_trace_*` directory,
materializes the existing canonical helper, and launches its probe through the
task's trusted worker with the **baseline overlay**. It invokes the unchanged
`scripts/_bench.py:selected_cases(..., reference=True)` adapter. Thus shapes,
strides, dtypes, dispatch attributes, output-buffer rules, and scored-case
selection come from the same package APIs as scoring. Correctness-only
robustness cases are not added to the trace set.

In the default `--mode timed-graph`, the tool invokes the existing `measure_case`
and canonical graph helper with the task's unchanged warmup/sample counts.
It retains the `TimedReplay` callback supplied by that helper, checks baseline
replay self-consistency with the existing comparator, and profiles subsequent
replay of **that same graph object**. It is the graph timed in this diagnostic;
it is not a saved graph from an earlier scoring run. Diagnostic timing metadata
is recorded only as context and is not consumed by the evaluator.

The replay callback includes input-state restoration before graph replay.
Consequently the raw trace can include restoration/copy activity as well as
operator kernels. Reports state this explicitly. A kernel is assigned a graph
node only when native profiler metadata supplies that information; the tool
does not manufacture graph-node attribution from a Python function name.

Graph capture or replay validation failure is fatal; there is no automatic
fallback. To inspect an eager baseline separately, select it explicitly:

```bash
python3 src/scripts/top5_head_kernels.py trace --config "$CONFIG_PATH" -- \
  --mode eager --max-cases 2 --timeout 600
```

Eager mode uses the same baseline call and cases, but its trace is labeled
`eager baseline trace; not a timed-graph trace`. It can expose different launch
or allocation behavior and must not be reported as evidence of graph dispatch.

## Evidence and comparison limits

The run directory contains `device-trace.json`, copied task inputs/source
fingerprints, the probe fingerprint, stdout/stderr, and each task's
`build/device_trace_report.json`. Raw per-case Chrome traces live under
`build/device-traces/` and have verified SHA-256 hashes. They can be opened in
Perfetto or another Chrome-trace viewer.

Only GPU events in profiler kernel categories supply device symbols. CPU op
names and HIP launch API names are never substituted for missing GPU events.
No device activity, no GPU kernel events, a profiling error, or an incomplete
worker report causes a nonzero exit. Grid, block, device, stream, correlation,
and graph-node fields are copied when exposed; unavailable fields are listed
explicitly under `missing_metadata`.

The report preserves the current runtime identity, tensor contracts, scored
case identity, declared historical device-symbol text, and hashes/paths of
retained selection-validation evidence. Compare original and current evidence
only where case geometry, dtype/layout, execution mode, symbol, and available
launch metadata are actually comparable. Historical symbols can describe
aggregate serving buckets, and historical per-case grid metadata can be absent.

Matching names and grids do not prove identical code objects, launch parameters,
all scenario coverage, or end-to-end TP8 serving behavior. Device code bytes are
not collected. The tool never sets `original_dispatch_equivalence_certified` or
`framework_PASS_claimed`; successful exit means that the requested sampled
baseline device traces were recorded, with the stated omissions and limits.
