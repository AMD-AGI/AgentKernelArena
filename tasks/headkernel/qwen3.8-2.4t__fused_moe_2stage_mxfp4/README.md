# qwen3.8-2.4t__fused_moe_2stage_mxfp4

This task tunes the native SG520 `aiter.fused_moe:fused_moe` dispatch on MI355X
through a constrained `block_size_M` policy. The editable file is
`source/routing.py`; its only function, `select_routes()`, returns a literal
mapping for both original captured cases. The evaluator parses this mapping as
data and never imports or executes candidate Python.

| Case | Hidden states | Top-k routes | Capture topology |
| --- | --- | --- | --- |
| `decode_m64_live` | BF16 `[64, 8192]`, stride `[8192, 1]` | `[64, 10]` | TP8/EP8, 64 local experts of 512 |
| `prefill_m8192_live` | BF16 `[8192, 8192]`, stride `[8192, 1]` | `[8192, 10]` | TP8/EP8, 64 local experts of 512 |

Every case may select `captured`, `block32`, `block64`, or `block128`. `captured`
passes the original keyword arguments unchanged. The other choices set only the
native `block_size_M` hint to 32, 64, or 128. The native implementation still
owns dispatch and every device kernel. A hint that is incompatible with its
selected kernel fails correctness or execution and fails the entire task.
Imports, callbacks, decorators, extra functions, module statements, dynamic
expressions, changed cases, and unlisted route values are rejected at compile.
This scope does not permit arbitrary device-source optimization.

The protected runtime verifies the installed SG520 `fused_moe.py` against
`source_provenance` in [ut/meta.json](ut/meta.json) and its frozen source copy.
Baseline reference calls use that image-native function directly. Both the
baseline and candidate graph paths use the same native function and the same
captured arguments, with the policy hint as the only candidate-dependent value.
The previous candidate-owned `baseline_callable` export and overlay are removed.

The fixture preserves packed MXFP4 weights (`w1 [64,4096,4096]`,
`w2 [64,8192,1024]`), uint8 scales (`w1_scale [64,4096,256]`,
`w2_scale [64,8192,64]`), the 512-element expert mask, and the captured top-k ids
and weights. The runtime checks the original 2,083,145,699-byte fixture's full
SHA256 before loading it. Fresh correctness and timing inputs change hidden
values only. Their shape, stride, dtype, and every remaining argument stay at
the captured ABI.

Run these commands in the digest-pinned SG520 image declared by `config.yaml`,
after the repository preparation tool has installed the registered original
fixture and materialized the canonical benchmark helper into the task workspace:

```bash
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

The compile phase performs CPU policy syntax validation and syntax-checks the
protected native Python source. It does not compile a GPU kernel. Native GPU
compilation and dispatch occur during correctness. Correctness checks the
frozen captured output, three fresh hidden-input draws, fresh independent output
storage, input immutability, two fresh graph values, no-op and corrupted graph
outputs, both MoE device stages, and the original two-repeat eager timing smoke.
The mixed tolerance remains `abs(out-ref) <= 0.02 * RMS(ref) + 0.02 * abs(ref)`
with the original `RMS(ref)` floor of `1e-6`.

Performance repeats those correctness gates before each case and uses the
workspace-materialized canonical graph helper: 10 warmups, 100 ordered device
samples, and one invocation per replay. Each preparation creates fresh hidden
values on CPU and poisons the previous output plus every exposed sorting
scratch tensor outside timing. Each completed invocation snapshots the output
and all input tensors to CPU before evaluating the protected native reference.
The exact graph that supplied the samples is replayed and checked again. Reports
require 114 checked invocations and 116 input preparations, accounting for the
helper's estimate, capture, prime and final validation operations. Event fallback,
missing cases, missing samples, stale source bindings and altered workload
metadata cannot produce a successful performance report.

Every phase has a new request id and atomically publishes a complete report to
`build/<phase>_report.json`, retaining request, raw worker output and logs under
`build/reports/<request_id>/`. Historical `ut/selection_validation.json`,
`ut/negative_check.json`, capture metadata and telemetry remain immutable.
`ut/unittest.py` is a compatibility entrypoint for the complete correctness phase.

Ordinary framework scoring and task validation require the explicit
`qwen_mxfp4_moe_graph_v1` quality adapter. It rechecks the protected request and
complete raw report, retaining all samples and the selector's byte hash. Gain
attribution uses the validated effective routing choices: unchanged choices,
including comment-only edits, earn no speedup. An extreme timing outlier in
either leg rejects the complete comparison. The harness guard freezes the full
task input package while keeping the constrained selector editable.

CPU regressions exercise the literal boundary, unchanged-source normalization,
complete report validation, native hint routing, mutation rejection before the
reference, and the actual canonical graph adapter with a simulated device. Run
these from a newly materialized task copy:

```bash
make materialize-perf-task TASK=tasks/headkernel/qwen3.8-2.4t__fused_moe_2stage_mxfp4 OUT=/tmp/moe-routing-check
python3 /tmp/moe-routing-check/headkernel/qwen3.8-2.4t__fused_moe_2stage_mxfp4/scripts/test_native_routing.py -v
```

The workload fixture and historical profiler/selection evidence come from SG518,
TP8/EP8. The native source is refreshed to SG520 using the digest in
[ut/runtime_refresh.json](ut/runtime_refresh.json). This is not a new SG520
TP8/EP1 serving capture. Fresh full GPU task-validator qualification is required
before this modified task can be treated as qualified. Historical profile shares
and serving uplifts in `config.yaml` remain provenance, not results from this
routing task.
