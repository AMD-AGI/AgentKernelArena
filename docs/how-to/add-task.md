---
myst:
    html_meta:
        "description": "The AgentKernelArena schema-v3 kernel task contract: operator interface, PyTorch reference, vendored baseline, workloads, run-selected target language, evaluation and QA rules."
        "keywords": "AgentKernelArena, task schema, task.yaml, target language, baseline, reference, workloads, qa_auditor"
---

# Task definition, schema, and authoring

Read this document before adding or modifying a kernel task. It is the canonical
task contract. The end-to-end serving tasks under `tasks/e2e/` keep their own
schema-v2 contract, described in [e2e kernel tasks](e2e-kernel-tasks.md).

## What a task is

A kernel task describes **one operator in one scenario** and nothing about how a
candidate must implement it:

| Part | File | Role |
| --- | --- | --- |
| Interface | `task.yaml` | Axes, inputs, outputs and the call signature every implementation uses. |
| Reference | `reference.py` | High-precision PyTorch (float32/float64) used **only** to judge correctness. |
| Baseline | `baseline/` | The implementation to beat, in any supported language, vendored into the task. |
| Workloads | `workloads.json` | The correctness and performance cases, taken from production use. |
| Inputs | `inputs.py` (optional) | Task-owned generators and layout transforms for inputs a built-in distribution cannot express. |

**The task never names the candidate's language.** The run configuration selects
it (`target_language`, below). The same task can be optimized in Triton, Gluon,
HIP or FlyDSL, and every language is scored against the same baseline on the
same workloads, so their speedups are comparable.

A *scenario* fixes everything except the varying axes: dtypes, constant
dimensions, layouts and attributes. Two scenarios of one operator are two tasks,
for example `tasks/triton/gemm/gemm_a8w8/n4096_k4096/` and `.../n1280_k8192/`.

```text
tasks/<baseline_language>/<category>/<operator>/<scenario>/
  task.yaml
  workloads.json
  reference.py
  inputs.py            # optional
  baseline/
    kernel.py          # Triton, Gluon, FlyDSL or TileLang baseline
    kernel.hip         # or a HIP baseline (plus headers)
    LICENSE            # when the upstream license is not Apache-2.0
  README.md
```

`<baseline_language>` is the task's `baseline.language` (`triton`, `gluon`,
`hip`, `flydsl` or `tilelang`). It records what the vendored baseline is written
in, not what a candidate must use: every target language can optimize every
task. `<category>` is the task's `operator.category`, one of: activation,
attention, convolution, elementwise, embedding, gemm, kv_cache, moe,
normalization, positional, quantization, reduction, sampling, sorting,
state_space, other. `<operator>` is its `operator.name`. The CPU pre-check
`python3 -m src.tools.check_task tasks/<...>/<scenario>` rejects a
package filed anywhere else, so scenarios of one operator with baselines in
different languages sit under different language directories.

The path below `tasks/` is the task ID. Correctness inputs are drawn from seeds
derived from it, so moving a task, including to follow a baseline replaced by
one in another language, changes its inputs and needs a fresh QA audit.

The names `config.yaml`, `solution/` and `_aka_eval/` are reserved: the
framework generates them in each run workspace.

## `task.yaml`

```yaml
schema_version: 3
description: >-
  out = x * rsqrt(mean(x^2) + epsilon) * weight over the hidden axis, rounded to bfloat16.
operator: {name: rms_norm, category: normalization}
interface:
  axes:
    n_tokens: {kind: var, min: 1, max: 262144}
    hidden_size: {kind: const, value: 8192}
  derived_axes: {}                       # optional: name -> integer expression over axes
  inputs:
    x: {kind: tensor, shape: [n_tokens, hidden_size], dtype: bfloat16,
        distribution: {kind: normal, mean: 0.0, std: 1.0}}
    weight: {kind: tensor, shape: [hidden_size], dtype: bfloat16, role: weight,
             distribution: {kind: uniform, low: 0.5, high: 1.5}}
    epsilon: {kind: scalar, dtype: float32, default: 1.0e-6}
  outputs:
    out: {shape: [n_tokens, hidden_size], dtype: bfloat16}
  mutable: []                            # inputs the operator updates in place
  generator: {file: inputs.py, symbol: make_inputs}      # optional
  layout: {file: inputs.py, symbol: to_kernel_layout}    # optional
distributed: {...}                       # optional: a multi-GPU task (see "Multi-GPU tasks")
reference: {file: reference.py, symbol: reference}
baseline:
  language: triton                       # triton | gluon | hip | flydsl | tilelang
  entry: baseline/kernel.py              # baseline/kernel.hip for HIP
  provenance:
    source: vllm-project/vllm
    revision: v0.11.0                    # a commit or release, never a branch
    path: vllm/model_executor/layers/batch_invariant.py
    license: Apache-2.0
    modifications: Wrapped in the run() interface.
compare:
  out: {bound: baseline_relative}        # one entry per output and mutable input
workloads: workloads.json
platform_support: {required_arch: [gfx950]}
```

### Axes and shapes

- A `var` axis varies across workload cases within `[min, max]` and, when given,
  `multiple_of`. A `const` axis is fixed by the scenario. At least one axis is
  `var`.
- Shapes are lists of integers, axis names, or integer expressions using `+ - *
  // %`, `min`, `max` and `ceil_div`.

### Inputs

Inputs are tensors or scalars, in call order. A tensor declares `shape`, `dtype`
and a `distribution`:

| Distribution | Parameters | Use |
| --- | --- | --- |
| `normal` | `mean`, `std` | Floating activations and weights. |
| `uniform` | `low`, `high` | Bounded floating values. |
| `log_uniform` | `log2_low`, `log2_high`, `signed` | Values spanning magnitudes. |
| `randint` | `low`, `high` (exclusive) | Integer data. |
| `permutation` | | A one-dimensional permutation of `0..n-1`. |
| `zeros` | | Only for `role: state` (a cache before its first write). |
| `generator` | | Produced by the task's `generator` function. |

`role` is `activation` (default), `weight`, `state`, `index` or `metadata`.

Inputs that a built-in distribution cannot express (quantized codes with
scales, routing tables, `cu_seqlens`, paged KV caches) use `generator`. The
function receives the case (`id`, `axes` including constants and derived axes,
`scalars`), the already generated built-in inputs, a seeded `torch.Generator`
and the device, and returns exactly its inputs:

```python
def make_inputs(case, inputs, generator, device):
    ...
    return {"topk_ids": ids, "topk_weights": weights}
```

Use only the generator for randomness. Seeds are derived by the framework from
the task ID and case ID; tasks never choose them.

Production kernels often receive weights in a prepared layout (preshuffled
tiles, padded scales). The optional `layout` function maps the logical inputs to
the kernel-facing inputs, outside timing:

```python
def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    inputs["w"] = preshuffle(inputs["w"])
    return inputs
```

The reference receives the logical inputs; baseline and candidate receive the
kernel-facing inputs. Declared shapes describe the kernel-facing tensors. A
layout transform cannot change a mutable input. Activation preparation belongs
in the kernel and is timed.

### Outputs and mutable inputs

The evaluator allocates every output from its declaration and poisons it (NaN
for floating dtypes, random bytes otherwise) before every call. An operator that
updates state in place (a KV-cache write) declares that input in `mutable`; its
final value is checked like an output, and the evaluator restores its initial
value before every call.

### Comparison

`compare` names every output and mutable input:

| Bound | Parameters | Meaning |
| --- | --- | --- |
| `baseline_relative` (default) | `k` (2), `max_violation_fraction` (0), `baseline_min_sqnr_db` (25) | Each element within `k ×` the baseline's own error (or its per-row RMS over the elements that are nonzero in the reference or the baseline; rows with fewer than 8 such elements use the whole output's RMS) plus one output rounding; the mean error must be unbiased relative to the baseline's. |
| `exact` | | Bitwise equality. Required for integer, boolean and packed outputs without `decode`. |
| `tolerance` | `atol`, `rtol`, `max_violation_fraction` | Fixed tolerances. |
| `sqnr` | `min_db` | Signal-to-noise ratio of the whole output. |
| `output_rounding` | `max_violation_fraction` | Within twice the output dtype's rounding of the reference. |

Use the default unless the operator needs another rule; the QA auditor reports
every departure as a warning that README.md must justify. `k` stays within
`[1, 8]`; `max_violation_fraction` within `[0, 0.01]`. Tune them per task, never
globally. Elements admitted by `max_violation_fraction` must still lie within 16 times the bound, so it absorbs accumulation-order outliers, not skipped or zeroed elements. `element_order: any_in_full_rows` compares sorted rows (top-k ids with
ties).

#### Decoded packed outputs

An integral or packed output (quantized codes and their scales packed into one
`uint8` tensor, for example) whose bytes legitimately differ between correct
implementations may declare a `decode` hook with a numeric bound
(`baseline_relative`, `tolerance`, `sqnr` or `output_rounding`):

```yaml
compare:
  index_cache:
    bound: baseline_relative
    decode: {file: inputs.py, symbol: index_cache_values, precision: float8_e4m3fn}
```

- `file` and `symbol` name a top-level function in a `.py` file at the task
  root, like `interface.generator`. It takes the packed tensor and returns its
  values as a float32 or float64 tensor. Write it in PyTorch only, deterministic
  and without modifying its argument; it runs on the device the tensor is on.
- The evaluator decodes the candidate's written tensor, the reference's and,
  for `baseline_relative`, the baseline's with the same function and compares the
  decoded tensors under the bound, with `element_order` applied to the decoded
  rows. The reference returns the packed tensor, with the operator's exact
  quantization and packing.
- `precision` (optional) is the floating dtype the decoded values carry, such as
  `float8_e4m3fn` for E4M3 codes times a scale. Because the decoded reference is
  quantized to it as well, a correct candidate's code may sit one code step from
  the reference's, so the bound's rounding unit is one code step of `precision`:
  twice its unit roundoff. Without `precision` the decoded dtype takes its
  place, which makes the rounding term negligible (use this for integer codes,
  whose step is not relative to the value).
- `decode` is not allowed with `exact` or on a floating output, and the QA
  auditor reports it as a bound override. Elementwise bounds hold every decoded
  element, including state a mutable input keeps unchanged. Keep
  `max_violation_fraction` at 0: with a coarse precision such as E4M3 the
  rounding term alone keeps a zeroed element within 16 times the bound, and when
  most of the tensor is unchanged state a fraction of the whole tensor can
  exceed every newly written element.

Under `baseline_relative`, the baseline itself must pass a sanity check: no
non-finite or unwritten element, and SQNR against the reference of at least
`baseline_min_sqnr_db` (default 25 dB). Without it a wrong baseline would loosen
every candidate's bound.

## `reference.py`

```python
import torch

def reference(*, x, weight, epsilon):
    x64 = x.double()
    return x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + epsilon) * weight.double()
```

- Keyword arguments are the logical inputs. Return the outputs, then the mutable
  inputs' final values: one tensor, a tuple in that order, or a dict by name.
- Compute in float64 (or float32 where float64 is impractical) with PyTorch only.
  Import only `torch`, the standard library math helpers and other task-root
  helper files; never the baseline, a kernel language or a production package.
- Implement exact semantics for quantized outputs: the same rounding mode,
  scale computation and packing as the operator.

## `baseline/`

The baseline is the code a candidate must beat. **All of it lives in the task.**

- Vendor it from its production source (AITER, vLLM, SGLang, FlyDSL, ...) and
  record `provenance`: source, a fixed revision, path(s), license and your
  modifications. Keep the upstream copyright header. Add `baseline/LICENSE` when
  the license is not Apache-2.0.
- It may not import a production package (`aiter`, `vllm`, `sglang`,
  `sgl_kernel`, ...), call `torch.ops`, load a binary, read absolute paths or
  spawn processes.
- Assembly (`.co`) and Composable Kernel baselines cannot be vendored as source.
  Write a functionally equivalent Triton baseline instead and say so in README.md.
  A vendored or rewritten baseline may be slower than the production function;
  it is still the task's baseline.
- Production dispatch logic and tuned configuration tables are part of the
  baseline when they are Python; copy them as is.
- It implements the same entry point as a candidate (below), in its language.

When the run's target language equals the baseline's, the candidate starts as a
copy of the baseline, so the baseline must also satisfy the candidate rules
below. The QA auditor checks this.

## `workloads.json`

```json
{
  "description": "Prefill token counts of ... ; decode batches are correctness-only.",
  "cases": [
    {"id": "t1", "axes": {"n_tokens": 1}, "checks": ["correctness"], "source": "decode, batch 1"},
    {"id": "t2048", "axes": {"n_tokens": 2048}, "checks": ["correctness", "performance"], "source": "prefill chunk"}
  ]
}
```

Each case names every `var` axis, optional `scalars`, its `checks` (`[correctness]`
or `[correctness, performance]`) and its production `source`. Take workloads
from real production use; the framework records the source but cannot verify
it. The framework enforces, when it loads the task:

- **4 to 16 performance cases**, whose var-axis size (the product of the var
  axes) spans at least **100×**, unless the task declares the production-range
  exemption below. One or two shapes let an agent tune for them alone; too many
  lengthen every timing round.
- **Boundary correctness cases**, distinct ones: a var axis at its declared
  minimum, a var axis that is not a power of two, and a var axis that is not a
  multiple of 16 (in units of its `multiple_of`) and larger than 16. The last
  one is required only when some var axis's domain holds such a value (its
  `max` exceeds 16 units); a task whose var axes all stay within 16 units, such
  as rows 1..8 or rows 8..64 with `multiple_of: 8`, needs only the first two.
  Correctness cases have no upper limit.
- Unique case IDs and axis/scalar combinations, and values inside the declared
  domain.

The QA auditor additionally requires every performance case to take the
baseline **at least 10 µs**: below that, launch overhead dominates the
measurement. Real shapes that run faster (decode with M=1) stay correctness-only.
A task whose production shapes are all below 10 µs is not a kernel task.

### Production-range span exemption

Some kernels are specialized to one model: the model fixes every non-batch
dimension, and the shapes production actually sends them that take the baseline
at least 10 µs span less than 100×. Such a task may declare, in
`workloads.json`:

```json
"performance_span": {"policy": "production_range", "evidence": "workload_evidence.json"}
```

Prefer a context or sequence axis first: if the kernel's work depends on the
context length, make it a var axis spanning the production range and meet the
100× rule without the exemption. The evidence file sits at the task root and is
part of the task package:

```json
{
  "source": "AgentX <corpus> rev <revision> + <model> serving config <flags>; or live capture job <id>",
  "method": "corpus_statistics",
  "axes": {
    "n_tokens": {"observed_min": 1, "observed_max": 65536,
                 "perf_min": 2048, "perf_max": 16384,
                 "frequency": {"1": 0.3, "2048": 0.3, "4096": 0.2, "8192": 0.1, "16384": 0.1}}
  },
  "fixed_dimensions": "every non-var axis is fixed by <model> at <TP/EP>",
  "notes": "how the numbers were derived"
}
```

| Field | Meaning |
| --- | --- |
| `source`, `fixed_dimensions` | Where the range comes from (corpus and revision, serving configuration, capture job) and what fixes the other dimensions. |
| `method` | `corpus_statistics` or `live_capture`. |
| `axes.<axis>.observed_min/max` | The full production range of the axis, recorded for transparency. It may span 100× or more. |
| `axes.<axis>.perf_min/max` | The part of that range whose baseline takes at least 10 µs: the performance range. |
| `axes.<axis>.frequency` | Production weight of each observed value (keys are values, weights sum to about 1). |
| `notes` | Optional derivation notes. |

When it loads the task the framework requires: the evidence names exactly the
var axes; `observed_min <= perf_min <= perf_max <= observed_max`, inside each
axis domain, with perf endpoints on the axis's `multiple_of`; frequency keys
inside the observed range and weights summing to 0.9-1.1; the performance range
spans **less than 100×** in var-axis size (otherwise the 100× rule is reachable
and the exemption is refused); and, for every var axis, a performance case at
`perf_min` and one at `perf_max`, with every performance case inside
`[perf_min, perf_max]`. The 4-16 performance-case count and the boundary
correctness cases are unchanged; weight the performance cases toward the
high-frequency values. Shapes below 10 µs stay correctness-only. Never batch
calls or add warm replays to reach the floor: a kernel with fewer than four
performance shapes at 10 µs or more is not a task.

The CPU pre-check reports the exemption. The QA auditor reports it as the
warning `QA-SPAN-EXEMPTION` (observed range, perf range and evidence source)
and fails the task with `QA-SPAN-EVIDENCE` when a `perf_min` or `perf_max`
endpoint case was not measured or takes the baseline under 10 µs;
`QA-TIMING-FLOOR` still applies to every performance case. Held-out cases of an
exempt task stay inside the observed range, and held-out performance cases
inside the performance range.

## The candidate interface

Every implementation, in every language, is one call:

```python
def run(*, x, weight, epsilon, out) -> None: ...
```

It takes the inputs, then the outputs, by keyword; writes every output (and every
mutable input) in place; and returns `None`.

| Language | Entry | Build |
| --- | --- | --- |
| Triton, Gluon, FlyDSL (and TileLang baselines) | `kernel.py::run` | Imported as one flat package; helper `.py` files import relatively (`from .helpers import ...`). |
| HIP | `kernel.hip`, registering `run` | Built with `torch.utils.cpp_extension` for the GPU's architecture. |

A HIP implementation registers exactly one op whose arguments are the call keys
and whose written tensors are `Tensor(a!)`:

```cpp
#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor weight, float epsilon, Tensor(a!) out) -> ()", &run);
}
```

Each run workspace's `_aka_eval/CANDIDATE.md` states the exact signature.

### Candidate rules

All computation runs in kernels written in the target language. The evaluator
enforces:

- **Static rules.** Imports are limited to the language's packages, `torch` and
  a few standard modules; no other kernel language, production package, task,
  reference, baseline or evaluator module. No `eval`/`exec`, reflection, frame
  access, module mutation, `global`, I/O, printing, synchronization, host round
  trips (`.item()`, `.cpu()`), randomness or global PyTorch state. Process, FFI
  and interpreter modules reached as attributes of an allowed package
  (`torch.ctypes`, `torch.os`, `torch.sys`, `torch.builtins`) are rejected, and
  `getattr` takes a literal attribute name (a computed name only on `self`).
  HIP sources
  include only HIP, ATen/c10, `torch/library.h`, standard and own headers, call
  no BLAS/MIOpen/ATen compute and keep no mutable static state. ATen operator and
  backend headers (`ATen/ops/`, `ATen/native/`, `ATen/*Functions.h`, `ATen/cuda/`
  and `ATen/hip/` other than the context, `Exceptions.h`, `Atomic.cuh`,
  `DeviceUtils.cuh` and `MasqueradingAsCUDA` guard/stream headers) and backend operator namespaces
  (`at::cuda::mm_out`, `at::native::`, `at::_ops::`, dispatcher redispatch) are
  rejected: they reach library kernels without the dispatcher, so the runtime
  rules below would not see them.
- **Runtime rules.** During every candidate call, PyTorch may only allocate,
  take views and make same-dtype copies; arithmetic, reductions, dtype
  conversions and library operators raise. Triton, Gluon and FlyDSL candidates
  must launch at least one of their own kernels per call, in their language
  (a Triton target may not launch Gluon kernels and vice versa).
- **Timing.** Each call is captured into a CUDA graph and timed per replay after
  a 1 GiB cache flush. Launches must be capturable on the current stream.

These checks are task-contract enforcement, not a security sandbox.

## Evaluation

The framework writes `_aka_eval/` (the evaluator), `config.yaml` (its internal
execution record) and `solution/` into each workspace. The seven actions are:

```sh
python3 -m _aka_eval.evaluate validate-task
python3 -m _aka_eval.evaluate baseline {compile|correctness|performance}
python3 -m _aka_eval.evaluate candidate {compile|correctness|performance}
```

- **Correctness** runs every case twice, with two framework seeds, on the same
  buffers: the second call loads new inputs into the first call's storage, so a
  cached or stale result fails. Read-only inputs must be unchanged afterwards.
- **Performance** times each performance case with
  `benchmark_cuda_graph_cold_cache_samples` (10 warmups, 100 samples, median),
  alternating the two input draws between samples and checking every timed
  sample's outputs.
- **Scoring.** The per-case speedup is baseline time over candidate time; the
  task's speedup is their geometric mean. Any failed correctness case scores 0.
  The final evaluation remeasures the baseline next to the candidate.

## Multi-GPU tasks

Some production kernels do their tensor-parallel collective inside the kernel:
every rank stores partial results into its peers' buffers over XGMI and polls
its own buffer. An optional `distributed` block makes a task run on several GPUs
of one node. Without it a task is single-GPU, exactly as above.

```yaml
distributed:
  world_size: 4                         # ranks = GPUs of one node, 2..8
  symmetric_bytes: "world_size * n_tokens * hidden_size * 2 + 4096"   # int or axis expression
  shard: {file: inputs.py, symbol: shard_inputs}         # logical inputs -> one rank's inputs
  expected: {file: inputs.py, symbol: rank_expected}     # logical outputs -> one rank's outputs
  ranks_agree: [out]                    # optional: outputs every rank writes bitwise identical
```

| Field | Rule |
| --- | --- |
| `world_size` | Integer from 2 to 8. Every rank has the same shapes. |
| `symmetric_bytes` | Bytes of each rank's framework-owned peer buffer for one call: an integer or an expression over the axes and `world_size`, between 1 byte and 4 GiB in every case. |
| `shard`, `expected` | Functions in a task-root `.py` file, like `generator` and `layout`. |
| `ranks_agree` | Distinct outputs or mutable inputs; omitted means none. |

**One rank's call.** `interface.inputs` and `interface.outputs` describe the
call one rank makes. Shape expressions, derived axes and `symmetric_bytes` may
use the reserved symbol `world_size`; a multi-GPU task cannot declare an axis
of that name.

**Logical inputs and outputs.** The inputs are drawn once per case and seed, as
for a single-GPU task. Every rank draws them identically, and the evaluator
fails the action when the ranks' draws differ. Built-in distributions draw the
declared (one-rank) shape, so they suit replicated tensors; a tensor that is
split across ranks comes from `generator`, which returns its full logical shape
(its `case["axes"]` includes `world_size`).
`reference.py` computes the logical outputs from the logical inputs in high
precision, unchanged. The evaluator then derives each rank's view:

```python
def shard_inputs(inputs, *, rank, world_size):
    """Logical inputs -> this rank's kernel inputs: every declared input, by name."""
    rows = inputs["w"].shape[0] // world_size
    return {**inputs, "w": inputs["w"][rank * rows:(rank + 1) * rows]}


def rank_expected(outputs, *, rank, world_size):
    """Logical reference outputs (and mutable final values) -> what this rank must write."""
    return {"out": outputs["out"]}            # a replicated output: the full tensor
```

`shard` receives copies of the logical inputs and returns every declared input
(scalars included) at the declared one-rank shape; the optional `layout`
transform then runs on its result. `expected` returns every output and mutable
input at its declared shape: the full tensor for a replicated output, the rank's
slice for a sharded one. Both run outside timing. The CPU pre-check
(`python3 -m src.tools.check_task`) and `validate-task` run both for every rank
and reject a missing input or a wrong shape.

**Reserved call inputs.** The evaluator appends four keyword inputs to every
rank's call. A multi-GPU task must not declare inputs or outputs with these names:

| Name | Type | Meaning |
| --- | --- | --- |
| `rank` | int (int32 range) | This process's rank; it runs on `cuda:rank` of the worker's GPU group. |
| `world_size` | int (int32 range) | The task's `world_size`. |
| `peers` | int64 device tensor `[world_size]` | `peers[p]` is the base address of rank `p`'s symmetric buffer mapped into this process; `peers[rank]` is the local buffer. |
| `symmetric` | uint8 device tensor `[symmetric_bytes]` | This rank's symmetric buffer (the same memory as `peers[rank]`). |

The candidate signature is therefore `run(*, <inputs>, <outputs>, rank,
world_size, peers, symmetric)`, and a HIP op schema ends with `int rank, int
world_size, Tensor peers, Tensor(x!) symmetric` (the buffer is written by peers
and by the kernel). `_aka_eval/CANDIDATE.md` states the exact signature.

**Symmetric memory.** Each rank allocates one uncached buffer
(`hipExtMallocWithFlags(..., hipDeviceMallocUncached)`, rounded up to 2 MiB
granules and sized for the largest case), exchanges its HIP IPC handle and maps
every peer's buffer with `hipIpcMemLazyEnablePeerAccess`. The buffers live for
the whole action. Before every call, outside timing, every rank zeroes its whole
buffer and restores `peers`, and no rank starts the call before every rank has
finished its previous call and zeroed its buffer. A call never sees a previous
call's mailbox words, so a fixed flag value is safe.

**Persistent-kernel state.** Keep no host-side state and change no state from
the host in `run` (no `epoch.add_()`). Put peer-visible mailbox words, flags and
guard words in `symmetric` and size `symmetric_bytes` for them: the buffer is
zeroed before every call and never compared. A device-local value the kernel
only reads (a fixed epoch tag) is an ordinary input that `shard` (or
`generator`) initializes, for example `epoch = 1`; every input is reloaded
before every call, and a read-only input the kernel writes fails correctness.
Device-local state the kernel writes is a `mutable` input: the reference returns
its final value (for example `epoch + 1`, or zeros for scratch the kernel
clears), `expected` returns this rank's part and `compare` checks it (`exact` for
integers). State whose final contents depend on the implementation belongs in
`symmetric`.

**Correctness.** For every case and both seeds every rank calls `run` on its
shard and compares each written tensor with its `expected` values under the
task's `compare` bound; `baseline_relative` uses that rank's baseline output.
Read-only inputs and `peers` must be unchanged on every rank. For `ranks_agree`
outputs every rank reports a SHA-256 digest of the written bytes and rank 0
requires equal digests (bitwise identity). A case passes only if every rank
passes and the ranks agree.

**Timing.** Per sample, on every rank: load the sample's inputs, poison the
outputs and zero the symmetric buffer; run the read-only 1 GiB eviction sweep on
the rank's own device; synchronize the device, then a host barrier; rank 0
broadcasts a `CLOCK_MONOTONIC` deadline 2 ms ahead and every rank spins until it,
records its start event, replays its captured graph (one complete call) and
records its end event. The sample's time is **the maximum over ranks of the
per-rank event spans**, and a case's time is the median of those maxima (10
warm-ups, 100 samples). A barrier after every sample keeps the next sample's
zeroing behind every rank's replay. A collective finishes when its slowest rank
finishes, so the maximum is conservative and applies one rule to baseline and
candidate; the shared deadline keeps launch skew to microseconds, where a
barrier alone leaves tens. The row metadata keeps every rank's raw samples
(`benchmark_rank_samples_ms`), the release skew of each sample
(`benchmark_release_skew_us`) and how often each rank was the slowest. Graphs are
captured per rank before timing, and a capture failure on any rank fails every
rank. Every timed sample's outputs are checked on every rank.

**Process model and failures.** A correctness or performance action (and the QA
control suite) starts `world_size` processes. They form a gloo group over a
FileStore in a private temporary directory, on the loopback interface, for the
handle exchange, barriers and result gathering; rank 0 merges the ranks' results
and the launching process prints them as the single `ARENA_EVAL_RESULT=` line. A rank that raises, dies or outlives
the action deadline (`timeout_s` minus 30 s) fails the whole action, and the
remaining ranks are killed. `validate-task` and `compile` run in one process.

**Candidate rules.** Unchanged, plus: kernels may load from and store to peer
memory through `peers` and `symmetric`. Host-side communication is rejected:
`torch.distributed`, `torch.multiprocessing` and storage sharing, RCCL/NCCL,
HIP IPC, peer-access, memory export and allocation APIs statically, and
collective operators at run time.

**Running and budget.** A multi-GPU task runs only on a worker whose GPU group
has exactly `world_size` GPUs, so its run configuration sets
`resources.gpu_groups` (see [parallel runs](parallel-run.md#gpu-groups-for-multi-gpu-tasks));
a mismatch, including an allocation with fewer GPUs than `world_size`, fails
before any container starts. One run cannot mix world sizes, or
multi-GPU with single-GPU tasks. Each task's full evaluation (compile,
correctness and performance, baseline and candidate) must finish in 10 minutes
of wall time on one node. Kernels that compile per specialization (FlyDSL
persistent kernels compile per row count) should keep the number of distinct
compiled shapes small (two or three) and vary another axis, such as the context
length, instead.

The GPU path (HIP IPC, peer stores, per-rank graphs and events) needs a ROCm
PyTorch and is verified only by running a task on MI355X. The framework's CPU
tests cover the process model, sharding, merging and timing aggregation with an
emulated symmetric memory.

## Run configuration

```yaml
agent: {template: claude_code}
target_gpu_model: MI355X
target_language: triton            # triton | gluon | hip | flydsl
tasks:
  - triton/normalization                          # Triton-baseline normalization tasks
  - "*/attention"                                 # attention tasks of every baseline language
  - {path: triton/gemm/gemm_a8w8, target_language: hip}   # per-task override
```

A schema-v3 task cannot run without a target language. A task selected twice
with different languages is rejected. The e2e serving tasks ignore
`target_language`.

The candidate starts as a copy of the baseline when the target language equals
the baseline's, and otherwise as an unimplemented template with the baseline as
read-only reference.

Language support by architecture:

| Language | MI355X (gfx950) | Other architectures |
| --- | --- | --- |
| Triton, Gluon, HIP, FlyDSL | Verified | Implemented; not verified. Report problems with the failing configuration. |

A language whose toolchain is missing fails the run before it starts.

## Quality assurance

Every new or changed task needs a framework-finalized `qa_report.yaml` with
`status: PASS` from the QA auditor on MI355X before its PR. See
[QA auditor](qa-auditor.md) for the rubric. In short: findings are itemized;
any **blocking** finding fails the task; warnings never block but carry a reason
and a fix.

## Writing a task

1. Choose the operator and one scenario; collect its production shapes.
2. Write `reference.py` in float64 PyTorch.
3. Vendor the baseline into `baseline/` with provenance, or write a Triton
   equivalent of an assembly/CK kernel. Wrap it in `run()`.
4. Write `task.yaml` and `workloads.json`; add `inputs.py` if needed.
5. Check it loads: `python3 -c "from src.task_contract import load_task_contract;
   load_task_contract(Path('tasks/.../task.yaml'), task_id='...')"`.
6. Run the QA auditor on MI355X (`make docker-run CONFIG=<qa config>`) until
   `qa_report.yaml` passes.
