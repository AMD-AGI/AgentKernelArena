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
| `exact` | | Bitwise equality. Required for integer, boolean and packed outputs. |
| `tolerance` | `atol`, `rtol`, `max_violation_fraction` | Fixed tolerances. |
| `sqnr` | `min_db` | Signal-to-noise ratio of the whole output. |
| `output_rounding` | `max_violation_fraction` | Within twice the output dtype's rounding of the reference. |

Use the default unless the operator needs another rule; the QA auditor reports
every departure as a warning that README.md must justify. `k` stays within
`[1, 8]`; `max_violation_fraction` within `[0, 0.01]`. Tune them per task, never
globally. Elements admitted by `max_violation_fraction` must still lie within 16 times the bound, so it absorbs accumulation-order outliers, not skipped or zeroed elements. `element_order: any_in_full_rows` compares sorted rows (top-k ids with
ties).

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
  axes) spans at least **100×**. One or two shapes let an agent tune for them
  alone; too many lengthen every timing round.
- **Boundary correctness cases**, three distinct ones: a var axis at its
  declared minimum, a var axis that is not a power of two, and a var axis that
  is not a multiple of 16 (in units of its `multiple_of`) and larger than 16.
  Correctness cases have no upper limit.
- Unique case IDs and axis/scalar combinations, and values inside the declared
  domain.

The QA auditor additionally requires every performance case to take the
baseline **at least 10 µs**: below that, launch overhead dominates the
measurement. Real shapes that run faster (decode with M=1) stay correctness-only.
A task whose production shapes are all below 10 µs is not a kernel task.

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
- **Sanitizers.** On MI355X the final evaluation also runs the default
  sanitizers between correctness and performance (see
  [Default sanitizers for schema-v3 tasks](use-evaluation-tools.md#default-sanitizers-for-schema-v3-tasks)).
  A gate's definite finding stops performance, so the candidate scores 0,
  unless the baseline shares it. Keep the baseline clean in every lane: a
  finding the baseline shares leaves that lane unable to protect the task.
  The framework runs `python3 -m _aka_eval.evaluate {baseline|candidate}
  sanitize LANE` itself.

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
