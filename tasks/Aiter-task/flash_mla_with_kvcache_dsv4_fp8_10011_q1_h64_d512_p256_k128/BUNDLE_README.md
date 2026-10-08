# Flash MLA with packed FP8 KV cache

Implement `flash_mla_with_kvcache_dsv4_fp8_10011_q1_h64_d512_p256_k128` through `run(**kwargs)` in
[source/kernel.py](source/kernel.py). This is an implemented Python starting
point to rewrite in Triton. The provided baseline and the unchanged initial
candidate call the installed SGLang production operator described below.

## Initial candidate, final implementation, and runtime

[config.yaml](config.yaml) declares `initial_state: implemented`,
`initial_language: python`, and target `language: triton`. During initial
`task_validation`, the executable Python wrapper in
[source/implementation/main.py](source/implementation/main.py) is the declared
starting implementation. It calls the same production operator as the separate
protected [baseline](scripts/baseline/main.py). This dependency is allowed for
those two roles. Initial qualification checks their executable behavior, full
numerical contract, and timed replay; it does not certify a completed rewrite.

The final submitted candidate must implement its own GPU computation in Triton.
It must not call the production SGLang operator, the protected baseline or
reference, or another library operator to perform that computation. Replace the
initial wrapper in the declared editable files while retaining `run(**kwargs)`.
The baseline remains protected and separate from candidate edits. All workload,
accuracy, input immutability and measured-replay requirements apply in both
phases; initial-language support does not waive the final Triton requirement.

The required GPU runtime is MI355X (`gfx950`) with the following immutable image:

```text
lmsysorg/sglang@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
```

This image supplies the installed SGLang package and its TileLang/AITER GPU
backend dependencies, plus ROCm PyTorch and Triton. Its exact package set is
bound to the image digest. The provided baseline and initial wrapper require:

```text
sglang.kernels.ops.attention.dsa.tilelang_kernel.dpsk_v4_fp8_attention_fwd
```

The task imports this installed entrypoint through
[scripts/baseline/main.py](scripts/baseline/main.py); it does not depend on a
sibling repository or a copied source tree. No task action installs packages or
downloads runtime code. Select the pinned image before materializing the task;
missing packages or this entrypoint are execution failures. The task-local
initializer, independent FP32 reference, comparator and evaluation runner are
bundled under `scripts/` and require the image's ROCm PyTorch installation.

## Interface and semantics

The exact tensor shapes, dtypes, scalar values, workload identities and callback
specifications are in [scripts/workload.json](scripts/workload.json). Retain all
13 original batch sizes (1 through 4096). Q is BF16 with shape `[batch, 1, 64, 512]`.
The packed KV format stores each page's FP8/BF16 payloads before its scale slots;
see [the reference decoder](scripts/reference/main.py). Inputs are functional and
must remain unmodified. The output is BF16 attention output plus FP32 LSE.

Sparse index widths (128) are capacities, not effective lengths.
Each runtime length selects an index prefix, and every negative index inside
that prefix is ignored. Nonnegative indices outside the prefix must have no
effect. Main and extra branches, when present, have independent lengths and
padding. Attention sinks participate in output normalization and are excluded
from returned LSE. Completely empty rows return exactly zero output and `+inf`
LSE. Finite output uses additive `atol=rtol=1e-2`; finite LSE uses additive
`atol=rtol=1e-3`, enforced by [the comparator](scripts/compare/main.py).

## Executed coverage

[scripts/mla_coverage.py](scripts/mla_coverage.py) defines the runtime profiles
stored explicitly in the workload manifest. This task has 29 independently
reported cases; every case participates in both correctness and performance.
The original shape cases remain. Batch 1 additionally sweeps empty, short,
intermediate and full lengths, including values immediately around 32, 64, 128,
256, 512 and 8192 when within its capacity. Main and extra sweeps hold the other
branch fixed, and joint cases exercise both-empty, either-empty and both-active
combinations. In-prefix holes and entirely negative prefixes are independent of
the length sweep; unselected suffixes retain legal nonnegative indices.

Larger batches mix boundary rows, independently drawn lengths and padding,
including a full row. Every timed replay is checked numerically, then checked
again on a second data seed with changed runtime lengths in the same buffers.
Both Q and the full packed caches, indices and sinks are refilled. This catches
stale outputs and host-side assumptions about tensor contents. Original and
refilled checks together cover multiple data draws for each boundary regime.

## Measurement and validation

Use [scripts/task_runner.py](scripts/task_runner.py) for all seven v2 actions.
Input initialization, reference calculations and checks occur outside kernel-only
timing. The benchmark retains 20 warmups, 100 repetitions and a 1 ms target,
with the same device timing helper and allocation boundaries for baseline and
candidate. Each original shape and each added runtime profile has its own case
identity and latency in the structured result; baseline and candidate manifests
must match.

Cache initialization batches pages into bounded scratch buffers. Within one
shape group, immutable seeded cache templates are copied into independent input
storage to avoid regenerating the large KV pools for every scalar-length case.
At most the original and refill seed templates are retained; templates are
released on a shape change. Adjacent batch-1 cases reuse input storage. Full
cache copies and all setup remain outside timing, and input-mutation checks
remain enabled.

Run `python3 scripts/task_runner.py validate-task`, followed by the baseline and
candidate `compile`, `correctness`, and `performance` actions in the pinned GPU
runtime. A fresh framework-finalized `task_validator` PASS on the declared
architecture is required for qualification. CPU regression tests check coverage,
reference semantics and rejection behavior; they do not qualify GPU execution.
