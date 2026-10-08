# Block-scaled FP8 GEMM: N=4096, K=512, logical activation scales

Implement a block-scaled matrix product with a BF16 output and AITER 16x16
preshuffled weights. This task fixes `n=4096`, `k=512`, `sn=32`, and `sk=4`.
The 13 [workload rows](scripts/workload.json) cover
`m = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096`.
Only these axis combinations are in scope; input values change across seeded
initialization and replay checks.

## Initial candidate, final implementation, and runtime

[config.yaml](config.yaml) declares `initial_state: implemented`,
`initial_language: python`, and target `language: triton`. During initial
`task_validation`, the unchanged Python wrapper in
[source/implementation/main.py](source/implementation/main.py) is the declared
starting implementation. It calls the same production operator as the separate
protected [baseline](scripts/baseline/main.py). This dependency is allowed for
those two roles. Initial qualification checks their executable behavior, full
numerical contract, and timed replay; it does not certify a completed rewrite.

The final submitted candidate must implement its own GPU computation in Triton.
It must not call the production AITER operator, the protected baseline or
reference, or another library operator to perform that computation. Replace the
initial wrapper in the declared editable files while retaining `run(**kwargs)`.
The baseline remains protected and separate from candidate edits. All workload,
accuracy, input immutability and measured-replay requirements apply in both
phases; initial-language support does not waive the final Triton requirement.

The required GPU runtime is MI355X (`gfx950`) with the following immutable image:

```text
lmsysorg/sglang@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
```

This image supplies the installed AITER package and its GPU backend dependencies,
plus ROCm PyTorch and Triton. Its exact package set is bound to the image digest.
The provided baseline and unchanged initial wrapper require this entrypoint:

```text
aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle
```

The task imports this installed entrypoint from both wrappers; it does not
depend on a sibling repository or a copied source tree. No task action installs
packages or downloads runtime code. Select the pinned image before materializing
the task; missing packages or this entrypoint are execution failures. The
task-local initializer, independent reference, comparator and evaluation runner
are bundled under `scripts/` and require the image's ROCm PyTorch installation.

## Interface and storage

```python
run(a, b, a_scale, b_scale) -> out
```

| Argument | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `a` | `[m, 512]` | `torch.float8_e4m3fn` | Logical activation values |
| `b` | `[4096, 512]` | `torch.float8_e4m3fn` | Preshuffled weight payload |
| `a_scale` | `[m, 4]` | `torch.float32` | Logical activation-scale values, described below |
| `b_scale` | `[32, 4]` | `torch.float32` | Scale for each 128-by-128 weight block |
| `out` | `[m, 4096]` | `torch.bfloat16` | Returned matrix product |

Inputs are independent contiguous tensors on the same GPU. The operation is
functional: do not modify input contents or return an output that aliases an
input. There is no bias input in this task.

`a_scale` contains ordinary logical row-major values: `S_a = a_scale`.
The production baseline creates a column-major view with
`a_scale.T.contiguous().T` inside its callback before calling AITER. This relayout
is part of this task's baseline execution; the input itself remains contiguous
and unchanged.

Decode the shuffled weight payload into the logical weight matrix `W` as in the
protected [reference](scripts/reference/main.py):

```python
W = b.reshape(n // 16, k // 32, 2, 16, 16).permute(0, 3, 1, 2, 4).reshape(n, k)
```

This is the inverse of AITER's `shuffle_weight(layout=(16, 16))` encoding for
one-byte FP8 weights. The existing attribution is retained in the reference.
Do not interpret `b` as an ordinary row-major weight matrix.

## Mathematical result and checks

With `S_a` decoded as above, the reference converts operands and scales to FP32,
then computes:

```text
A_dequant[i, r] = float32(a[i, r]) * S_a[i, r // 128]
B_dequant[j, r] = float32(W[j, r]) * b_scale[j // 128, r // 128]
out = bfloat16(A_dequant @ B_dequant.T)
```

The [initializer](scripts/initialize/main.py) draws normal activation and weight
values, converts them to FP8, draws nonuniform positive scales uniformly from
`[0.125, 1.0)`, and applies the declared payload encodings in place. The seed is
derived from the policy seed and workload UUID; replay refill uses a new seed
without changing input addresses or strides.

The independent FP32 reference imports no AITER operator. The protected
[comparator](scripts/compare/main.py) requires identical output shape, BF16 dtype,
and device, and rejects NaN or infinity. Every element must satisfy either
absolute error `<= 1e-2` or relative error `<= 1e-2`, where relative error is
`abs(actual - expected) / (abs(expected) + 1e-8)`. The exported comparator governs
acceptance; the generic policy tolerances do not replace it.

The provided [baseline](scripts/baseline/main.py) calls
`aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle` with BF16 output.
It allocates the output inside the callback; this is not a persistent-output
contract. Use the configured runtime with proxying and schema dumping disabled
(`SIKL_DISABLE_PROXY=1`, `SIKL_SCHEMA_DUMP=0`). Device timing measures the callback's
GPU work; it does not measure end-to-end host API latency.

## Evaluation and timing

The protected [runner](scripts/task_runner.py) implements `validate-task` and
`baseline`/`candidate` actions for `compile`, `correctness`, and `performance`.
Compilation executes every workload to exercise lazy GPU compilation. Task
validation checks deterministic initialization, the reference against itself,
and rejection of deliberately incorrect finite outputs. Correctness uses the
protected reference and comparator for every workload row.

Run the task through the repository's Docker task-validator workflow on MI355X
(`gfx950`), as required by [config.yaml](config.yaml). The framework materializes
the shared benchmark helper before invoking the task-local runner. A successful
runner action alone is not a framework-finalized task qualification report.

Performance uses the policy in [scripts/workload.json](scripts/workload.json):
20 warmups, 100 repetitions, and a 1 ms target for the shared GPU graph/event
benchmark helper. Baseline and candidate use the same workload and timing
policy. Input generation and reference evaluation stay outside the timed
callback; all GPU work needed to produce the returned outputs belongs inside it.
The runner poisons returned outputs and checks the exact measured replay, then
refills the same input buffers with new seeded values and checks replay again.
A candidate must recompute from current input values, preserve the inputs, and
write every output on each invocation. These checks do not permit cached answers
or input/output aliasing.

## Implementation boundary

Expose `run(**kwargs)` from [source/kernel.py](source/kernel.py). The only
editable files are that entrypoint and
[source/implementation/main.py](source/implementation/main.py), as declared in
[config.yaml](config.yaml). The unchanged initial Python wrapper is permitted to
call the installed production operator during initial `task_validation`, as
described above. The final submitted candidate must implement
its own GPU computation in Triton and must not delegate that computation to the
protected baseline, reference, or a library operator. Keep the protected scripts,
workload rows, dtypes, comparison thresholds, and benchmark policy unchanged.
