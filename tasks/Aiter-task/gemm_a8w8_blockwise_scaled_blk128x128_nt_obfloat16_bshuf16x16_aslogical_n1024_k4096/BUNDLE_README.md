# Block-scaled FP8 GEMM: N=1024, K=4096, logical activation scales

Implement a block-scaled matrix product with a BF16 output and AITER 16x16
preshuffled weights. This task fixes `n=1024`, `k=4096`, `sn=8`, and `sk=32`.
The 13 [workload rows](scripts/workload.json) cover
`m = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096`.
Only these axis combinations are in scope; input values change across seeded
initialization and timing draws.

## Candidate, baseline, and runtime

[config.yaml](config.yaml) declares `language: flydsl` and `initial_state: unimplemented`.
[source/kernel.py](source/kernel.py) is the empty generation target: it defines no
builder, and task validation verifies that state without executing it. It is the
only editable file.

The candidate must implement its own GPU computation in FlyDSL. It must not call
the production AITER operator, the protected baseline or reference, or another
library operator to perform that computation. The runner statically rejects
imports other than FlyDSL, PyTorch and a small set of standard-library modules,
imports of AITER or the task scripts, library matrix products and torch
normalizations. This guards against ordinary violations, not against arbitrary
reflection; every checked invocation is also compared with the reference.

The protected [baseline](scripts/baseline/main.py) calls this installed production
entrypoint:

```text
aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle
```

The validator run configuration pins MI355X (`gfx950`) and this immutable image:

```text
lmsysorg/sglang@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
```

That image supplies the installed AITER package, its GPU backend dependencies and
ROCm PyTorch; the candidate also requires FlyDSL in the selected runtime. No task
action installs packages or downloads runtime code, and missing packages or the
entrypoint are execution failures. The task-local initializer, independent
reference, comparator and evaluation runner are bundled under `scripts/`.

## Interface and storage

```python
build_gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_aslogical_n1024_k4096_module(*, m, n, k, sn, sk) -> launch
launch(a, b, a_scale, b_scale) -> out
```

The builder receives every declared axis of a workload row as a keyword argument and
returns `launch`, which the runner calls with the definition's inputs as keyword
arguments. Prepare compilation, shape-dependent choices and reusable scratch in the
builder; `launch` must recompute the outputs from the current input values on every
call.

| Argument | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `a` | `[m, 4096]` | `torch.float8_e4m3fn` | Logical activation values |
| `b` | `[1024, 4096]` | `torch.float8_e4m3fn` | Preshuffled weight payload |
| `a_scale` | `[m, 32]` | `torch.float32` | Logical activation-scale values, described below |
| `b_scale` | `[8, 32]` | `torch.float32` | Scale for each 128-by-128 weight block |
| `out` | `[m, 1024]` | `torch.bfloat16` | Returned matrix product |

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
`[0.125, 1.0)`, and applies the declared payload encodings in place. The base draw's
seed is derived from the policy seed and workload UUID; timing draws use fresh
seeds and keep input addresses and strides.

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
`baseline`/`candidate` actions for `compile`, `correctness`, and `performance`, and
reports every workload row separately. Task validation checks deterministic
initialization, the reference against itself, rejection of deliberately incorrect
finite outputs, that the timing draws vary, and that the candidate is still the
unimplemented target. Compilation executes every row once; correctness compares
every row with the protected reference and comparator. No invocation may modify
its inputs.

Run the task through the repository's Docker task-validator workflow on MI355X
(`gfx950`), as required by [config.yaml](config.yaml). The framework materializes
the shared benchmark helper before invoking the task-local runner. A successful
runner action alone is not a framework-finalized task qualification report.

Performance uses the policy in [scripts/workload.json](scripts/workload.json): 20
warmups, 100 repetitions and a 1 ms target for the shared GPU graph/event benchmark
helper, with the same workload and protocol for baseline and candidate. Input
generation and reference evaluation stay outside the timed invocation; all GPU work
needed to produce the returned outputs belongs inside it.

Each timed sample is one invocation in one graph replay. Before every sample a
fresh draw of the call-varying inputs is loaded into the same buffers, while
`b`, `b_scale` stay fixed as a production caller holds them. The draws come from
seeds chosen by the operating system when the row is timed. Eight samples, chosen
secretly, have their outputs compared with the reference on the draw they
consumed. After the samples, the timed invocation runs once over each of four
draws it has never read; those outputs are compared too, and the fastest of these
invocations may take at most 1.5 times the reported time. A capture that batches
several invocations into one replay, a mismatch of any checked output, an input
modification, or unseen-draw invocations slower than that bound fail the row.

## Implementation boundary

Define `build_gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_aslogical_n1024_k4096_module`, the builder declared in [config.yaml](config.yaml),
in [source/kernel.py](source/kernel.py), the only editable file. Keep the protected
scripts, workload rows, dtypes, comparison thresholds, and benchmark policy
unchanged.

## Production baseline numerical evidence

The production AITER dispatch remains the performance baseline. It has an
explicit `diagnostic` numerical policy for one case; the candidate must still
satisfy the complete reference comparison, including the timed invocations.
This changes no tolerance, workload, warmup, sample count, input/output
contract, or candidate acceptance rule.

On MI355X/gfx950 with AITER commit `dbd8bf5bd624120197a7a26780a8c72201824f0f`,
AITER's tuned table selects, for `M = 128, N = 1024, K = 4096`, the row
`gfx950,256,128,1024,4096,asm,5,6` of
`aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3.5_397b.csv`
(SHA256 `65246705468a77baacc29af9831825efdbba78b8aab5e324d484463f4ddfea97`):
the assembly kernel `_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E`
with `splitK = 6`. The neighbouring cases select a CK kernel without split-K.
Over four input draws with five calls each, none of the 20 `m_128` outputs met
the task comparison (about 16% of the elements outside tolerance, maximum
absolute error 1.0), and the five calls on one draw produced five different
outputs. All other twelve cases passed the comparison. These measurements are
task-specific GPU evidence for this runtime; they do not qualify other runtime
versions.

During evaluation, baseline correctness still emits the actual per-case
PASS/FAIL, and baseline timing records its full numerical comparison of the
timed outputs. Only completed finite numerical mismatches may be diagnostic.
Crashes, missing cases, compile errors, dependency failures, invalid outputs
and stale cached answers remain failures.
