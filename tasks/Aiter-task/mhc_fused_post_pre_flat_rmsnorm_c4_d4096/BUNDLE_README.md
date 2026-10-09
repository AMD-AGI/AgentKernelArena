# Fused mHC post/pre with RMSNorm: four streams, hidden size 4096

Implement the mHC post operation followed by the next pre operation, preserving
the BF16 intermediate residual and applying RMSNorm to the next layer input.
This task fixes four residual streams, hidden size 4096, and projection width
16384. Its 13 [workload rows](scripts/workload.json) cover
`tokens = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096`.
Only the supplied axis and scalar combinations are in scope; tensor values vary
across seeded initialization and timing draws.

## Candidate, baseline, and runtime

[config.yaml](config.yaml) declares `language: flydsl` and `initial_state: unimplemented`.
[kernel.py](kernel.py) is the empty generation target: it defines no builder, and
task validation verifies that state without executing it. It is the only editable
file, and the candidate is a single self-contained FlyDSL file.

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
aiter.ops.mhc.mhc_fused_post_pre
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
build_mhc_fused_post_pre_flat_rmsnorm_c4_d4096_module(*, tokens, streams, hidden_size, projection_size, mixes, scale_groups, one) -> launch
launch(x, residual, post_mix, comb_mix, proj_weight, mix_scale, mix_bias,
       rms_eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters,
       norm_weight, norm_eps) -> (next_post_mix, next_comb_mix, layer_input, next_residual)
```

The builder receives every declared axis of a workload row as a keyword argument and
returns `launch`, which the runner calls with the definition's inputs as positional
arguments in this order. Prepare compilation, shape-dependent choices and reusable
scratch in the builder; `launch` must recompute the outputs from the current input
values on every call.

All tensor inputs are independent, contiguous, and on the same GPU. Inputs are
functional and must not be modified or used as aliased output storage.

| Tensor input | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `x` | `[tokens, 4096]` | BF16 | Current layer output |
| `residual` | `[tokens, 4, 4096]` | BF16 | Incoming residual streams |
| `post_mix` | `[tokens, 4]` | FP32 | Flat incoming post gates |
| `comb_mix` | `[tokens, 4, 4]` | FP32 | Incoming stream-combination matrix |
| `proj_weight` | `[24, 16384]` | FP32 | Projection into pre, post, and combination logits |
| `mix_scale` | `[3]` | FP32 | Scale for each logit group |
| `mix_bias` | `[24]` | FP32 | Bias for all logits |
| `norm_weight` | `[4096]` | BF16 | Final RMSNorm weight |

All rows use Python float scalars `rms_eps = pre_eps = sinkhorn_eps = norm_eps =
1e-6`, `post_multiplier = 2.0`, and Python integer `sinkhorn_iters = 20`.
Preserve their values and types.

Return the four outputs in this order, matching the protected reference and
production entrypoint:

| Output | Shape | Dtype |
| --- | --- | --- |
| `next_post_mix` | `[tokens, 4, 1]` | FP32 |
| `next_comb_mix` | `[tokens, 4, 4]` | FP32 |
| `layer_input` | `[tokens, 4096]` | BF16 |
| `next_residual` | `[tokens, 4, 4096]` | BF16 |

## Mathematical result

The independent [reference](scripts/reference/main.py) defines the following
stages. Except for the explicit BF16 conversions, calculations use FP32.

1. Combine incoming streams and add the gated current layer output:
   `next_residual[t,j,h] = BF16(x[t,h] * post_mix[t,j] +
   sum_i comb_mix[t,i,j] * residual[t,i,h])`.
   The transpose of `comb_mix` matters. Convert this result to BF16 before the
   next projection and reduction; retaining an FP32 intermediate changes the
   operation.
2. Flatten the BF16 residual into `[tokens, 16384]` and convert it to FP32 as
   `F`. Compute `Z = (F @ proj_weight.T) * rsqrt(mean(F**2, dim=1) + rms_eps)`.
3. Split the 24 logits into four pre logits, four post logits, and sixteen
   combination logits. Apply the corresponding `mix_scale` entry and
   `mix_bias` slice. The pre gates are `sigmoid(pre_logits) + pre_eps`;
   `next_post_mix` is `sigmoid(post_logits) * post_multiplier`, with a trailing
   singleton dimension.
4. Reshape the combination logits into `[tokens, 4, 4]`, apply softmax over the
   last dimension, add `sinkhorn_eps`, and divide by the column sums plus
   `sinkhorn_eps`. Repeat row normalization followed by column normalization
   `sinkhorn_iters - 1` times, adding the epsilon to each denominator. This is
   `next_comb_mix`.
5. Sum `next_residual` across the four streams using the pre gates. Apply
   RMSNorm with `norm_eps` and `norm_weight`, then convert to BF16 to obtain
   `layer_input`.

## Baseline, initialization, and comparison

The provided [baseline](scripts/baseline/main.py) calls
`aiter.ops.mhc.mhc_fused_post_pre` with the canonical inputs mapped to its
production parameter names. The protected reference expresses the same stages
with PyTorch operations and does not call that AITER operator. The complete
post, projection, gate, Sinkhorn, aggregation, and RMSNorm work is included in
both baseline and candidate invocations.

The [initializer](scripts/initialize/main.py) samples normal BF16 activations
and residuals, incoming post gates uniformly from `[0.1, 0.9)`, and projection
weights with standard deviation `1 / sqrt(16384)`. It sets `mix_scale` to one,
uses normal `mix_bias` values with standard deviation 0.1, and samples
`norm_weight` from `[0.5, 1.5)`. Incoming combination matrices are positive and
approximately doubly stochastic after softmax and 20 row/column normalization
iterations. Scalars are validated and preserved. Timing draws change tensor
values while retaining their addresses and strides.

The [task adapter](scripts/task_api.py) maps tuple outputs to their declared
names before the protected [comparator](scripts/compare/main.py) checks all four
outputs with their declared shapes, dtypes, and device. Every element must satisfy
`abs(actual - expected) <= 1e-2 + 1e-2 * abs(expected)`; NaN and infinity are
rejected. The FP32 mix outputs retain this operator's tolerance because their
computation starts from BF16 residuals. The exported comparator governs
acceptance; the generic policy tolerances do not replace it.

## Baseline numerical policy

The reference is the bundle's FP32 PyTorch computation and its comparator. The
production baseline is the performance reference and is not required to meet
that comparison: [config.yaml](config.yaml) declares
`correctness_policy: diagnostic` for the deepseek-v4-flash tasks. Baseline
correctness still reports every case's actual PASS/FAIL, and baseline timing
keeps the full comparison of its timed outputs. Only a completed finite numerical
mismatch is accepted; crashes, missing cases, invalid outputs and input mutation
remain failures. Candidates have no exception: candidate correctness and every
checked timed invocation must pass the comparator.

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
`proj_weight`, `mix_scale`, `mix_bias`, `norm_weight` stay fixed as a production caller holds them. The draws come from
seeds chosen by the operating system when the row is timed. Eight samples, chosen
secretly, have their outputs compared with the reference on the draw they
consumed. After the samples, the timed invocation runs once over each of four
draws it has never read; those outputs are compared too, and the fastest of these
invocations may take at most 1.5 times the reported time. A capture that batches
several invocations into one replay, a mismatch of any checked output, an input
modification, or unseen-draw invocations slower than that bound fail the row.

## Implementation boundary

Define `build_mhc_fused_post_pre_flat_rmsnorm_c4_d4096_module`, the builder declared in [config.yaml](config.yaml),
in [kernel.py](kernel.py), the only editable file. Keep the protected scripts,
workload rows, dtypes, comparison thresholds, and benchmark policy unchanged.
`python3 test_kernel_harness.py <action>` runs the same actions as the framework.
After acceptance, `scripts/export_solution.py` writes the candidate and a tensor-call
binding as a SIKL solution; it never computes or writes Arena scores.
