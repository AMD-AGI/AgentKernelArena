# Fused mHC post/pre with RMSNorm: four streams, hidden size 4096

Implement the mHC post operation followed by the next pre operation, preserving
the BF16 intermediate residual and applying RMSNorm to the next layer input.
This task fixes four residual streams, hidden size 4096, and projection width
16384. Its 13 [workload rows](scripts/workload.json) cover
`tokens = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096`.
Only the supplied axis and scalar combinations are in scope; tensor values vary
across seeded initialization and replay checks.

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
aiter.ops.mhc.mhc_fused_post_pre
```

The task imports this installed entrypoint from both wrappers; it does not
depend on a sibling repository or a copied source tree. No task action installs
packages or downloads runtime code. Select the pinned image before materializing
the task; missing packages or this entrypoint are execution failures. The
task-local initializer, independent reference, comparator and evaluation runner
are bundled under `scripts/` and require the image's ROCm PyTorch installation.

## Interface and storage

```python
run(x, residual, post_mix, comb_mix, proj_weight, mix_scale, mix_bias,
    rms_eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters,
    norm_weight, norm_eps)
```

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
iterations. Scalars are validated and preserved. Replay refill changes tensor
values while retaining their addresses and strides.

The [task adapter](scripts/task_api.py) maps tuple outputs to their declared
names before the protected [comparator](scripts/compare/main.py) checks all four
outputs with their declared shapes, dtypes, and device. Every element must satisfy
`abs(actual - expected) <= 1e-2 + 1e-2 * abs(expected)`; NaN and infinity are
rejected. The FP32 mix outputs retain this operator's tolerance because their
computation starts from BF16 residuals. The exported comparator governs
acceptance; the generic policy tolerances do not replace it.

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
