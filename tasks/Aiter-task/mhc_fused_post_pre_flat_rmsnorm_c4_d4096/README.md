# Fused mHC post→pre to FlyDSL

Implement the fused manifold-constrained hyper-connection (mHC) step with
`HC = streams = 4` residual streams and `D = hidden_size = 4096`: the post step
of one layer followed by the pre step of the next, with an output RMSNorm. The
fixed axes, scalar values, seed and all variable `tokens` cases are in the
declared workload JSON. Every computation below is per token `t`.

**Post.** With FP32 arithmetic, rounded to BF16:

```text
next_residual[t, j, :] = sum_i comb_mix[t, i, j] * residual[t, i, :] + post_mix[t, j] * x[t, :]
```

**Pre**, on the BF16 `next_residual`, with `f = next_residual[t]` flattened to
`streams * hidden_size` values:

```text
mixes      = (f @ proj_weight.T) * rsqrt(mean(f^2) + rms_eps)                       # 24 values
pre[i]     = sigmoid(mixes[i]      * mix_scale[0] + mix_bias[i])      + pre_eps      # i < 4
post[i]    = sigmoid(mixes[4 + i]  * mix_scale[1] + mix_bias[4 + i])  * post_multiplier
comb       = softmax(reshape(mixes[8:] * mix_scale[2] + mix_bias[8:], 4x4 row-major), last axis) + sinkhorn_eps
comb       = comb / (column sums + sinkhorn_eps)
repeat sinkhorn_iters - 1 times: comb = comb / (row sums + eps); comb = comb / (column sums + eps)
s          = sum_i pre[i] * next_residual[t, i, :]
layer_input = s * rsqrt(mean(s^2) + norm_eps) * norm_weight                           # BF16
```

The configured builder signature is `builder(tokens, streams, hidden_size) ->
launch`, invoked with keyword arguments. Launch takes the definition's inputs
as positional arguments in declared order and returns the four outputs as a
tuple in declared order (a mapping keyed by the output names is also accepted):

```python
launch(x, residual, post_mix, comb_mix, proj_weight, mix_scale, mix_bias,
       rms_eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters,
       norm_weight, norm_eps) -> (next_post_mix, next_comb_mix, layer_input, next_residual)
```

| Argument | Shape / value | Dtype |
| --- | --- | --- |
| x | `[tokens, 4096]`, layer output | bfloat16 |
| residual | `[tokens, 4, 4096]`, incoming streams | bfloat16 |
| post_mix | `[tokens, 4]`, incoming post gates | float32 |
| comb_mix | `[tokens, 4, 4]`, incoming combination matrices | float32 |
| proj_weight | `[24, 16384]` | float32 |
| mix_scale | `[3]`, one scale per mix group | float32 |
| mix_bias | `[24]` | float32 |
| rms_eps, pre_eps, sinkhorn_eps, norm_eps | `1e-6` | Python float |
| post_multiplier | `2.0` | Python float |
| sinkhorn_iters | `20` | Python int |
| norm_weight | `[4096]` | bfloat16 |
| next_post_mix (out) | `[tokens, 4, 1]` | float32 |
| next_comb_mix (out) | `[tokens, 4, 4]` | float32 |
| layer_input (out) | `[tokens, 4096]` | bfloat16 |
| next_residual (out) | `[tokens, 4, 4096]` | bfloat16 |

Outputs are returned on the input GPU.

The baseline entry is `aiter.ops.mhc.mhc_fused_post_pre`, with materialized
source at `aiter_source/aiter/ops/mhc.py`. It selects a fused HIP kernel or the
split `mhc_post` + `mhc_pre` path from the token count and device.

`task_initialize.py` draws the residual streams, layer output and mix biases
from normal distributions, the incoming post gates uniformly from `[0.1, 0.9]`,
the incoming combination matrices as softmax rows refined by 20 Sinkhorn
iterations, the projection weights at fan-in scale, unit mix scales, and the
norm weight uniformly from `[0.5, 1.5]`. Each case re-seeds the initializer
independently. `task_reference.py` computes the operator in FP32 with the BF16
residual boundary above. `task_compare.py` requires **every element of every
output** to satisfy `|actual - expected| <= 0.01 + 0.01 * |expected|`, with
matching output names, shapes, dtypes, device and finite values. The baseline
and candidate both require this full gate, including timed outputs.

## Task contract

`config.yaml` is the only task configuration. `evaluation.workloads` locates the
protected case data; `candidate.entrypoints` declares the candidate file and the
exact builder symbol. The symbol is not derived from the operator identity.
The initial candidate is an empty module. The agent must implement the operator
in FlyDSL in the declared editable file. All other task files are protected.

The builder receives shape arguments only, once per case, and returns a callable
launch. Prepare compilation, shape-dependent tile choices and reusable scratch
in the builder. Every launch must compute the complete operator on the supplied
current tensor contents; it must not cache answers or alter input tensors.
State kept across launches may be derived only from the layer parameters
(`proj_weight`, `mix_scale`, `mix_bias`, `norm_weight`; for example a one-time
re-layout keyed on the weight tensor); it must never be derived from `x`,
`residual`, `post_mix`, `comb_mix` or outputs. Recognizing inputs seen before
and returning a stored or partial result games the measurement; it is not an
optimization. All 13 variable-axis sizes (1 through 4096, powers of two) are
scored. Tiling, fusion, split reductions and per-shape dispatch are
implementation choices.

## Baseline, reference and dependencies

The production baseline calls the **installed AITER package**. The framework
materializes the image's `/sgl-workspace/aiter/aiter` package subtree at
`aiter_source/aiter`, preserving every declared `baseline.source_files` path.
This is read-only explanatory code, not a standalone AITER checkout or an
installation/build input; it must not shadow the installed Python package.
Acquisition excludes only `jit/build`, `jit/flydsl_cache` and the package-root
`__pycache__` from this copy. Baseline callbacks execute the installed package,
whose complete build dependencies and runtime artifacts remain in the pinned
Docker image. The selected Docker runtime must provide AITER, FlyDSL and ROCm
PyTorch on gfx950. The runner records runtime versions, the executed baseline
module's source hash and its entry point. The original initialize, reference,
compare and baseline callbacks are included unchanged under `scripts/`.

Candidate computation must be implemented in FlyDSL. Allowed imports are FlyDSL,
PyTorch for tensor allocation/views/dtypes/launch plumbing, and these host-only
Python modules: `__future__`, `typing`, `collections`, `dataclasses`, `enum`,
`functools`, `itertools`, `math`, `operator`, `numbers`, `abc`, `types`.
Calling a library implementation of the operator or its parts, including
PyTorch matrix products, softmax, sigmoid, rsqrt or normalization, is
forbidden. Importing AITER, protected task modules, other local modules or
another GPU computation library is forbidden. Loading code or task files
dynamically is also forbidden. The AST guard checks direct imports, imported
members/aliases, common matrix-product forms and torch-rooted references to
these operations. It is a guard against ordinary violations, not a proof
against arbitrary Python reflection or tensor-method calls. Numerical and
timed-path checks and evaluator review also remain required.

## Evaluation

From a materialized task workspace, the public CLI supports:

```bash
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Each command emits exactly one `ARENA_EVAL_RESULT=` JSON line and exits nonzero
on failure. All case rows retain the same ID, shape, dtype and semantic params,
including the scalar values. `validate-task` enumerates the complete protected
manifest before executing any candidate. It checks dependencies, inputs and
reference validity; in framework `task_validation` phase it verifies the actual
initial stub and reports `metadata.candidate_state`. Compilation executes each
specialization, including lazy JIT compilation on its first launch.
Correctness uses the task's original comparison, independently for the
requested role, over every case. An absent candidate, missing builder or
`NotImplementedError` always fails candidate actions, in both phases; only the
framework can defer candidate checks for an initially empty task.

Both roles use the materialized canonical `_aka_benchmark.py` helper with the
unchanged workload warmup, repetition and target duration, over every case.
Preparation and input allocation occur outside timing; the timed callable is
the complete operator invocation, including its device work and output/scratch
allocation. CUDA-graph timing is used; event fallback is recorded and the
framework checks that baseline and candidate timing methods match. Each sample
times one logical invocation; calls are never batched into one capture. Before
each sample, outside timing, the call-varying operands (`x`, `residual`,
`post_mix`, `comb_mix`) are overwritten in place with one of several draws from
the bundle's initializer, while the layer parameters stay fixed. Draw seeds
come from the operating system when the case is timed. After the samples, the
timed unit runs once over each of several further draws it has never read,
timed like a sample, and the fastest of those may take at most
`UNSEEN_DRAW_MARGIN` (in `scripts/task_measure.py`) times the reported mean.
The outputs of randomly chosen reported samples and of every unseen-draw
invocation are compared, with the original comparator, against the reference
on the draw each one consumed, and the parameters and loaded operands must be
unchanged afterwards. Input mutation, nonfinite output, missing work or runtime
failures fail the case. No timing from an instrumented sanitizer build may
become an official score.

A task does not require any agent-specific driver or environment variable.
Forge and other adapters invoke these same commands; their private search
protocols do not change the task's comparison or measurement policy.

## Export

The common Arena post-processing stage invokes `exports[].command` after final
acceptance. `scripts/export_solution.py` reads the framework-owned
`task_result.yaml`, the declared candidate and workload paths, and the protected
`solution.json` template; it writes only the declared artifact output. It
requires compilation, correctness, tool policy and complete comparable timings.
It rejects failed/stub candidates and never computes or writes Arena scores.
The artifact includes the candidate and a tensor-call wrapper whose `run`
signature is the definition's input order; it binds the declared builder.
There is no external publication or sync. Consumers must support the
artifact's `flydsl` language; exporting alone does not prove compatibility or
acceptance by a separate SIKL installation.

## Validation controls

`validate-task` executes every workload case's input generator and reference,
checking each declared output's shape, dtype, device and finiteness. In
addition it executes an independent small known-answer fixture and comparator
positive/negative controls once per task. They appear in
`metadata.validation_controls`, with observed values and device, and do not
add or remove scored cases. A failed control fails task validation.

The fixture uses two tokens, four streams and hidden size 4 with exactly
representable operands and the workload's scalar values. Token 0 combines
streams through a permutation and token 1 through an average of two
permutations, so a missing transpose of `comb_mix` changes `next_residual`.
Projection weights, mix scales and biases differ per mix, so a wrong scale
group, bias offset, reshape or softmax axis changes the mixes. Its expected
outputs come from scalar float64 Python arithmetic written from the formulas
above, not from the reference. `next_residual` must match exactly; the FP32
mixes within `1e-5` relative; the BF16 `layer_input` within one BF16 rounding
step. The comparator must accept a nonzero error within its gate (1.015625
versus 1 in every output) and reject a larger one (1.03125 versus 1) in each
output separately, plus wrong signs, shape/dtype mismatches, nonfinite output
and a missing output.

These controls exercise specific oracle/comparator properties; they are not
proof of every reference operation for every possible input. They do not
replace complete workload evaluation, an actual accepted candidate, or GPU
validation. The fixture's rounding bounds do not alter the candidate tolerance
or scored workload data.
