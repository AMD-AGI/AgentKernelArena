# Blockwise-scaled FP8 GEMM to FlyDSL

Implement `out = dequant(a) @ dequant(b).T` with 128x128 block scaling and a
BF16 output. `a` is `[m, k]` and `b` is `[n, k]`, both E4M3FN; the fixed `n`,
`k`, the scale extents `sn = ceil(n / 128)` and `sk = ceil(k / 128)`, the seed
and all variable `m` cases are in the declared workload JSON:

```text
out[i, j] = sum_t a[i, t] * a_scale[i, t // 128] * b[j, t] * b_scale[j // 128, t // 128]
```

**Stored layouts.** `b` is stored in AITER's 16x16 weight shuffle
(`shuffle_weight(layout=(16, 16))`): logical element `(row, col)` is at
physical element `((((row // 16) * (k // 32) + col // 32) * 2 + (col % 32) // 16) * 16 + row % 16) * 16 + col % 16`
of the row-major `[n, k]` buffer. `b_scale` is the plain `[sn, sk]` scale
table. `a_scale` is a `[m, sk]` FP32 buffer whose meaning is the workload's
`a_scale_storage`:

- `raw`: the buffer holds the column-major payload, so its flat element
  `t * m + i` is the scale of row `i`, block `t`;
- `logical`: the buffer holds the logical row-major scales.

The configured builder signature is `builder(m, n, k) -> launch`, invoked with
keyword arguments. Its launch signature is `launch(a, b, a_scale, b_scale) -> out`,
returning a `[m, n]` BF16 tensor on the same GPU.

| Argument | Shape | Dtype |
| --- | --- | --- |
| a | `[m, k]` | float8_e4m3fn |
| b | `[n, k]`, 16x16 shuffled | float8_e4m3fn |
| a_scale | `[m, sk]`, stored as declared | float32 |
| b_scale | `[sn, sk]` | float32 |
| out | `[m, n]` | bfloat16 |

The baseline entry is `aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle`,
with materialized source at `aiter_source/aiter/ops/gemm_op_a8w8.py`. It selects
a kernel per `(m, n, k)` from AITER's tuned blockscale bpreshuffle table; the
runner records the selected row per case. With `logical` storage the baseline
first materializes the column-major scale copy its kernels read, inside the
timed call, as the bundle's baseline does. The bundle's note about an isolated
process with SIKL proxy settings concerns the SIKL harness's own proxy, which
is not part of this runtime; every Arena action already runs in its own
process.

`task_initialize.py` draws `a` and `b` from a standard normal distribution
converted to E4M3FN and both scale tables uniformly from `[0.125, 1]`, then
applies the weight shuffle and the declared scale storage. Each case re-seeds
the initializer independently. `task_reference.py` decodes the stored layouts,
dequantizes per block and computes the product in FP32, returning BF16.
`task_compare.py` requires **every element** to meet its absolute or relative
tolerance (BF16: 0.01 absolute **or** 0.01 relative), with matching shape,
dtype, device and finite values. Candidate and timed candidate outputs must
meet that same rule.

The baseline policy is explicit in this task's `config.yaml`. Tasks with
`correctness_policy: required` must pass baseline correctness, while
`diagnostic` retains only the documented numerical exception, with its
evidence below. Crashes, shape/dtype errors and nonfinite outputs never
qualify. Candidate correctness and timed candidate outputs have no diagnostic
exception.

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
State kept across launches may be derived only from the weights `b` and
`b_scale` (for example a one-time re-layout keyed on the weight tensor); it must
never be derived from `a`, `a_scale` or outputs. Recognizing inputs seen before
and returning a stored or partial result games the measurement; it is not an
optimization. All 13 variable-axis sizes (1 through 4096, powers of two) are
scored. Tiling, split reductions and per-shape dispatch are implementation
choices.

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
module's source hash and dispatch evidence. The original initialize, reference,
compare and baseline callbacks are included unchanged under `scripts/`.

Candidate computation must be implemented in FlyDSL. Allowed imports are FlyDSL,
PyTorch for tensor allocation/views/dtypes/launch plumbing, and these host-only
Python modules: `__future__`, `typing`, `collections`, `dataclasses`, `enum`,
`functools`, `itertools`, `math`, `operator`, `numbers`, `abc`, `types`.
Calling a library implementation of the operator, including PyTorch matrix
products and `_scaled_mm`, is forbidden. Importing AITER, protected task
modules, other local modules or another GPU computation library is forbidden.
Loading code or task files dynamically is also forbidden. The AST guard checks
direct imports, imported members/aliases and common matrix-product forms. It is
a guard against ordinary violations, not a proof against arbitrary Python
reflection. Numerical and timed-path checks and evaluator review also remain
required.

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
on failure. All case rows retain the same ID, shape, dtype and semantic params.
`validate-task` enumerates the complete protected manifest before executing any
candidate. It checks dependencies, inputs and reference validity; in framework
`task_validation` phase it verifies the actual initial stub and reports
`metadata.candidate_state`. Compilation executes each specialization, including
lazy JIT compilation on its first launch. Correctness uses the task's original
comparison, independently for the requested role. An absent candidate, missing
builder or `NotImplementedError` always fails candidate actions, in both phases;
only the framework can defer candidate checks for an initially empty task.

Both roles use the materialized canonical `_aka_benchmark.py` helper with the
unchanged workload warmup, repetition and target duration. Preparation and input
allocation occur outside timing; the timed callable is the complete operator
invocation, including its device work and output/scratch use. Graph timing is
used; event fallback is recorded and the framework checks that baseline and
candidate timing methods match. Each sample times one logical invocation;
calls are never batched into one capture. Before each sample, outside timing,
the call-varying operands `a` and `a_scale` are overwritten in place with one
of several draws from the bundle's initializer, while the weights `b` and
`b_scale` stay fixed. Draw seeds come from the operating system when the case
is timed. After the samples, the timed unit runs once over each of several
further draws it has never read, timed like a sample, and the fastest of those
may take at most `UNSEEN_DRAW_MARGIN` (in `scripts/task_measure.py`) times the
reported mean. The outputs of randomly chosen reported samples and of every
unseen-draw invocation are compared, with the original comparator, against the
reference on the draw each one consumed, and the weights and loaded operands
must be unchanged afterwards. Input mutation, nonfinite output, missing work or
runtime failures cannot be treated as numerical diagnostics. No timing from an
instrumented sanitizer build may become an official score.

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
The artifact includes the candidate and a tensor-call wrapper with the
baseline's signature, binding the declared builder per shape. There is no
external publication or sync. Consumers must support the artifact's `flydsl`
language; exporting alone does not prove compatibility or acceptance by a
separate SIKL installation.

## Validation controls

`validate-task` executes every workload case's input generator and reference,
checking the declared output shape, BF16 dtype, device and finiteness. In
addition it executes an independent known-answer fixture and comparator
positive/negative controls once per task. They appear in
`metadata.validation_controls` and do not add or remove scored cases. A failed
control fails task validation and cannot be covered by baseline diagnostics.

The fixture has `m = 2`, `n = 16` and `k = 256` (two scale blocks), integer
E4M3 operands and power-of-two scales, so every partial sum is exact in FP32.
Its weight is placed by scalar shuffle-position arithmetic and its activation
scales are stored as declared, and the reference output must equal the scalar
block-scaled products rounded to BF16 exactly. The comparator must accept a
nonzero BF16 error within its gate (1.0078125 versus 1) and reject a larger one
(1.03125 versus 1), wrong signs, shape/dtype mismatches and nonfinite output.

These controls exercise specific oracle/comparator properties; they are not
proof of every reference operation for every possible input. They do not
replace complete workload evaluation, an actual accepted candidate, or GPU
validation.

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
