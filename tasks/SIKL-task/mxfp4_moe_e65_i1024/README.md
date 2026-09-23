# MXFP4 routed MoE to FlyDSL

Implement the routed-expert layer with MXFP4 blocks of 32, SiLU activation and
BF16 input/output. The protected workload axes define the actual dimensions;
`I = w1_rows / 2`, `D = model_dim`, `E = num_experts`. Each case varies
`num_tokens`. Packed weights are already quantized and preshuffled. The
reference defines activation quantization, both expert projections, SiLU/gating,
routing weights and final expert reduction.

The configured builder signature is
`builder(num_tokens, model_dim, inter_dim, num_experts, topk) -> launch`, invoked
with keyword arguments. Launch takes the following positional arguments:

```python
launch(hidden_states, w1, w2, topk_weights, topk_ids,
       w1_scale, w2_scale, activation, doweight_stage1) -> out
```

| Argument | Shape / value | Storage dtype |
| --- | --- | --- |
| hidden_states | `[num_tokens, D]` | bfloat16 |
| w1 | `[E, 2*I, D/2]`, preshuffled gate/up | float4_e2m1fn_x2 |
| w2 | `[E, D, I/2]`, preshuffled | float4_e2m1fn_x2 |
| w1_scale | `[E, 2*I, D/32]`, preshuffled E8M0 bits | uint8 |
| w2_scale | `[E, D, I/32]`, preshuffled E8M0 bits | uint8 |
| topk_weights | `[num_tokens, topk]` | float32 |
| topk_ids | `[num_tokens, topk]` | int32 |
| activation | `0` (SiLU) | Python int |
| doweight_stage1 | `False`; weights applied during stage 2 reduction | Python bool |
| out | `[num_tokens, D]`, returned on the input GPU | bfloat16 |

The baseline entry is `aiter.fused_moe.fused_moe`, with explanatory source at
`aiter_source/aiter/fused_moe.py`; dispatch wrappers select stage implementations
from the runtime's tuned configuration. `AITER_LOG_TUNED_CONFIG` is enabled by
the runner so its actual selection is retained in the command logs.

`task_initialize.py` owns quantization and input construction: activations are
standard normal; expert weights are drawn at logical fan-in scale and quantized
to nearest-even E2M1 under upward-rounded E8M0 scales, followed by the expert
shuffle. Random packed bytes are not equivalent inputs. Each case re-seeds the
initializer independently, so it cannot share another case's generated weights.
`task_compare.py` requires output-dtype **SQNR >= 13 dB**, as well as matching
shape, dtype, device and finite values. Its exact-match infinite SQNR is a valid
comparison result, never permission to emit nonfinite JSON numbers. The
baseline and candidate both require the full numerical gate, including replay.

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
State kept across launches may be derived only from the weights (for example a
one-time re-layout keyed on the weight tensor); it must never be derived from
activations, routing or outputs. Recognizing inputs seen before and returning a
stored or partial result games the measurement; it is not an optimization.
All 13 variable-axis sizes (1 through 4096, powers of two) are scored. Tiling,
fusion, split reductions and per-shape dispatch are implementation choices.

## Baseline, reference and dependencies

The production baseline calls the **installed AITER package**. The framework
materializes the image's `/sgl-workspace/aiter/aiter` package subtree at
`aiter_source/aiter`, preserving every declared `baseline.source_files` path.
This is read-only explanatory code, not a standalone AITER checkout or an
installation/build input; it must not shadow the installed Python package.
All package Python sources, including `jit/__init__.py`, `jit/core.py`, operator
implementations and configuration data, are retained. Acquisition excludes only
`jit/build`, `jit/flydsl_cache` and the package-root `__pycache__` from this copy.
Repository-level third-party CK headers, examples and packaging files outside
this package are not task inputs. Baseline callbacks execute the installed
package, whose complete repository/build dependencies and runtime artifacts
remain in the pinned Docker image. Task-local references and inputs are bundled
under `scripts/`; they do not import or build the explanatory snapshot.
The selected Docker runtime must provide AITER, FlyDSL and ROCm PyTorch on gfx950.
The runner records runtime versions, the executed baseline module's source hash
and dispatch evidence. No SIKL repository clone is needed: the original
initialize, reference, compare and baseline callbacks are included under
`scripts/`. These callbacks are unchanged by the Arena v2 migration.

Candidate computation must be implemented in FlyDSL. Allowed imports are FlyDSL,
PyTorch for tensor allocation/views/dtypes/launch plumbing, and these host-only
Python modules: `__future__`, `typing`, `collections`, `dataclasses`, `enum`,
`functools`, `itertools`, `math`, `operator`, `numbers`, `abc`, `types`.
Calling a library implementation of the operator, including PyTorch matrix
products, is forbidden. Importing AITER, protected task modules, other local
modules or another GPU computation library is forbidden. Loading code or task
files dynamically is also forbidden. The AST guard checks direct imports,
imported members/aliases and common matrix-product forms. It is a guard against
ordinary violations, not a proof against arbitrary Python reflection. Numerical
and timed-path checks and evaluator review also remain required.

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
preferred; event fallback is recorded and the framework checks that baseline
and candidate timing methods match. Each sample times one logical invocation;
calls are never batched into one capture. Before each sample, outside timing,
the call-varying operands (`hidden_states`, `topk_weights`, `topk_ids`) are
overwritten in place with one of several draws from the bundle's initializer,
while the expert weights and scales stay fixed. Draw seeds come from the operating system when
the case is timed. After the samples, the timed unit runs once over each of
several further draws it has never read, timed like a sample, and the fastest
of those may take at most `UNSEEN_DRAW_MARGIN` (in `scripts/task_measure.py`)
times the reported mean. The outputs of randomly chosen reported samples and
of every unseen-draw invocation are compared, with the original comparator,
against the reference on the draw each one consumed, and the weights and loaded
operands must be unchanged afterwards. Input mutation, nonfinite output, missing
work or runtime failures cannot be treated as numerical diagnostics.
No timing from an instrumented sanitizer build may become an official score.

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
The artifact includes the candidate and a tensor-call wrapper for its declared
builder; it does not require consumers to derive a builder name from task ID.
The original template slot name is retained for provenance; the artifact does
not imply that Forge produced it. There is no external publication or sync.
Consumers must support the artifact's `flydsl` language; exporting alone does
not prove compatibility or acceptance by a separate SIKL installation.

## Validation controls

`validate-task` executes every original workload case's input generator and
reference, checking the declared output shape, BF16 dtype, device and finiteness.
In addition it executes independent small known-answer controls and comparator
positive/negative controls once per task. They appear in
`metadata.validation_controls`, with observed values and device, and do not add
or remove scored cases. A failed control fails task validation and cannot be
covered by baseline diagnostics.

For GEMM, an asymmetric 2x3 and 4x3 integer fixture has the literal product
`[[1, 3, 0, 4], [3, 0, -1, -4]]`. Its reference output is checked exactly,
independently of the task comparator. The comparator must accept a nonzero BF16
error within its existing gate (1.0078125 versus 1) and reject a larger one
(1.03125 versus 1), wrong signs, shape/dtype mismatches and nonfinite output.

For MoE, controls check the literal E2M1 codebook, E8M0 powers, nearest-even
midpoint codes, scalar-address weight unshuffling and the gate/up formula using
Python scalar `exp`. A three-expert quantized fixture routes two experts and
leaves one unselected. With D=I=256, sparse activations 1 and 2, expert W1 values
1 and -1/2, and W2 values 1/4 and -1/2, quantized intermediate values are
(.75, .09375) and (4, .25). Routing weights (.25, .75) yield final output rows
filled with 3 and 40; these are hand-derived, not generated by the reference or
initializer. Constant per-expert weights/scales make this fixture independent
of the layout shuffle, while a separate nonconstant layout control checks that
step. The comparator must accept 1.125 versus 1 (about 18 dB SQNR) and reject
1.5 versus 1 (about 6 dB), plus contract violations. Positive controls prevent
an always-reject comparator from being mistaken for a working guard.

These controls exercise specific oracle/comparator properties; they are not
proof of every reference operation for every possible input. They do not
replace complete workload evaluation, an actual accepted candidate, or GPU
validation. The fixture checks' exact/FP32 rounding bounds do not alter the
original candidate tolerances or scored workload data.
