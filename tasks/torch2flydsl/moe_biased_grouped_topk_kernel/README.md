# moe_biased_grouped_topk_kernel: task-owned v2 contract

Implement or optimize moe_biased_grouped_topk in FlyDSL, preserving all task inputs, outputs and numerical gates.

The candidate starts **unimplemented**. The runner explicitly selects the provided baseline before loading any candidate source. An empty candidate is allowed only at initial task validation.
The original harness's primary implementation timing is retained; additional
reference/operator timings are diagnostic only. `test_kernel_harness.py` defines
the exact dispatch and allocation boundary for this task. The provided path uses
the task-local PyTorch model or installed AITER operator specified there, with its
original graph/event policy. `model.py` is protected reference/source material;
its presence alone does not select the performance baseline.

There are 3 declared cases in `cases.json`. All original dimensions,
parameter variants, seeds, tolerances, numerical metrics, output checks, warmups
and repetition counts remain in the protected harness. It validates the PyTorch
reference against the original independent AITER comparison wherever that check
was present. Small independent known-answer and negative-output controls in
`scripts/reference_controls.py` supplement the full GPU checks.

Edit only `candidate.editable` paths from `config.yaml`. Preserve each declared
public operator/builder interface and all outputs (including residuals, packed
quantization codes/scales, routing indices or state when applicable). Inspect the
protected harness calls and `model.py` to understand shapes, strides and layout.
The final operator computation must run FlyDSL GPU kernels. PyTorch is allowed
for allocation, views and launch preparation, not replacement operator compute.
Do not import the model, harness, reference or baseline from candidate code.
No Triton, AITER operator calls, external kernels, dynamic module loading, native
launch bypasses or subprocess dispatch are allowed as the final computation.
Bundled implementation utilities remain protected unless config explicitly lists
them as editable. Candidate absence, a stub or a None output is a final failure.

Use the public task-local runner from the materialized workspace:

```sh
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Compile syntax-checks the actual role's Python sources; correctness then exercises
real case-specific GPU compilation and execution. Every action emits
`ARENA_EVAL_RESULT=` with an `arena-eval-v1` JSON envelope. Arena owns final scores
and reports. Agents do not write `task_result.yaml`.

The runtime image supplies ROCm, PyTorch, FlyDSL and required AITER operators. Arena
must materialize the canonical `_aka_benchmark.py` helper. CPU controls do not
establish GPU correctness or timing support. Historical validation files predate
this migration; the parent integration schedules fresh GPU validation.


The primary provided baseline is installed AITER `biased_grouped_topk_hip`;
`model.py` is the independent PyTorch routing oracle. All three DeepSeek-V3
cases use FP32 logits/bias, eight selected experts from the four selected groups,
and routed scale2.5. Return FP32 weights and INT32 unique in-range expert IDs,
both shaped `[tokens, topk]` on the input device. Preserve logits and bias.
The original genuine-boundary-tie allowance1e-4 and matched-expert normalized
weight-error bound1e-2 apply unchanged to correctness and actual timed replay.

Both roles retain the original10 external warmups and100 CUDA-graph samples,
including their original output allocation boundary. The harness validates the
output captured by the timing helper, then rolls logits and bias, poisons both
weights and IDs, and replays that same graph against the original reference.
Setup/comparison/input restoration happen outside timing. Replay metadata names
`captured_graph`; a capture failure remains a failure. Reference timing is a
separate diagnostic and never substitutes for primary measured outputs.
Candidate-only dispatch/profile checks require actual FlyDSL calls and reject
PyTorch routing computations, without instrumenting any timed invocation.

## Local implementation helpers

Declared editable candidate sources may import additional task-local Python
device/compiler helpers, including packages. Dependency review and syntax
compilation follow the complete normal local-import closure. The same declared
implementation restrictions apply to every helper: protected references,
alternative final-language backends and dynamic implementation loading remain
disallowed. Source symlinks, imports escaping the task and local modules shadowing
the supported runtime libraries are rejected. Installed library source is not
traversed. The candidate-only PyTorch preparation and FlyDSL launch audits apply
to these helpers too; the performance policy states how timed calls are checked. These checks enforce the task contract;
they are not a Python security sandbox.

## Candidate execution policy during performance

The candidate-only PyTorch preparation policy remains active on every actual
operator call made by canonical benchmark warmup, graph capture and eager event
timing. Tensor arithmetic, reductions and library operators must execute as
FlyDSL kernels even when candidate behavior changes after earlier calls. A
probe of the actual performance closure observes FlyDSL invocation outside
clocks and capture; Python profiling is disabled inside canonical measurement.
Allocation and view preparation retain the existing allowlist. Baseline and
reference operators keep their original policy and all original device timing
parameters. One additional untimed audit probe of each performance closure runs
before canonical benchmarking; stateful closures use the existing prepare
callback before and after that probe. This probe is separate from the retained
benchmark warmup/sample counts. A forbidden operation remains a failure even if
a capture fallback or candidate code catches its exception. These checks are
implementation-contract evidence, not a Python
security sandbox.

Candidate sources and their normal local-import helpers must not access the
evaluator's module registry, stack frames or profiling controls through `sys`,
including aliases and dynamic attribute lookup. Ordinary `sys` metadata remains
available, and installed FlyDSL library internals are not scanned as candidate
code. Protected reference imports remain prohibited.

## Protected evaluator state

Candidate sources may not import the materialized timing helper `_aka_benchmark`,
assign or delete attributes of imported modules, read frame or function globals,
or disable the evaluator's PyTorch dispatch mode. When the local import closure
is resolved, a same-name `.py` module takes precedence over a namespace-package
directory, matching Python. These structural checks are not a security sandbox.
