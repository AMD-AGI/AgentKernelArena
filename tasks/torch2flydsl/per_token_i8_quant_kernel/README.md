# per_token_i8_quant_kernel: task-owned v2 contract

Implement or optimize per_token_i8_quant in FlyDSL, preserving all task inputs, outputs and numerical gates.

The candidate starts **unimplemented**. The runner explicitly selects the provided baseline before loading any candidate source. An empty candidate is allowed only at initial task validation.
The original harness's primary implementation timing is retained; additional
reference/operator timings are diagnostic only. `test_kernel_harness.py` defines
the exact dispatch and allocation boundary for this task. The provided path uses
the task-local PyTorch model or installed AITER operator specified there, with its
original graph/event policy. `model.py` is protected reference/source material;
its presence alone does not select the performance baseline.

There are 5 declared cases in `cases.json`. All original dimensions,
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

The public operator returns `(codes, scale)`; the unused starter builder is not
an evaluation entrypoint. Codes have the input shape and the original quantized
dtype; scales keep the original per-tensor/per-token/per-group FP32 layout.
Both outputs are finite and on the input device; input bytes are read-only.
Measured outputs and both outputs of a poisoned replay with changed input must
pass the original code-distance and scale-relative-error comparison against
AITER. All original cases, seeds, rounding/quantization algorithms, tolerances,
graph timing and sample counts are unchanged. The provided PyTorch model remains
the primary baseline. Candidate computation must use actual FlyDSL launches;
allocation, layout/casts and launch preparation are allowed host operations.


## Local implementation helpers

`kernel.py` may import additional task-local Python device/compiler helpers,
including packages. Candidate dependency review and syntax compilation follow
the complete normal local-import closure. The same implementation restrictions
apply to every helper: protected references, alternative backends and dynamic
implementation loading remain disallowed. Source symlinks, imports escaping the
task and local modules shadowing the supported runtime libraries are rejected.
Installed library source is not traversed. Existing candidate-only PyTorch
preparation and per-call FlyDSL launch audits remain active outside timing.
These checks enforce the task contract; they are not a Python security sandbox.


## Zero-row correctness control

An additional unscored BF16 `3 x 4096` all-zero input checks the provided
model and actual candidate against independently constructed INT8 zero codes
and FP32 scale ones. Candidate calls remain inside the existing execution audit;
inputs are read-only and restored on failure. The original five scored random
cases, AITER comparisons, code tolerance of one, scale tolerance of 1e-3, seeds
and timing are unchanged. The zero control does not assume AITER zero-row
behavior and introduces no additional timed case.

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
