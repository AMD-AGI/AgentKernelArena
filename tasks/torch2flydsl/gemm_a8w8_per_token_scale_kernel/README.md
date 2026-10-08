# gemm_a8w8_per_token_scale_kernel: task-owned v2 contract

Implement or optimize gemm_a8w8_per_token_scale in FlyDSL, preserving all task inputs, outputs and numerical gates.

The candidate starts **unimplemented**. The runner explicitly selects the provided baseline before loading any candidate source. An empty candidate is allowed only at initial task validation.
The original harness's primary implementation timing is retained; additional
reference/operator timings are diagnostic only. `test_kernel_harness.py` defines
the exact dispatch and allocation boundary for this task. The provided path uses
the task-local PyTorch model or installed AITER operator specified there, with its
fixed paired Event timing policy described below. `model.py` is protected reference/source material;
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

## Paired timing policy

Baseline and candidate both use the task's predetermined Event timing method.
The provided baseline already selects this method before attempting capture;
an implemented candidate must use the same method even if it supports Graph
capture. Selecting Event for the baseline and Graph for the candidate makes
their timings incomparable and prevents Arena from scoring a correct candidate.
This repair preserves the baseline method, operator calls, allocation
boundaries, declared shapes, numerical gates, 10 warmups and 100 timed samples.
It does not accept a runtime capture failure as permission to switch methods.
Diagnostic reference timing uses the same fixed method.

Sample zero uses the original seeded BF16 operands. The remaining 99 measured
samples use distinct, deterministic BF16 operands with the same shapes and
input distribution for both baseline and candidate. Preparation is outside
Event timing. After each Event ends, the harness reads the live operands and
copies that sample's complete output for an independent quantized-reference
check; the final output is also poisoned and the exact timed callable rerun
with changed inputs. Oracle checks and output copies are outside timing. The
diagnostic reference timing receives the same input stream. Since the timed
input sequence changed, historical latencies and speedups are not directly
comparable; the changed task packages require fresh GPU qualification.

The output must be a finite BF16 tensor of shape `[M,N]` on the input device;
raw A and weight tensors are read-only. Both the actual measured output and a
poisoned-output replay with changed activation/weight values must pass the
original quantized `Model` reference and normalized max-error gate. The separate
unquantized PyTorch GEMM remains only a diagnostic performance comparison.
Original five cases, quantization/reference algorithms, seeds, tolerances,
zero-reference denominator, timing policy, warmups and samples are unchanged.
Candidate-only auditing requires actual FlyDSL computation, with host allocation,
layout/casts and launch preparation allowed. Baseline library dispatch and final
candidate are checked separately; a starter is not a final implementation.
