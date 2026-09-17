# rmsnorm2d_dynamicquant_kernel: task-owned v2 contract

Implement or optimize rmsnorm2d_dynamicquant in FlyDSL, preserving all task inputs, outputs and numerical gates.

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


Only the entrypoints listed in config.yaml are required interfaces. The primary
operator is `flydsl_rmsnorm2d_dynamicquant`; any additional declared callables used by the harness
remain required. Legacy build/compile helpers are optional implementation details;
no builder return protocol is required by this task. A candidate may choose its
own internal compilation helpers, while implementing all tested work in FlyDSL.


The five original BF16 RMSNorm/dynamic-quantization workloads and seed20260401
remain. Require exactly (FP8 codes[m,n], FP32 scales[m,1]) on the input device,
with finite values and read-only input/gamma tensors. The hardware-selected
FP8 dtype must match the model/AITER contract. Preserve the existing gate:
all code bytes differ by at most1; maximum scale error divided by the original
maximum reference scale plus1e-12 is at most1e-3. Exact-code percentage remains
a diagnostic, not a new threshold.
The model's FP32 RMS reduction/weight multiplication and direct quantization
are unchanged. The independent AITER operator remains the numerical oracle.
Both diagnostic timings and the actual role timing use the same canonical
collector with the original10warmups/100samples. Check each measured tuple,
then negate/halve the BF16 gamma outside timing: codes change sign and the
expected scales halve. Poison codes with the format's NaN byte and scales with
NaN before replaying the exact measured invocation; no fresh untimed candidate
call may stand in for that result. Inputs are restored before later timings.
The final candidate's operator arithmetic must launch FlyDSL and may not call
AITER/Triton/PyTorch computation or protected model/reference code; host storage
preparation remains permitted and is audited outside timing.
