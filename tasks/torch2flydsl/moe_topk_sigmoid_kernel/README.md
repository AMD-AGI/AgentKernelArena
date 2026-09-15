# moe_topk_sigmoid_kernel: task-owned v2 contract

Implement or optimize moe_topk_sigmoid in FlyDSL, preserving all task inputs, outputs and numerical gates.

The candidate starts **unimplemented**. The runner explicitly selects the provided baseline before loading any candidate source. An empty candidate is allowed only at initial task validation.
The original harness's primary implementation timing is retained; additional
reference/operator timings are diagnostic only. `test_kernel_harness.py` defines
the exact dispatch and allocation boundary for this task. The provided path uses
the task-local PyTorch model or installed AITER operator specified there, with its
original graph/event policy. `model.py` is protected reference/source material;
its presence alone does not select the performance baseline.

There are 4 declared cases in `cases.json`. All original dimensions,
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

Both the actual measured weights/expert IDs and the measured invocation's replay
are checked with the original tie-aware ID comparison and normalized maximum
weight-error gate. Shape, FP32 weights, INT32 IDs, device, finite values, valid
unique IDs and read-only inputs are enforced. Replay permutes expert identities,
recomputes the protected reference and poisons both outputs outside timing;
original inputs are restored afterward. The provided AITER baseline and final
candidate use the same controls and retain all original cases, seeds, warmups,
sample counts and timing-method selection. Graph evidence means captured replay;
explicit Event evidence means re-invocation of the same eager callable.

Candidate correctness includes a protected, candidate-only PyTorch dispatch
audit: allocations, initialization, copies and views may prepare a launch;
arithmetic, reductions, sorting and library operator calls must execute in
FlyDSL. AITER imports and direct native/subprocess dispatch are rejected in
final candidate source. Calling an unrelated FlyDSL kernel does not authorize
PyTorch operator computation. These checks supplement source review and the
numerical/replay gates; they are not a sandbox against arbitrary hostile Python.
FlyDSL launch evidence is required inside each candidate operator invocation;
launches by the reference or baseline cannot satisfy it.
The audit wraps candidate import/correctness only, leaving baseline/reference
execution and the original device-timing callable unchanged.
