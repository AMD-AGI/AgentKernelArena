# fmoe_g1u1_tkw1_kernel: task-owned v2 contract

Implement or optimize fmoe_g1u1_tkw1 in FlyDSL, preserving all task inputs, outputs and numerical gates.

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

Timing begins from the same raw BF16 hidden/weight tensors and selected routing
for both roles. The AITER baseline now quantizes and shuffles weights (and, for
the block-scaled task, quantizes activations and prepares sorted routing) inside
each measured call, matching the candidate's input boundary. Router logits,
top-k selection and the diagnostic reference's expert plan stay outside timing
as before. This corrects the older baseline-only preprocessing exclusion; old
kernel-only timings cannot be compared as if the timed work were unchanged.
The original three cases, seed, numerical gates, warmups and samples remain.

Outputs must be finite BF16 tensors with the hidden tensor's shape/device.
Hidden, raw weights and selected routing are read-only. The actual measured
output and input-perturbed replay use the original normalized max-error rule;
replay changes hidden and both weight tensors, then restores all inputs.
Candidate-only dependency/dispatch auditing requires FlyDSL computation while
allowing host launch preparation. GPU before/after timing-boundary qualification
is required; this maintenance correction is not an optimization speedup.

Correctness records each completed case separately, including normalized and
absolute baseline errors and the original tolerance. If one case fails, other
completed cases retain their actual PASS status. Only a failed numerical gate
receives `numerical_mismatch`; launch errors, invalid outputs and input mutation
remain execution/contract failures. Baseline correctness is still required;
these diagnostics do not grant a numerical exception or widen tolerances.
