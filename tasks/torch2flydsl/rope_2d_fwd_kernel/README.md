# rope_2d_fwd_kernel: task-owned v2 contract

Implement rope_2d_fwd in FlyDSL, preserving the input/output interface and satisfying the declared numerical gate.

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

## Correctness and measured-output repair

The comparator now enforces both originally documented bounds: normalized
maximum error at most `0.01` **and** at least `99.9%` of elements passing
`atol=rtol=0.01`. The old code used OR despite its AND specification. One huge
finite error among 2,000 otherwise exact outputs could therefore pass solely
on percentage. This is an explicit stricter acceptance policy; previous
candidate acceptance does not establish passage of this repaired gate.
Task validation includes that sparse-error negative control and the independent
90-degree height/width rotation known answer.

All outputs must retain the reference shape, BF16 dtype and device. The original
four workload cases, inputs, model, operator calls, 10 warmups and 100 samples
remain unchanged. Each actual measured output is checked, then poisoned and
replayed with a negated data input against a fresh AITER reference. Angle tables
and data inputs are read-only to the implementation, and all replay inputs are
restored before the next timing. Additional checks run outside device timing
for the provided baseline, operator diagnostic and candidate alike.

Candidate-only launch checks reject AITER/PyTorch operator delegation; a
baseline/reference launch cannot count as candidate FlyDSL execution. The
public submission interface is `flydsl_rope_2d_fwd`; the harness does not call a
builder, so it no longer requires an otherwise unused builder definition.
The stricter gate and added replay require fresh GPU task validation.
