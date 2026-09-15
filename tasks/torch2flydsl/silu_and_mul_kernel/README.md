# silu_and_mul_kernel: task-owned v2 contract

Implement or optimize silu_and_mul in FlyDSL, preserving all task inputs, outputs and numerical gates.

The candidate starts **unimplemented**. The runner explicitly selects the provided baseline before loading any candidate source. An empty candidate is allowed only at initial task validation.
The original harness's primary implementation timing is retained; additional
reference/operator timings are diagnostic only. `test_kernel_harness.py` defines
the exact dispatch and allocation boundary for this task. The provided path uses
the task-local PyTorch `Model`, with its original graph/event policy. The
installed AITER SiLU operator independently validates this baseline and has an
additional diagnostic timing; it is not the scored baseline for this task. `model.py` is protected reference/source material;
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

The scored invocation must also pass its task-specific numerical comparison. The
benchmark exposes the last measured output through the canonical `TimedRun`,
then checks that output before changing inputs. Outside all timed regions it
perturbs inputs in place, poisons the output, replays the measured unit, and
compares against the original numerical policy again. Read-only inputs must
remain unchanged by either execution. Output shape, dtype and device are part
of the contract. Unsupported replay collection fails; it is never a PASS/SKIP.

The public candidate API is `flydsl_silu_and_mul(input, limit)`. Only that
operator is declared in config: the unused starter `build_silu_and_mul_module`
name has no task-defined arguments/launch ABI and is not an additional public
contract. Candidates may organize their internal compilation helpers freely.
The starter file itself is unchanged.

All four original LIMIT=0 random cases and formal benchmark work remain intact.
Additional correctness probes on every original shape exercise positive limits
2.0, 2.03 and 7.0, gate upper clamping, two-sided up clamping and the documented
BF16 gate recast. Independent Python scalar known answers supplement the actual
AITER/model/candidate checks. The limit setting is restored before timing.
The original normalized error tolerance remains1e-2. Candidate execution auditing
rejects AITER/PyTorch computation shortcuts and requires each operator invocation
to launch FlyDSL, while permitting allocation/views/copies outside kernel compute.
