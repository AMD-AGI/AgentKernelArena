# jagged_dense_bmm_kernel: task-owned v2 contract

Implement or optimize jagged_dense_bmm in FlyDSL, preserving all task inputs, outputs and numerical gates.

The candidate starts **implemented**. Arena freezes the implemented source in a separate baseline workspace.
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
operator is `flydsl_jagged_dense_bmm`; any additional declared callables used by the harness
remain required. Legacy build/compile helpers are optional implementation details;
no builder return protocol is required by this task. A candidate may choose its
own internal compilation helpers, while implementing all tested work in FlyDSL.


The original five jagged-group cases (including empty groups), BF16 source/model,
N=K=128, seed20260601 and normalized max error<=0.01 remain unchanged. Zero
reference uses raw maximum error. Output must be finite BF16[total_M,N] on the
input device; jagged rows, dense weights, biases and INT32 offsets are read-only.
The original performance entry is the prepared `jagged_dense_bmm` launch, while
ordinary correctness also calls `flydsl_jagged_dense_bmm`. Both paths now run
correctness checks under the original model/gate; final FlyDSL auditing covers
both calls. The module proxy keeps their internal calls intact. Preserve the
existing prepared launch interface: BLOCK_M, flyc.from_dlpack and fx.Stream
are used for stable padded output/operand views before timing; the low-level
function receives these views, dense/bias data and fixed group metadata.
Check the actual measured prepared-launch output, then negate dense weights
and biases outside timing, poison output and replay the same measured call.
Restore all inputs afterwards. The prepared function, padded output allocation,
fixed metadata,10external warmups/100samples, diagnostic reference10warmups and
graph policy are unchanged. No metadata construction moves into timed work.
The unchanged source uses older FlyDSL APIs; record its pinned compatible image
in full GPU qualification, and do not infer support for another runtime.


The candidate audit permits the original launch-metadata calculation: slices
and int32/int64 conversions of the protected `seq_offsets`, pairwise subtraction
of those integer offsets, the maximum group length, and scalar extraction.
The public entry and separately prepared entry pass their actual offset argument
as the sole provenance root. Only integer views/conversions derived from that
root inherit this permission. Arbitrary integer allocations, dense/jagged/bias
values, and float casts cannot use this arithmetic exception; overwriting tracked
metadata is rejected. Both candidate paths still require a FlyDSL launch and
retain their ordinary output/input checks. The original group sizes, mathematical
reference, numerical gate and prepared timing scope are unchanged.
