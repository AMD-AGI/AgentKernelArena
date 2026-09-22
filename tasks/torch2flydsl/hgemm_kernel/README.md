# hgemm_kernel: task-owned v2 contract

Implement or optimize hgemm in FlyDSL, preserving all task inputs, outputs and numerical gates.

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

Measured outputs and subsequent invocations are checked against the protected
FP32-accumulating, BF16-output reference using the original numerical gate.
The harness also rejects wrong shape/dtype/device, non-finite outputs and
read-only input mutation. Input perturbation, reference calculation and output
poisoning occur outside timing, identically for baseline and candidate.
Explicit Event timing observes the last actual measured sample; its validation
re-invokes the same eager callable and is not captured graph replay. Graph timing
validates the actual captured replay. Original shapes, seeds, tolerance, warmup,
sample counts and Graph/Event selection remain unchanged.

Runtime qualification: the initial implementation uses the legacy FlyDSL
`expr.buffer_ops` and `expr.vector` APIs. It passed the full task validator on
MI355X with the repository's pinned SGLang 0.5.14 / FlyDSL 0.2.2 runtime
(2026-09-15, job 139392 on an available-memory MI355X allocation; all five
correctness and performance cases, including
measured Event outputs and eager re-invocation). The tested SGLang 0.5.19 /
FlyDSL 0.3.2 runtime removes these APIs and fails before kernel execution.
Select the qualified runtime through the run-level Docker image setting;
an unchanged source under 0.5.19 is not qualified. Candidate and baseline must
use the same runtime and timing method. This initial-task validation does not
certify a subsequently modified candidate.


Only the entrypoints listed in config.yaml are required interfaces. The primary
operator is `flydsl_hgemm`; any additional declared callables used by the harness
remain required. Legacy build/compile helpers are optional implementation details;
no builder return protocol is required by this task. A candidate may choose its
own internal compilation helpers, while implementing all tested work in FlyDSL.

Candidate dependency enforcement runs before candidate import for compile,
correctness and performance. AITER package/operator imports are forbidden,
including `aiter.ops.flydsl` implementations; a FlyDSL runtime call from an
imported operator is not candidate-owned arithmetic. Import aliases and
`from ... import ...` do not change this rule. External backend/native dispatch
(`ctypes`, subprocesses, or `torch.ops`) and dynamic implementation loading are
also forbidden. Ordinary Python utilities, PyTorch allocation/layout operations,
and the task's bundled `kernels/` helpers remain available under the existing
numerical and timing contract. Baseline checks retain their declared initial
backend; the final candidate must use FlyDSL.
