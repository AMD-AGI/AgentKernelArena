# gemm_a8w8_bpreshuffle_kernel: task-owned v2 contract

Implement or optimize gemm_a8w8_bpreshuffle in FlyDSL, preserving all task inputs, outputs and numerical gates.

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
operator is `flydsl_gemm_a8w8_bpreshuffle`; any additional declared callables used by the harness
remain required. Legacy build/compile helpers are optional implementation details;
no builder return protocol is required by this task. A candidate may choose its
own internal compilation helpers, while implementing all tested work in FlyDSL.


The actual required interfaces are `flydsl_gemm_a8w8_bpreshuffle` and
`preshuffle_weight_a8`, both declared in config. The latter is untimed host
layout preparation: preserve exactly the original (16,16) byte permutation of
FP8[N,K] weights. PyTorch views/permutes/copies are allowed for that operation;
GEMM arithmetic must execute FlyDSL. The protected harness verifies packed bytes
against its independent layout construction and rejects mutation of the weights.
The GEMM returns finite BF16[M,N] on the input device, with all raw/quantized
inputs and FP32[M,1]/FP32[N,1] scales read-only. The gate stays
max_abs_error/max_abs_reference<=0.01 (zero reference uses raw max_abs_error);
elementwise allclose percentage is diagnostic only. The five cases, tiling,
seed, original quantization/model and source kernel remain unchanged.
Both roles retain explicit Event timing: ten external warmups, zero additional
collector warmups and 100 measured samples. The unquantized torch matmul timing
remains diagnostic; Arena's baseline is the frozen initial FlyDSL implementation.
Validate the last actual measured output against quantized FP32 GEMM/BF16 cast,
then halve both scale tensors outside timing, poison the old output and rerun
the same eager callable. Compare the result against the original numerical gate
and restore inputs. No captured-graph claim is made for this Event invocation.
The original source imports the older FlyDSL buffer_ops API: qualify it with the
pinned compatible runtime and record that image digest, not an untested image.
