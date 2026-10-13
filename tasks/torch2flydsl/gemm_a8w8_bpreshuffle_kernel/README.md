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
collector warmups and 100 measured samples. Sample zero uses the original
seeded data; each later sample receives a distinct seeded BF16 activation and
weight pair, quantized and preshuffled outside timing. A read-only observer
prepares one quantized FP32 GEMM/BF16 reference before each start Event, then
checks that sample's actual output and input immutability after its end Event.
The complete output is compared under the original numerical gate and released,
so all 100 samples are checked with bounded host memory. Oracle preparation
can change cache state identically for baseline and candidate. The unquantized torch matmul diagnostic
receives the same raw input stream. Poisoning and replay also use independent
new quantized operands and verified packed weights, with a distinguishable
reference; every input is restored on exit. The five shapes, tolerance and
full public operator timing boundary remain unchanged, though this stronger
data stream makes historical performance measurements not directly comparable.
No captured-graph claim is made for this Event invocation. The task-local
FlyDSL 0.3.2 buffer adapter is part of the initial implementation; edits to it or the harness require fresh qualification in the pinned image.
