# InnerProd

Implement the operator defined by `pytorch_code_module/py_11709_InnerProd.py` in HIP. Edit only
`hip/hip_11709_InnerProd.hip`; the functional adapter, references, inputs, build policy,
comparison, timing code, and workload manifest are protected.

The extension must export `forward` through `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`.
Match the argument order used by `pytorch_code_functional/py_11709_InnerProd_func.py` and preserve its return
structure, shapes, dtypes, devices, parameter state, and numerical semantics.
GPU work must run through the submitted HIP extension, with the current PyTorch
HIP stream. Do not load or call protected baseline/reference code from the candidate.
PyTorch/ROCm, PyYAML, Ninja, and compiler headers are runtime dependencies
provided by the selected image.

The initial candidate is **implemented**. The baseline is the separately provided
HIP implementation `hip/hip_11709_InnerProd_ref.hip`.
An empty target is allowed only during initial task validation; every final
candidate action must compile/load the actual HIP implementation. It never uses
a baseline fallback. Baseline correctness compares against the protected PyTorch
module; the PyTorch baseline is cross-checked against the independent functional form.

All **4 cases** in `workload.json` are mandatory. The manifest was enumerated
from the unchanged module `get_inputs()`; it is not inferred from candidate output.
The original correctness tolerance and RNG schedule remain in
`eval_tools/correctness_check.py` (model seed 0; comparison seed 1337 + case index).
The performance path retains the original `eval_tools/cal_kernel_perf.py` case
iteration, model state alignment, warmup 10, repetitions 100, input restoration
where supplied, and canonical graph/event benchmark helpers. The selected timing
policy is shared by both roles; candidate changes cannot select a weaker policy.

Use the argv prefix in `config.yaml` followed by one of:

- `validate-task`
- `baseline compile`, `baseline correctness`, `baseline performance`
- `candidate compile`, `candidate correctness`, `candidate performance`

Every action prints one `ARENA_EVAL_RESULT=` JSON envelope. Nonzero exits and
missing cases are failures. Runtime helpers such as `_aka_benchmark.py` are
materialized by Arena; do not replace them or edit generated helper regions.
The runner is task-local and does not import the Arena source tree or an agent.

## Original task instructions

You are a hip expert and good at gpu kernel implementation. Please implemnt a target HIP kernel code corresponding to pytorch modullle code provided as followings, which includes hip kernel, kernel laucher and python bliding code for the hip launcher.

The benchmark observes the actual timed output, compares it in full against the
protected functional reference, poisons it and validates the measured unit
again. Metadata distinguishes captured-graph replay from re-invoking an explicit
Event callable. Caller input values and independent output storage are checked.
The declared workload uses the original `get_init_inputs()`/`workload.json`
configurations in eval mode; optional API modes are not extra scored cases.
All original cases, input generation, seeds, numerical gates, 10 warmups and
100 samples are retained. Checks run outside measured samples.


The scored affine state is explicit in `workload.json`: channel scale
`scale[c] = 0.5 + 0.5 * (c+1)/C`, with nonzero scalar biases varying by case.
Both module and functional models receive identical state before correctness
and timing, outside measured calls and without consuming RNG. All original
case IDs, shapes, input values/seeds, numerical tolerances, warmups and samples
remain. Replacing all-one scale/zero bias is a task coverage repair; its timing
is not directly comparable to the former degenerate parameter workload.
The declared entrypoint is `forward`, with reduction over C and scalar bias
added afterward. Other methods are not substituted for this operator.

The scored Python call path is read-only: the benchmark checks caller inputs
and all model parameters/buffers after the actual timed call and its validated
re-execution. A modification fails validation. Original input and model tensor
values are restored in `finally`, including on exceptions, so a failed role
cannot alter the next role's starting state. Snapshot, checks and final cleanup
run outside the reported samples; existing per-invocation prepare callbacks
and the baseline's graph/Event policy retain their timing boundaries.

The initial HIP channel reduction uses FP64 accumulation after the original
FP32 image-times-scale multiplication, then rounds to FP32 before adding bias.
The former sequential FP32 sum failed the unchanged timed-reference gate on
case 3 with nontrivial affine state. Both provided baseline and initial candidate
receive the same numerical repair. Input shapes/values, explicit affine controls,
reference tolerance and sampling remain unchanged; the new baseline timing is
not comparable to the former less accurate implementation. Final candidates may
use any allowed reduction that passes the full original numerical gate.

The native kernels receive the bias device pointer and load its scalar inside
the kernel on the current HIP stream. The host must not dereference a GPU
pointer or capture a potentially stale bias value before queued state updates.
This applies to forward, nosum and pixelwise paths in both candidate and provided
baseline. It also keeps captured replay bound to the current device bias buffer;
no host synchronization or additional timed operator is introduced.
