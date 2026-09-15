# Matmul_Swish_Sum_GroupNorm

Implement the operator defined by `pytorch_code_module/py_l2n37_Matmul_Swish_Sum_GroupNorm.py` in HIP. Edit only
`hip/hip_l2n37_Matmul_Swish_Sum_GroupNorm.hip`; the functional adapter, references, inputs, build policy,
comparison, timing code, and workload manifest are protected.

The extension must export `forward` through `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`.
Match the argument order used by `pytorch_code_functional/py_l2n37_Matmul_Swish_Sum_GroupNorm_func.py` and preserve its return
structure, shapes, dtypes, devices, parameter state, and numerical semantics.
GPU work must run through the submitted HIP extension, with the current PyTorch
HIP stream. Do not load or call protected baseline/reference code from the candidate.
PyTorch/ROCm, PyYAML, Ninja, and compiler headers are runtime dependencies
provided by the selected image.

The initial candidate is **unimplemented**. The baseline is the separately provided
PyTorch module `pytorch_code_module/py_l2n37_Matmul_Swish_Sum_GroupNorm.py`.
An empty target is allowed only during initial task validation; every final
candidate action must compile/load the actual HIP implementation. It never uses
a baseline fallback. Baseline correctness compares against the protected PyTorch
module; the PyTorch baseline is cross-checked against the independent functional form.

All **5 cases** in `workload.json` are mandatory. The manifest was enumerated
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

You are a HIP expert skilled at GPU kernel implementation. Please implement target HIP kernel code corresponding to the provided PyTorch module, including the HIP kernel, kernel launcher, and Python binding code for the HIP launcher.


Timing observes and checks the actual complete output of the measured graph or
explicit Event callable, then poisons its output and checks a re-execution of
that measured unit against the protected functional reference. All original
cases, input generation, seeds, numerical tolerances, 10 warmups and 100 samples
are preserved. Models run in the original eval mode: module dropout is disabled
and batch-normalization uses frozen statistics. Checks are outside samples.

The scored Python call path is read-only: the benchmark checks caller inputs
and all model parameters/buffers after the actual timed call and its validated
re-execution. A modification fails validation. Original input and model tensor
values are restored in `finally`, including on exceptions, so a failed role
cannot alter the next role's starting state. Snapshot, checks and final cleanup
run outside the reported samples; existing per-invocation prepare callbacks
and the baseline's graph/Event policy retain their timing boundaries.
