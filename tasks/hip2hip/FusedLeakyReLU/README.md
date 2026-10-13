# FusedLeakyReLU

Implement the operator defined by `pytorch_code_module/py_10190_FusedLeakyReLU.py` in HIP. Edit only
`hip/hip_10190_FusedLeakyReLU.hip`; the functional adapter, references, inputs, build policy,
comparison, timing code, and workload manifest are protected.

The extension must export `forward` through `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`.
Match the argument order used by `pytorch_code_functional/py_10190_FusedLeakyReLU_func.py` and preserve its return
structure, shapes, dtypes, devices, parameter state, and numerical semantics.
GPU work must run through the submitted HIP extension, with the current PyTorch
HIP stream. Do not load or call protected baseline/reference code from the candidate.
PyTorch/ROCm, PyYAML, Ninja, and compiler headers are runtime dependencies
provided by the selected image.

The initial candidate is **implemented**. The baseline is the separately provided
HIP implementation `hip/hip_10190_FusedLeakyReLU_ref.hip`.
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

You are a hip expert and good at gpu kernel implementation. Please implemnt a target HIP kernel code corresponding to pytorch modullle code provided as followings, which includes hip kernel, kernel laucher and python bliding code for the hip launcher.

The performance adapter checks the complete observed timed output against the
protected functional reference, poisons output storage and replays the exact
measured graph. It also checks the public non-aliasing and unchanged-input
contract. Validation runs outside measured samples with the existing tolerance,
cases, warmups and repetition counts. The framework identifies captured-graph and explicit Event observations;
automatic unsupported fallback cannot be reported as replay-validated.

The implemented initial candidate is a supplied optimization starting point and
may have identical source to the provided HIP baseline. Both are compiled from
their own declared files and checked against the independent PyTorch reference.
Every candidate action must execute its own compiled entrypoint; protected
baseline/reference imports, calls and data access remain prohibited.

## Explicit parameter coverage

The five existing case IDs, tensor shapes, input seeds, numerical tolerances,
and timing policy are retained. `workload.json` now declares nonzero model state:
`bias[c] = (-1)^c * (1/8 + (c+1)/128)`, with slope/scale pairs
`(0.1, 0.5)`, `(0.2, sqrt(2))`, `(0.35, 1.25)`, `(0.5, 2)`, `(0.75, 3)`.
This intentionally replaces zero-only bias coverage; it is a task-quality repair,
not a comparable optimization result against the former zero-bias workload.

The protected `case_controls.py` applies the manifest state before each
correctness case and before each baseline/candidate timing call. Both roles
evaluate identical operator parameters. State construction is outside the
measured calls; graph capture, 10 warmups, 100 samples, input allocation and
restoration, and the full-reference timed replay checks retain their boundaries.
Nonzero channels and varied slopes/scales make omitted bias, wrong channel
indexing, or hardcoded activation parameters detectable.

After checking the actual timed output and same-input poisoned replay, both
roles also replay the same captured graph with floating inputs changed in place
to `0.5 - x`. The protected reference is recomputed for those fresh values and
the entire result must pass the original `rtol=1e-4`, `atol=1e-5` comparison.
The fresh oracle must distinguish the old answer under that same rule. Original
inputs are restored in `finally`, including on a failed check, so the other role
receives identical scored inputs. This control consumes no RNG, changes no
model parameters, and occurs outside all timing samples. It detects returning
a cached original answer; it does not prove absence of every caching strategy.

The scored Python call path is read-only: the benchmark checks caller inputs
and all model parameters/buffers after the actual timed call and its validated
re-execution. A modification fails validation. Original input and model tensor
values are restored in `finally`, including on exceptions, so a failed role
cannot alter the next role's starting state. Snapshot, checks and final cleanup
run outside the reported samples; existing per-invocation prepare callbacks
and the baseline's graph/Event policy retain their timing boundaries.
