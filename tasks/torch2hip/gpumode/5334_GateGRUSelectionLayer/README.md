# GateGRUSelectionLayer

Implement the operator defined by `pytorch_code_module/py_5334_GateGRUSelectionLayer.py` in HIP. Edit only
`hip/hip_5334_GateGRUSelectionLayer.hip`; the functional adapter, references, inputs, build policy,
comparison, timing code, and workload manifest are protected.

The extension must export `forward` through `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`.
Match the argument order used by `pytorch_code_functional/py_5334_GateGRUSelectionLayer_func.py` and preserve its return
structure, shapes, dtypes, devices, parameter state, and numerical semantics.
GPU work must run through the submitted HIP extension, with the current PyTorch
HIP stream. Do not load or call protected baseline/reference code from the candidate.
PyTorch/ROCm, PyYAML, Ninja, and compiler headers are runtime dependencies
provided by the selected image.

The initial candidate is **unimplemented**. The baseline is the separately provided
PyTorch module `pytorch_code_module/py_5334_GateGRUSelectionLayer.py`.
An empty target is allowed only during initial task validation; every final
candidate action must compile/load the actual HIP implementation. It never uses
a baseline fallback. Baseline correctness compares against the protected PyTorch
module; the PyTorch baseline is cross-checked against the independent functional form.

All **5 cases** in `workload.json` are mandatory. The manifest was enumerated
from the protected module `get_inputs()`; it is not inferred from candidate output.
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

The scored Python call path is read-only: the benchmark checks caller inputs
and all model parameters/buffers after the actual timed call and its validated
re-execution. A modification fails validation. Original input and model tensor
values are restored in `finally`, including on exceptions, so a failed role
cannot alter the next role's starting state. Snapshot, checks and final cleanup
run outside the reported samples; existing per-invocation prepare callbacks
and the baseline's graph/Event policy retain their timing boundaries.

## Workload geometry

Three 1024-to-512 gated projections, with 1/17/63/258/1024 flattened rows and distinct input tensors.

Both protected models use `get_init_inputs()` = `[[], {'dim_model': 512, 'dim_ff': 2048, 'prob_dropout': 0.5}]`.
Each manifest row records this constructor in `params.model_init`; task
validation checks it against both implementations. Inputs remain FP32, and
the exact input shapes and strides are listed in `workload.json`.
The flattened output-row counts are `1, 17, 63, 258, 1024`.

The largest declared input tuple contains 4.000 MiB; model
parameters contain 6.006 MiB and the largest output contains
2.000 MiB. The largest single operator intermediate is at most
4.000 MiB. These are FP32 tensor sizes, excluding
the evaluator's reference copies, snapshots, allocator and graph/compiler memory.
The existing `dim_ff` and `prob_dropout` constructor arguments remain interface
compatibility values; this operator computes its three gates with `2D -> D`
linear layers and applies no dropout.

These enlarged workloads change the task contract and require fresh GPU
`task_validator` qualification plus a correct HIP candidate on every declared
case. An initial generation-stub validation does not establish candidate
feasibility, and earlier small-workload results do not qualify this contract.

## Independent measured inputs and replay validation

The original correctness inputs, shape/dtype/stride rows, constructors, model
parameters, reference math, tolerances, 10 warmups and 100 reported samples are
retained. Performance now uses an independent input draw before every prepared
invocation, including warmup, capture/calibration and each measured call.
Preparation 0 retains the original input; later preparations draw the same FP32
CPU standard-normal distribution used by the protected `get_inputs()` generators,
using local generators. It does not advance their global RNG. Persistent model
weights remain fixed for the case. Both roles share the same seed plan.

The seed base is derived from the model class and declared input shape/dtype/stride
identity; operand seed = seed_base + 4 * preparation_index + operand_index.
Both roles use this same generation policy. The actual scored seeds and bytes
match when the roles use the same invocation kind and preparation schedule.
Explicit Event timing and graph timing have different preparation prefixes;
across those kinds, only the equivalent input distribution is claimed, and the
actual scored draw IDs must be inspected rather than assumed to be paired.
Every measured draw records its actual preparation index, seeds and SHA256 of
all input tensor bytes. The raw metadata and qualified runtime/source identities
are required to reproduce those draws; no equality across PyTorch versions is
assumed. Each of the 100 observed samples must have a distinct input fingerprint.

The existing canonical `prepare_fn` runs generation, input copies and protected
reference computation before the start event. Its existing behavior captures one
logical operator invocation per graph replay when preparation is present; this
prevents amortizing a result cached across identical calls inside one graph.
The timer implementation and sample count are unchanged. `before_sample` and
`after_sample` only inspect tensors and update evidence counters, and require
one observed invocation per sample. Every output and caller/model tensor state
is checked with the original rule. The protected canonical schedule is verified
(10 warmups plus five graph preparation steps before the measured stream; the
explicit Event path has the warmups alone). Missing observations or a different
preparation schedule fail instead of accepting partial evidence. With the actual
`TimedRun` collector, an effectively empty graph estimate or a capture exception
fails closed in the canonical helper; it does not produce a scored automatic
Event fallback. Explicit `use_cuda_graph=False` and the protected forced-Event
option use the observable eager path. CPU stub tests execute these actual
canonical branches, without claiming native GPU timing or graph qualification.

After timing, the exact measured graph or Event callable is checked again with
poisoned output and independent per-operand input perturbations, using the
protected reference to require a meaningful expected change. Original caller
inputs are also restored and checked with that same measured unit. Final cleanup
restores inputs and model tensor values even on rejection. Preparation and all
validation stay outside the reported device-time intervals.

These timed draws and single-invocation graph semantics materially change the
old fixed-input reward workload. Old scores are not directly comparable and old
GPU qualification does not transfer: fresh paired measurements and task-validator
PASS are required. Finite controls reject stale-result copies; a valid cache that
actually recomputes for every fresh draw can pass, but cannot skip that timed work.
This is not a general native-code sandbox or a proof against every conditional
cache strategy. NumPy, already required by the runtime smoke, supplies byte views
for the FP32 input fingerprints.
