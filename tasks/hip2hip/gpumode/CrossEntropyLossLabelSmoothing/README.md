# CrossEntropyLossLabelSmoothing

Implement the operator defined by `pytorch_code_module/py_12501_CrossEntropyLossLabelSmoothing.py` in HIP. Edit only
`hip/hip_12501_CrossEntropyLossLabelSmoothing.hip`; the functional adapter, references, inputs, build policy,
comparison, timing code, and workload manifest are protected.

The extension must export `forward` through `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`.
Match the argument order used by `pytorch_code_functional/py_12501_CrossEntropyLossLabelSmoothing_func.py` and preserve its return
structure, shapes, dtypes, devices, parameter state, and numerical semantics.
GPU work must run through the submitted HIP extension, with the current PyTorch
HIP stream. Do not load or call protected baseline/reference code from the candidate.
PyTorch/ROCm, PyYAML, Ninja, and compiler headers are runtime dependencies
provided by the selected image.

The initial candidate is **implemented**. The baseline is the separately provided
HIP implementation `hip/hip_12501_CrossEntropyLossLabelSmoothing_ref.hip`.
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

## Declared workload and timed-output validation

The existing five cases use float targets and the default constructor: no
weights or smoothing distribution, `smooth_eps=None`, `from_logits=True`, and
mean reduction. The protected module applies log-softmax and the weighted
class sum on the last dimension. These declared cases define the scored
workload; other optional library configurations are not additional scored cases.
No existing case, input value, seed, or numerical gate is removed.

The supplied HIP baseline's explicit Event-only policy remains unchanged. Both
roles follow that fixed policy. The observer compares the actual last measured
Event output against the protected reference, poisons its storage and checks a
new invocation of the same callable. That invocation may allocate a new output;
it is not called a captured-graph replay. Graph-enabled paths instead check the
actual captured graph. Metadata identifies the observed invocation kind.
Reference comparison, output-contract and unchanged-input checks are outside
timed samples; 10 warmups and 100 samples are retained. Automatic graph-to-Event
fallback with an observer still fails closed in the canonical helper.

The scored Python call path is read-only: the benchmark checks caller inputs
and all model parameters/buffers after the actual timed call and its validated
re-execution. A modification fails validation. Original input and model tensor
values are restored in `finally`, including on exceptions, so a failed role
cannot alter the next role's starting state. Snapshot, checks and final cleanup
run outside the reported samples; existing per-invocation prepare callbacks
and the baseline's graph/Event policy retain their timing boundaries.

The scored operator is mean cross entropy with floating probability targets,
`from_logits=True`, and the **last axis as classes**, exactly as the original
HIP/PyTorch loss computes. The original five shapes, seeds and case IDs remain.
Target generation now normalizes that same last axis; the prior generator used
axis 1 even though the loss reduced the last axis. This is an explicit input
quality repair; old workload speedups are not directly comparable.

Cases declare smoothing epsilon 0, 0.1, 0.2, 0.4 and 0.6 and a nonuniform class
ramp distribution in `workload.json`. Both roles receive identical state outside
the timed invocation. PyTorch smoothing uses an out-of-place mixture, matching
the native implementation and keeping caller-owned targets immutable. The
smoothing buffer is included in replay state checks. An independent FP64
log-sum-exp formula checks the mean loss at the original comparison rule;
small analytic controls reject wrong axes, ignored smoothing and mutated targets.
The loss, mixture and reduction remain in the measured call. Integer labels,
ignore-index behavior, weighting, and other reductions are not scored by these
five declared cases; the task does not claim qualification for those API modes.
