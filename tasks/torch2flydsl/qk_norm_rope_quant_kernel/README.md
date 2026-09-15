# qk_norm_rope_quant_kernel: task-owned v2 contract

Implement or optimize qk_norm_rope_quant in FlyDSL, preserving all task inputs, outputs and numerical gates.

The candidate starts **implemented**. Arena freezes the implemented source in a separate baseline workspace.
The original harness's primary implementation timing is retained; additional
reference/operator timings are diagnostic only. `test_kernel_harness.py` defines
the exact dispatch and allocation boundary for this task. The provided path uses
the task-local PyTorch model or installed AITER operator specified there, with its
original graph/event policy. `model.py` is protected reference/source material;
its presence alone does not select the performance baseline.

There are 6 declared cases in `cases.json`. All original dimensions,
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
operator is `flydsl_qk_norm_rope_quant`; any additional declared callables used by the harness
remain required. Legacy build/compile helpers are optional implementation details;
no builder return protocol is required by this task. A candidate may choose its
own internal compilation helpers, while implementing all tested work in FlyDSL.


The six original cases exercise the BF16 `quant=False` path: output must be
(BF16 Q[T,H,D], BF16 KV[T,D], None, None), on the input device. Quantization
scales must remain None. Q/KV, KV weight, cosine/sine tables and positions are
read-only; the original strided KV input remains strided. Preserve headwise
RMSNorm eps1e-6, KV-only gamma and GPT-J paired rotation of the RD tail, with
the original normalized max-error gate<=1e-2 independently for Q and KV
(denominator max_abs_reference+1e-9). The allclose percentages are diagnostic.
Validate all four values of the actual measured tuple, then negate Q and KV
outside timing, poison both output tensors and replay the same measured
invocation against the unchanged model. Recheck the two None scale slots on
replay too, and restore the inputs. Baseline/candidate retain the original
10external warmups,100samples and graph timing; diagnostic model timing retains
its original10warmups. Source kernel, model, cases, group-size variants and seed
are unchanged. Final candidate arithmetic must run FlyDSL; the candidate-only
auditor permits host preparation and checks operator calls outside timing.
This original source uses the older FlyDSL buffer_ops API, so its full GPU
qualification requires the pinned compatible image recorded with the report.

The allowed preparation dependencies include exactly
`from aiter.utility import dtypes` (with an optional alias), used by the original
lazy quantization helper to select a hardware dtype constant. Importing the
AITER package, other utility members, operators or wildcard members remains
forbidden. This allowance does not permit AITER operator compute; the final
candidate's actual calls still undergo FlyDSL and PyTorch-operation checks.
The six evaluated cases all use quant=False and do not execute that helper.

Module lifetime is held through the whole action: correctness and performance
reloads may overwrite the same import alias, but previously loaded FlyDSL
modules remain strongly referenced until the action exits. This prevents old
compiled-module finalizers from unloading HIP modules during a later graph
capture. It does not alter kernels, inputs, warmups, samples or the graph policy.
The motivating failed run reported hipModuleUnload/StreamCaptureUnsupported,
followed by capture invalidation; a fresh full GPU run must qualify this fix.

The imported dtype alias may only be read as `<alias>.fp8`; module reassignment,
passing the module to another function, dynamic attribute access, other members
(including any library imported by the dtype module), and relative/package
import alternatives are rejected. Reading a hardware dtype constant does not
expose an AITER operator dependency to the candidate.
