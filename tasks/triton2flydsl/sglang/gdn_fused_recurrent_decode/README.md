# gdn_fused_recurrent_decode: Triton to FlyDSL task contract

The initial implementation is real **Triton**, not an empty FlyDSL starter.
Arena freezes it in a separate baseline workspace. The required final backend
is **FlyDSL**. Finishing with the original Triton implementation is not accepted,
even when it passes the numerical tests. No agent-specific driver is required.

Edit only `candidate.editable` paths in `config.yaml`. Keep the declared callable
interfaces and all outputs/state changes exercised by the protected harness.
Names containing `triton` are historical public API names; preserve those names
while replacing their implementation with FlyDSL. Private GPU function names can
change unless they are explicitly declared or called by the protected harness.

The final candidate-owned arithmetic must execute FlyDSL GPU kernels. Python and
PyTorch may prepare layouts, allocate storage and launch kernels. Candidate code
must not use PyTorch/AITER/Triton/reference/model code as a replacement operator,
or import task runners or protected reference functions. Do not introduce dynamic
module loading, external native kernels, subprocess dispatch or launch bypasses.
The protected harness's existing glue operations and allocation/reset boundaries
remain identical for baseline and candidate.

`cases.json` contains 10 independent correctness identities and 10
performance identities, including every original dtype, bias, activation, routing
and shape variant. Performance variants have correctness coverage. Input generators,
explicit seeds, numerical gates, output-contract checks, warmups, sample counts,
state reset and graph/event benchmark calls remain in `test_kernel_harness.py`.
Where the original suite allowed an environment dtype override, the manifest now
pins its original default; selecting a different suite requires updating both the
protected manifest and case definition. The small independent known answers in
`scripts/reference_controls.py` supplement the complete original GPU suite.

The source and module docstrings retain upstream operator semantics/provenance.
The task's public actions, from a materialized workspace, are:

```sh
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

The compile action syntax-checks actual role sources. Correctness performs real
GPU compilation/launches and the original comparisons. Every action emits one
`ARENA_EVAL_RESULT=` + `arena-eval-v1` JSON envelope. Failed or incomplete timings
cannot be scoreable. Arena owns score aggregation and final result files.

The runtime image provides ROCm, Triton (initial baseline) and FlyDSL (candidate),
plus any operator dependencies stated by the source. Arena must materialize the
canonical `_aka_benchmark.py` helper before GPU execution. CPU controls/protocol
checks do not qualify these GPU kernels. Existing legacy reports are historical;
the parent integration schedules new GPU validation.


All ten original packed-decode cases remain. Require finite output[B,1,HV,V]
and the updated full state pool in the input dtype/device. The original BF16
allclose rule remains atol=0.02,rtol=0.01 for output and selected state rows.
Replay negates packed V (leaving Q/K fixed) and initial state, so both results
negate while normalization, beta and decay parameters remain unchanged.

Unselected state slots must remain byte-identical to the initial pool. All
projection/gating/index inputs and the pristine initial state are read-only;
only the designated working state is mutated. The declared GPU cases use
unique nonnegative state indices within the pool; repeated/negative indices
are not part of this suite, and no concurrent-update ordering is promised.
Actual measured outputs include the working state. Replay poisons outputs and
uses the canonical collector's existing prepare_fn to restore the requested
initial state outside the measured invocation. The original preparation,
seed42+i, ten externalwarmups and100samples are unchanged for both roles.
Final candidate FlyDSL calls are audited independently of reference execution.

## Candidate execution policy during performance

The candidate-only PyTorch preparation policy remains active on every actual
operator call made by canonical benchmark warmup, graph capture and eager event
timing. Tensor arithmetic, reductions and library operators must execute as
FlyDSL kernels even when candidate behavior changes after earlier calls. A
probe of the actual performance closure observes FlyDSL invocation outside
clocks and capture; Python profiling is disabled inside canonical measurement.
Allocation and view preparation retain the existing allowlist. Baseline and
reference operators keep their original policy and all original device timing
parameters. One additional untimed audit probe of each performance closure runs
before canonical benchmarking; stateful closures use the existing prepare
callback before and after that probe. This probe is separate from the retained
benchmark warmup/sample counts. A forbidden operation remains a failure even if
a capture fallback or candidate code catches its exception. These checks are
implementation-contract evidence, not a Python
security sandbox.

Candidate sources and their normal local-import helpers must not access the
evaluator's module registry, stack frames or profiling controls through `sys`,
including aliases and dynamic attribute lookup. Ordinary `sys` metadata remains
available, and installed FlyDSL library internals are not scanned as candidate
code. Protected reference imports remain prohibited.


## Normalization option coverage

The public `use_qk_l2norm_in_kernel` option defaults to `False`: raw q/k are
used in the recurrence. The original ten scored cases explicitly use `True`;
their shapes, seeds, input draws, BF16 gates, state reset, 10 warmups, 100 samples
and canonical timing path are retained. Two extra correctness-only cases call
the explicit `False` option and omit it to exercise the public default. Both use
K128/V128, non-unit dyadic q/k, grouped value heads and unique nonsequential
state slots, with independent closed forms for the output and full state pool.
They check readonly inputs, caller-owned output/state aliases and unused slots.
The internal oracle's True default preserves historical scored call sites; its
explicit False branch is independently checked against these closed forms.
These finite controls cover the exposed mode without promising other shapes,
layouts or dtypes beyond the declared suite. No extra case is scored.

## Protected evaluator state

Candidate sources may not import the materialized timing helper `_aka_benchmark`,
assign or delete attributes of imported modules, read frame or function globals,
or disable the evaluator's PyTorch dispatch mode. When the local import closure
is resolved, a same-name `.py` module takes precedence over a namespace-package
directory, matching Python. These structural checks are not a security sandbox.
