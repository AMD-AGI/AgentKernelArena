# fused_dual_residual_rmsnorm: Triton to FlyDSL task contract

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

`cases.json` contains 8 independent correctness identities and 8
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


All eight original cases remain, including BF16/FP16/FP32 input and non-power-of-2
hidden dimensions. Both output and intermediate residual must retain the input
shape, dtype and device. All input tensors are read-only. The two outputs use
the original FP32 allclose checks: atol=rtol=0.01 for BF16/FP16,0.0001for FP32,
with EPS=1e-6 in both normalizations and the original intermediate rounding.

The actual measured pair and the same captured graph replay must both satisfy
these rules. Untimed replay negates x and residual, which negates both outputs
without changing either RMS denominator. Both measured output buffers are
poisoned before replay; inputs are restored afterwards. Independent EPS-scale
known answers check both stages separately. Originalcase identities,seeds42+i,
ten externalwarmups and100graphsamples remain; capture failure cannot bypass
validation. Only final candidate calls are audited for FlyDSL computation;
the original frozen Triton implementation and numerical reference are unchanged.

## Local implementation helpers

Declared editable candidate sources may import additional task-local Python
device/compiler helpers, including packages. Dependency review and syntax
compilation follow the complete normal local-import closure. The same declared
implementation restrictions apply to every helper: protected references,
alternative final-language backends and dynamic implementation loading remain
disallowed. Source symlinks, imports escaping the task and local modules shadowing
the supported runtime libraries are rejected. Installed library source is not
traversed. The candidate-only PyTorch preparation and FlyDSL launch audits apply
to these helpers too; the performance policy states how timed calls are checked. These checks enforce the task contract;
they are not a Python security sandbox.

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

## Protected evaluator state

Candidate sources may not import the materialized timing helper `_aka_benchmark`,
assign or delete attributes of imported modules, read frame or function globals,
or disable the evaluator's PyTorch dispatch mode. When the local import closure
is resolved, a same-name `.py` module takes precedence over a namespace-package
directory, matching Python. These structural checks are not a security sandbox.
