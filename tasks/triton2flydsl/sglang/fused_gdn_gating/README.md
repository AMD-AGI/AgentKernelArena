# fused_gdn_gating: Triton to FlyDSL task contract

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

`cases.json` contains 11 independent correctness identities and 9
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


All nine original BF16 decode cases remain. Return (g, beta_output), each with
shape [1,batch,heads], FP32 dtype and its input device. The existing baseline
stores beta_output in an FP32 tensor after rounding sigmoid values to b.dtype;
returning a BF16 container changes the interface. Both results must be finite.
The original FP32 comparisons remain: g atol=rtol=0.001; beta_output atol=0.02,
rtol=0.01 for the declared BF16 input. The nine timed cases retain BETA=1 and THRESHOLD=20.
All four inputs are read-only. Replay adds one to a and negates b, preserves
A_log/dt_bias, and checks both new outputs with these same rules.

Both actual measured outputs and the same captured graph replay are checked.
Outputs are poisoned before replay; reference work, perturbation and restoration
are outside timing. Original seed42+i, ten external warmups and100samples are
unchanged. An unavailable measured-output collector fails evaluation. Final
candidate calls are audited separately for FlyDSL arithmetic; the original
Triton implementation remains the frozen performance baseline.

## Local implementation helpers

Declared editable candidate sources may import additional task-local Python
device/compiler helpers, including packages. Dependency review and syntax
compilation follow the complete normal local-import closure. The same declared
implementation restrictions apply to every helper: protected references,
alternative final-language backends and dynamic implementation loading remain
disallowed. Source symlinks, imports escaping the task and local modules shadowing
the supported runtime libraries are rejected. Installed library source is not
traversed. Existing candidate-only PyTorch preparation and per-call FlyDSL launch
audits remain active outside timing. These checks enforce the task contract;
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

## Softplus correctness controls

Two additional correctness-only identities call the actual operator. One uses
the public beta=1/threshold=20 defaults with inputs at and above the threshold,
including a value where an unused naive FP32 exponential would overflow. The
other uses the public beta=2/threshold=1 parameters so ignoring the threshold
produces an observable error under the unchanged gate. The finite controls
return both outputs, compare against an independent scalar piecewise formula,
check read-only inputs and reuse the same allocations for base, changed and
base-again draws. All nine original scored shapes, seeds, warmups and samples
remain unchanged; these six calls are outside timing.

## Source module lifetime

The protected loader retains each successfully imported raw module for the
current isolated action, keyed by resolved source path and SHA256 of its bytes.
Correctness and performance reuse that module; a changed source gets a distinct
entry. Failed imports are not retained. This keeps compiled-owner globals alive
through graph capture without changing the operator, the nine timed cases or
their original timing parameters. Loading and hashing remain outside timing.
Candidate policy wrappers are created afresh for each audit context and remain
active on the cached raw module's actual calls.
