# fav3_sage_mxfp4: Triton to FlyDSL task contract

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

`cases.json` contains 5 independent correctness identities and 5
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


This task retains its original five BF16 bshd cases, with q_smooth=True,
hadamard_rotation=True, BLOCK_R=min(64,head_dim), R=None, block_lut=None,
return_lse=False and smooth_k=True. The required numerical gate is unchanged:
normalized maximum error<=0.25 AND at most1.5% elements exceeding the original
atol=0.3/rtol=0.25 rule. The original gfx950 constraint remains.

These fixed workload choices define the evaluated Arena suite; extra upstream
wrapper modes are not qualified by these reports. No original case or selected
flag was removed. The candidate must preserve the public entry and all calls
made by this suite; private module constants/kernel symbol names are not new
candidate requirements. Output must be finite BF16 with q.shape/device, and
Q/K/V are read-only. The independent full-precision reference and its original
comparison expressions remain unchanged. The actual measured output is checked,
then V is negated outside timing, the previous output poisoned, and that same
measured invocation replayed under both numerical gates. This also validates the
quantization and internal preparation performed inside the original wrapper.
The original10external warmups,100samples, seed42+i, kernel config and allocation
boundaries are unchanged for baseline and candidate. Final FlyDSL invocation
auditing is separate from oracle computation and runs outside timing.

## Protected evaluator state

Declared editable candidate sources may import new task-local Python helper
modules or packages. Those helpers are part of the candidate implementation:
before candidate import, dependency review follows the normal task-local import
closure and applies the same restrictions to every helper. Protected harness,
reference and script imports, symlinked or escaping helper imports, and local
modules that shadow the supported runtime libraries are rejected; installed
library source is not traversed. Final FlyDSL candidate sources and their local
helpers also may not reach the evaluator's module registry, stack frames or
profiling controls through `sys`, including `os.sys`, aliases and dynamic
attribute lookup. Ordinary `sys` metadata remains available.

Candidate sources may not import the materialized timing helper `_aka_benchmark`,
assign or delete attributes of imported modules, read frame or function globals,
or disable the evaluator's PyTorch dispatch mode. When the local import closure
is resolved, a same-name `.py` module takes precedence over a namespace-package
directory, matching Python. These structural checks are not a security sandbox.

## Initial baseline numerical implementation

The baseline uses a two-component MXFP4 expansion for the rotated Q/K operands.
Each component has E2M1 values and E8M0 scales. Four feature contractions compute
`(Q0 + Q1) @ (K0 + K1).T`, retaining the first quantization residual instead of
changing the reference or the numerical gates. Conversion uses nearest-even
rounding and a shared scale derived from the block maximum. Zero V blocks avoid
0/0 during FP8 conversion. External shapes, BF16 I/O, seeds, selected flags and
both numerical acceptance rules are unchanged. This increases internal work and
storage, so it defines a new baseline identity requiring fresh measurements of
both roles; old measurements cannot establish a speedup against this baseline.
The declared wrapper remains the task interface. Internal quantized Q/K feature
widths include the residual expansion and need not equal the external head size.
