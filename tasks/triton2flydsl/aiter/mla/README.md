# mla: Triton to FlyDSL task contract

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

`cases.json` contains 6 independent correctness identities and 6
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


All six original BF16 paged-latent attention cases remain, including128query
heads/lora512 and the four-query non-ALL_DECODE path. The result must write and
alias the supplied [total_queries,query_heads,lora_rank] BF16 output buffer.
Q, KV pages, block tables, cumulative query offsets and sequence lengths are
read-only. The original normalized maximum error<=0.01 remains the sole numerical
gate; allclose at0.01 remains diagnostic. No reference or baseline algorithm changed.

The actual measured output and the same captured graph replay must both pass.
Untimed replay negates Q and KV together: QK logits stay the same, latent values
and the attention output negate. Page addresses and lengths stay unchanged.
Replay preparation, independent reference work and input restoration are outside timing.
All original seeds42+i, ten external warmups and100graph samples remain.
Each measured invocation poisons the supplied output before launching the kernel,
including every captured graph replay. Both roles include this reset in timing;
earlier timings without the per-invocation reset are not directly comparable.
Correctness also exercises reversed physical-page addresses and differing partial
sequence lengths using the same independent oracle and unchanged 1% gate.

Runtime qualification must bind the exact image. The gfx950 decode kernel uses
two pipeline stages with four waves per block: the two-wave pipelined schedule
produced incorrect output on the ROCm 10 Triton runtime. The four-wave schedule
retains asynchronous loads and double-buffered shared memory instead of disabling
the pipeline. The operator, all six workloads and their numerical gates remain
unchanged. The task correctness action retains its six seeded workloads and adds
the page-routing and partial-length checks described above. Separately, repository
GPU regression tests exercise additional seeds and repeated calls for the
128-head/lora512 case. A full validator report is still required for the selected
runtime; do not change the baseline after a session has frozen it.

## Protected evaluator state

Before candidate import, dependency review follows the candidate's normal
task-local import closure, including new helper modules and packages. Protected
harness, reference and script imports, symlinked or escaping helper imports, and
local modules that shadow the supported runtime libraries are rejected; installed
library source is not traversed. Final FlyDSL candidate sources and their local
helpers also may not reach the evaluator's module registry, stack frames or
profiling controls through `sys`, including `os.sys`, aliases and dynamic
attribute lookup. Ordinary `sys` metadata remains available.

Candidate sources may not import the materialized timing helper `_aka_benchmark`,
assign or delete attributes of imported modules, read frame or function globals,
or disable the evaluator's PyTorch dispatch mode. When the local import closure
is resolved, a same-name `.py` module takes precedence over a namespace-package
directory, matching Python. These structural checks are not a security sandbox.
