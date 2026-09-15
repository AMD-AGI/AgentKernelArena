# mla_decode

Optimize the declared MLA decode launch functions for the configured AMD GPU.

## Candidate and baseline contract

The actual initial `kernel.py` functions are implemented. Optimize the declared
function scopes (and permitted implementation helpers); keep other functions,
references, input builders, parameter tables and output interfaces unchanged.
Some declared entrypoints are Python launch functions over Triton kernels; keep
their calling convention. A final empty or missing entrypoint is an error, never
a request to run a reference implementation instead.

Arena creates an independent frozen `initial_candidate` workspace for baseline
measurement. Both roles execute this task's local `kernel.py`; no `GEAK_WORK_DIR`
or external worktree selects a candidate or baseline. Installed PyTorch/Triton are runtime image dependencies. AITER modules are loaded
from the declared, checked source tree described below. No Arena or agent code
is imported.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or
`python3 _arena_eval.py baseline|candidate compile|correctness|performance`, selecting
one role and action. The runner emits one `ARENA_EVAL_RESULT=` envelope using
`arena-eval-v1`. Arena owns final reports and scoring.

`workloads.json` independently records 129 required cases, including all
128 original full-benchmark cases. The original correctness selection
(16 cases) runs first with its original ordering/seeding. Any additional
benchmark cases then receive the same numerical/output checks. Distinct original
correctness-only cases remain in the manifest; repeated performance configurations
keep their separate indices. No skipped/missing case or incomplete action passes.

Compilation retains the original syntax check. Numerical checks actually execute
the candidate and compare with the protected references; original numerical tolerances are retained, with the full-output checks below. Full benchmark input order, seeds, allocation/reset behavior,
warmups, iterations and the canonical helper's median calculation remain unchanged.
The adapter collects fresh device measurements directly from the benchmark calls;
old `build/performance_report.json` files and log text cannot supply evidence.
The original optional reference timings remain diagnostic; Arena uses the frozen
initial implementation's measured times for scoring. No GPU qualification is
implied by the CPU migration checks.

Do not edit the materialized dependencies, `runtime-dependencies.json`,
`_aiter_dependency.py`, `test_kernel_harness.py`, `_timed_contract.py`, `_arena_*.py`, `workloads.json`, or generated
`_aka_benchmark.py`. The runtime must materialize the canonical benchmark helper
next to the original harness even though the public runner is `_arena_eval.py`.
Unsupported hardware or missing dependencies return a failing envelope; use a
compatible image/GPU before scheduling this task.

The original wrapper source provenance remains revision
`22122345c03991cb8026947b8df05e02f50d1f88`. Its runtime dependency is now explicitly
materialized by `workspace.sources` from the selected image's AITER Triton source.
`runtime-dependencies.json` pins the required implementation modules and tuning
configs by SHA256 at qualified AITER revision
`4ad99832823dde2315b361cbd3b54b1c5c12acd5`. Setup verifies these bytes before the
baseline is frozen. A missing or different dependency fails; it never silently
uses an installed AITER package instead.

The protected `_aiter_dependency.py` binds Python source namespaces to the copied
tree so the original kernel imports execute those exact modules. This avoids
unrelated AITER quant/comms/C++ package initializers. It does not replace either
Triton stage or modify imported source bytes. The complete task works from its
materialized dependency tree, with PyTorch/Triton supplied by the runtime image.

## Complete output and measured invocation checks

The scored domain is one BF16 query per sequence, page size one, two KV splits,
`use_rope=False`, and `logit_cap=0`. Both declared Python launch functions call
declared materialized AITER **Triton** primitives; there is no HIP assembly stage in this
local source. The fixed image's actual AITER revision must be qualified.

The original elementwise `atol=rtol=0.01`, at most 5% mismatching-elements policy
is retained as a legacy diagnostic, and **all elements must additionally pass**
`atol=rtol=0.01`; the 5% allowance cannot exempt any bad coordinate. Every output must have the expected shape/dtype/device,
be finite, and lie in its value-coordinate convex range (with the same 0.01
rounding allowance). The legacy 5% policy is not permission for NaNs/infinities
or arbitrary unbounded values. A separate, unscored zero-query control checks
all coordinates against the known uniform-attention mean. There are 129
correctness cases and the unchanged 128 scored cases. Optional fused RoPE is not
part of this task's declared workload; its broader library interface is not an
unmeasured score claim.

Correctness uses pristine, independent oracle inputs prepared before the candidate.
Performance retains the original helper, warmups, repetitions, allocation and
scored case order. After timing, the actual `TimedRun.outputs` must match that
oracle; the same captured invocation is then replayed after changing inputs in
place and poisoning writable outputs. Its complete outputs must match a newly
computed private oracle. Every read-only input is checked byte-for-byte and
restored in `finally`, including failure paths. These checks run outside timing
and identically for the frozen initial candidate and submitted candidate.
