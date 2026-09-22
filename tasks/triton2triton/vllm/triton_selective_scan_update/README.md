# triton_selective_scan_update

The starting candidate is implemented Triton. Improve only the declared kernel symbols and permitted new implementation helpers;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_selective_scan_update_kernel` for maximum GPU throughput.
Single step SSM update: state = state * exp(A*dt) + B*dt*x; out = state @ C + D*x.
Supports optional D residual, z gating (SiLU), dt_bias, softplus, and batch indices.
Constraints:
- Must maintain the same function signature for `selective_state_update`
- Output must match reference within atol=1e-2, rtol=1e-2


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


The protected manifest requires the declared kernel symbols to remain Triton JIT
functions, including kernels originally decorated with `@triton.jit()`. Removing
the decorator is rejected before compilation. This structural check supplements
the numerical and timed-path checks; it does not by itself attest every dispatch.

Public wrappers, imports, allocations and dispatch are protected. Candidate computation
must remain in the declared Triton kernels and implementation helpers; task references
and other operator implementations are not candidate dependencies.

All five scored cases, original comparisons, seeds, 10 warmups, 100 samples and state
reset/allocation boundaries are retained. The actual captured graph returns both its
output and the updated state/cache for validation. The harness checks the measured
output and state, then perturbs operands, poisons output storage and replays the same
graph with the original preparation callback. The state reset remains outside timing.
All output/state values use the original numerical gate; readonly operands are checked.
Unobservable graph fallback fails.

`validate-task` checks independent known answers and negative comparator controls.
Correctness additionally runs the unscored public-interface controls in
`scripts/semantic_controls.py`. They never replace or change scored workloads.
