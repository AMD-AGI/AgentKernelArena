# triton_ep_scatter_2

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_fwd_kernel_ep_scatter_2` kernel for maximum GPU throughput.
This kernel scatters tokens to expert-ordered layout using atomic adds for slot
allocation. Each token is copied to its assigned expert's memory region.

Key optimization opportunities:
- Grid sizing for token parallelism
- Memory coalescing for hidden state copies
- Atomic contention reduction

Constraints:
- Must maintain the same function signature for `ep_scatter_2`
- Each token must be correctly scattered to the right expert slot


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks validate every active (token, route) assignment, including tokens
beyond the original harness's first-16 subset. Slots must be unique, in bounds
and in the assigned expert's actual token region; final atomic counters must
match initial starts plus route counts. Data copies retain the original
`atol=rtol=1e-5` comparison, with shape/dtype/device and finite-output checks.
Read-only tokens/routes, inactive index entries and unused output padding must
remain unchanged. Any legal atomic allocation order is accepted.

Unscored diagnostics cover 17 tokens, 515 hidden elements, three routes, negative
and duplicate expert IDs, noncontiguous row strides with contiguous inner
columns; a separate 8193-token case covers the public grid-loop tail.
Performance still directly launches the original raw kernel and preserves its
`prepare_fn` counter reset, allocations, seed 0, 10 warmups and 100 samples.
The actual `TimedRun` outputs/counters are checked after timing, then replayed
with changed tokens, expert IDs and corresponding prefix starts, poisoned
required outputs, and the original prepare callback. All six caller/prepare
buffers are restored in `finally`, including failure paths. All five original
scored cases and correctness seeds 42+i remain unchanged.
