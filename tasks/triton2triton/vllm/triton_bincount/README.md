# triton_bincount

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton bincount kernel that computes prompt token bitmasks and output token bin counts for penalty computation using atomic operations.

Constraints:
- Must maintain the same function signature for `bincount`
- Output must match reference within atol=1e-2, rtol=1e-2 for float outputs or exactly for integer outputs


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

The scored implementation region is the declared `_bincount_kernel` launch.
As in the original benchmark, its output buffers are zeroed before each sample,
outside device timing, and the launch uses 10 warmups, 100 samples and the original
20 ms target. The public `bincount` wrapper remains subject to correctness checks;
its advanced-index reset is not included in either baseline or candidate timing.
This reports kernel latency rather than complete wrapper latency.

Protected checks compare both exact integer outputs against CPU histograms and
bit packing from pristine tokens. An unscored diagnostic covers partial/nonidentity
request mapping, untouched inactive rows, empty prompt/prefill ranges, bit31/32/64,
and a 1031-token sequence crossing the launch-block boundary. Performance checks
the actual timed buffers, then changes all four read-only inputs and validates
the same captured/eager invocation with its original zero-reset preparation.
Checks restore all input/output buffers on success or failure. The five scored
cases, seeds, integer gate, original harness and reset/timing policy are unchanged.
