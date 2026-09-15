# triton_ssd_bmm

The starting candidate is implemented Triton. Improve only the declared Triton kernel symbols and permitted new implementation helpers;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_bmm_chunk_fwd_kernel` for maximum GPU throughput.
Batch matrix multiply: A @ B.T within chunks with optional causal masking.
Used in Mamba/SSD for computing chunk-level attention scores.
Constraints:
- Must maintain the same function signature for `bmm_chunk_fwd`
- Output must match reference within atol=1e-1, rtol=1e-1


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


The public Python wrapper, imports, allocations and dispatch are protected. Implement
computation in the declared Triton kernels; do not call the task reference, harness,
or another operator implementation from candidate code. The original source is frozen
independently for baseline execution.

`validate-task` checks independent small known answers and a deliberately incorrect
output against the unchanged task comparator. Correctness additionally exercises
unscored public-interface controls from `scripts/semantic_controls.py`. These controls
do not replace or add scored cases. Full output shape, dtype, device, finiteness and
read-only inputs are checked.

The timed invocation keeps the original allocation/dispatch boundary, 10 warmups and
100 samples. `TimedRun` retains outputs of the actual captured graph. After measurement,
the harness checks those outputs, changes an operand, poisons outputs, and replays that
same graph against the original numerical reference. All comparisons, snapshots,
perturbations and restoration are outside the timed window. An unobservable graph
fallback fails instead of validating a different untimed invocation.
