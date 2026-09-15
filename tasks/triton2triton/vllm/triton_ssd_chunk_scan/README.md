# triton_ssd_chunk_scan

The starting candidate is implemented Triton. Optimize the declared Triton functions in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_chunk_scan_fwd_kernel` for maximum GPU throughput.
For each chunk, applies causal attention: iterates over chunk positions,
accumulates state with decay, produces output. Core SSD scan operation.
Constraints:
- Must maintain the same function signature for `chunk_scan_fwd`
- Output must not contain NaN values


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


## Protected semantic and replay controls

The five original scored cases, input generation, seeds, tolerances, ten warmups,
100 benchmark repetitions and canonical graph timing policy are unchanged.
Their performance results describe those original configurations; optional modes
are checked for correctness without adding or reweighting scored cases.

`scripts/semantic_controls.py` adds unscored public-interface checks for causal prefix sums; grouped heads; continuing and reset sequences; nonzero per-sequence initial states; scalar-per-head and per-dimension D residuals; z/SiLU gating; partial chunks and the complete supplied output buffer.
`validate-task` verifies the CPU reference against independent closed-form known
answers and confirms that the same comparator rejects a deliberately wrong dense
output. Both roles execute the unscored GPU controls during correctness, followed
by all five original cases. A failing control rejects the action. Output shape,
dtype, device, finiteness, every output member and read-only inputs are checked.

The benchmark callable still invokes the original public wrapper, including its
original allocation or supplied-output boundary. It now returns the actual output
to the canonical `TimedRun` collector. After measurement, protected code compares
all measured output elements with the original reference and tolerance. It then
changes real operand values, poisons the captured outputs and validates another
replay of the same graph against a fresh reference. Input snapshots detect
mutation during ordinary correctness, timed work and replay; all perturbations,
poisoning, reference calculations and restoration occur outside the timed window.
An unavailable observable graph fails rather than substituting an unrelated
untimed call. Evidence distinguishes `captured_graph` from any explicitly selected
`eager_callable`; this task retains its original graph-first policy.

Candidate code must implement the declared Triton computation. Protected task
references, controls, harnesses and timing helpers are evaluation dependencies,
not implementation libraries; importing them to produce candidate results is
outside the task contract. Task commands never write framework scores.

The editable scope is the named Triton kernel (and the existing `softplus` JIT
helper for cumsum), with new implementation helpers allowed. The public Python
wrapper, imports, allocation behavior and launch dispatch remain protected by
the framework's symbol-aware harness guard. Keeping an unused nominal kernel
and replacing the wrapper with a PyTorch computation is not a valid submission.
This boundary repair changes no initial kernel bytes or measured workload.
