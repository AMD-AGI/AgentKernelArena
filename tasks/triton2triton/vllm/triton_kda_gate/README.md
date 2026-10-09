# triton_kda_gate

The starting candidate is implemented Triton. Improve only the declared Triton kernel symbols and permitted new implementation helpers;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `kda_gate_fwd_kernel` for maximum GPU throughput
while maintaining numerical correctness.

KDA gate forward: -exp(A) * softplus(g + bias).

Constraints:
- Must maintain the same function signature for `fused_kda_gate`
- Output must match reference within atol=1e-3, rtol=1e-3


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
100 samples. The graph captures exactly one complete operator call per replay. The
first reported sample retains the original seeded inputs; the other 99 use distinct,
deterministic same-shape inputs. The same stream and prepared oracles are used for
baseline and candidate. Oracle construction, input loading, previous-input integrity
checks, and poisoning of the exact captured output buffer all occur before the start
Event. After each end Event, a read-only observer compares the actual output against
the oracle bound to that sample. The final input is checked explicitly. The
changed-input replay leaves the last valid output intact: a kernel that reuses it
without recomputing on the new input fails the numerical comparison. Original inputs
are restored even if a check fails. A graph/Event fallback that cannot provide the
same one-call captured-buffer observation fails. The distinct measured inputs, graph
batching, and inter-sample preparation change cache and clock conditions; previous
latencies are not directly comparable and cannot be reported as a kernel gain.
