# triton_kda_gla_fwd_o

The starting candidate is implemented Triton. Improve only the declared Triton kernel symbols and permitted new implementation helpers;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `chunk_gla_fwd_kernel_o` for maximum GPU throughput
while maintaining numerical correctness.

GLA forward output computation for KDA with chunked attention.

Constraints:
- Must maintain the same function signature for `kda_gla_fwd_o`
- Output must match reference within atol=5e-2, rtol=5e-2


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original case IDs, seeds and geometry; the adapter
verifies them against the protected harness table. The scored q/v/h random draws
use amplitude 0.5 instead of the former 0.1, and A uses 0.05 instead of 0.01;
g remains at 0.01. The original small-amplitude draws remain unscored correctness
regressions. This makes an all-zero output fail the declared atol=rtol=5e-2 on
every scored seed. Historical timings from the old input distribution are not
directly comparable. Comparisons, tolerances, warmups, sample counts and graph/event
timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


The protected manifest requires the declared kernel symbols to remain Triton JIT
functions, including kernels originally decorated with `@triton.jit()`. Removing
the decorator is rejected before compilation. This structural check supplements
the numerical and timed-path checks; it does not by itself attest every dispatch.

The public Python wrapper, imports, allocations and dispatch are protected. Implement
computation in the declared Triton kernels; do not call the task reference, harness,
or another operator implementation from candidate code. The original source is frozen
independently for baseline execution.

`validate-task` checks independent small known answers and a deliberately incorrect
output against the unchanged task comparator. Correctness additionally exercises
unscored public-interface controls from `scripts/semantic_controls.py`. These controls
do not replace or add scored cases. Full output shape, dtype, device, finiteness and
read-only inputs are checked.

The public invocation still allocates the output and dispatches the kernel; the
device timer measures the GPU commands, not Python allocation time. Python
allocation/dispatch runs during graph capture; each of 100 reported samples
replays only the captured device commands after 10 warmups. `TimedRun` checks every reported sample output after its end event and
retains outputs of the actual captured graph. After measurement,
the harness checks those outputs, changes an operand, poisons outputs, and replays that
same graph against the original numerical reference. All comparisons, snapshots,
perturbations and restoration are outside the timed window. An unobservable graph
fallback fails instead of validating a different untimed invocation.

Unscored known-answer controls also exercise chunk sizes 32 and 128 with two
batches, three heads, mixed feature widths and partial final chunks. The 32-token
case crosses into a third chunk. Independent scalar state and causal-attention
sums validate complete outputs at the same 5e-2 tolerances; the five scored
64-token cases, timing and input generation remain unchanged.
