# triton_fla_fused_recurrent

The starting candidate is implemented Triton. Improve only the declared Triton kernel symbols and permitted new implementation helpers;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `fused_recurrent_gated_delta_rule_fwd_kernel` for maximum GPU throughput
while maintaining numerical correctness.

Fused recurrent gated delta rule forward: sequential state update with gating.

Constraints:
- Must maintain the same function signature for `fused_recurrent_gated_delta_rule_fwd`
- Output must match reference within atol=5e-2, rtol=5e-2


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares the five scored cases with input distribution
`strong_signal_v2`, plus explicit historical correctness cases. The adapter
verifies the distribution, history and original shape/seed table against the
protected harness. Comparisons, tolerances, warmups, sample counts and graph/event
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

The timed invocation keeps the original allocation/dispatch boundary, 10 warmups and
100 samples. `TimedRun` retains outputs of the actual captured graph. After measurement,
the harness checks those outputs, changes an operand, poisons outputs, and replays that
same graph against the original numerical reference. All comparisons, snapshots,
perturbations and restoration are outside the timed window. An unobservable graph
fallback fails instead of validating a different untimed invocation.

Each scored shape also has an unscored strong-signal check with the same seed,
keys, gates, and beta. Scaling `(v, initial_state)` by eight scales the complete
state history by eight; additionally scaling `q` by eight scales the output by
64. Each returned tensor must reject a zero fill at the original tolerance.
The same amplification is checked on the captured graph after timing. Shape,
allocation boundary and sample counts remain unchanged. Before accepting either
ordinary or actual timed results, the reference for the current scored draw must
independently reject zero fills for output and complete state history.


## Timed input distribution

`strong_signal_v2` keeps the original five seeds 42–46 and fixed
`(B,T,H,K,V)=(1,32,2,16,16)`, float32 tensors and scale 0.25. After the exact
original RNG draws, q, v and initial state are multiplied in place by 8. Keys,
gates and beta are unchanged. Thus q/v have nominal amplitude 0.8 and initial
state 0.08. The recurrence is linear in `(v, initial_state)`, so state history
scales by 8; scaling q too scales output by 64. This makes omitted output work
observable at the unchanged atol=rtol=0.05.

The five `historical1`–`historical5` correctness-only cases reproduce the exact
old weak draws through `gen_inputs(..., historical=True)`. They retain full
output and read-only-input checks and do not contribute device timings. The
original structured controls also remain. Baseline and candidate use the same
new scored distribution, protected wrapper, 10 warmups and 100 samples.

Three additional FP32 correctness-only rows exercise distinct launch geometry:

| Case ID | `(B,T,H,K,V)` | Initial state |
| --- | --- | --- |
| `control_tail_no_initial_state` | `(2,3,3,17,35)` | None |
| `control_tail_initial_state` | `(2,3,3,17,35)` | Nonzero |
| `control_decode_multitile` | `(2,1,4,64,64)` | Nonzero |

Their protected closed-form answers place q/k on the final K coordinate, use
distinct nonzero values for every batch/head/V coordinate and retain an
orthogonal state column. With decay and beta both one half, the observed state
satisfies `a_t = a_(t-1)/4 + v_t/2`; the orthogonal state decays by one half.
The answers use the closed geometric-series formula, independently of the
kernel and the token-loop references. These rows check K/V tails, multiple V
tiles, head/batch indexing, optional state and single-token decode, at the same
output/history tolerance and with read-only inputs. They add no timed cases.
The two original structured `(2,3,2,16,16)` controls remain as well. The finite
correctness domain consists of these documented cases; the wrapper's general
tensor notation describes its interface, without claiming every possible shape
or dtype is covered. Empty sequences are not part of the declared cases.

All five timed cases now use a new input distribution. Old per-case timings,
aggregate performance and reward ratios are historical and cannot be combined
with this revision. Establish a fresh baseline for comparisons within v2.
