# triton_prepare_eagle_docode

The starting candidate is implemented Triton. Improve the declared kernel in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_prepare_eagle_docode_kernel` for maximum GPU throughput.
This kernel prepares decode-step inputs for EAGLE speculative decoding: copies
draft tokens to input IDs, copies hidden states from output to input buffers,
computes positions and seq_lens, and initializes query_start_loc for CUDA graphs.
Note: "docode" preserves the original vLLM spelling.

Key optimization opportunities:
- Block size tuning for hidden state copy
- Vectorized memory access for hidden states
- Efficient padding loop for query_start_loc

Constraints:
- Must maintain the same function signature for `prepare_eagle_decode`
- Output must match reference within atol=1e-5, rtol=1e-5 for hidden states
- Integer outputs must match exactly

Only `_prepare_eagle_docode_kernel` and new implementation helpers are editable.
The existing `prepare_eagle_decode` wrapper, imports, and launch setup are
protected by the shared symbol-scoped harness guard. Keep the wrapper's actual
Triton invocation: replacing it with PyTorch work while retaining an unused JIT
symbol does not satisfy this task. This boundary changes no input, numerical
check, allocation, or measured call.

The target must have exactly one innermost `@triton.jit` decorator. Optional
outer `@triton.autotune(...)` and `@triton.heuristics(...)` decorators may tune
the native kernel; arbitrary host launcher decorators are forbidden. Tuning
must preserve the existing mutable-buffer semantics. The task checks the
loaded object against the runtime's actual JIT/tuning classes, native launch
methods and declared source function before compile, correctness and performance
actions. A same-named Python class, subclass or host proxy does not qualify.
These loading checks run outside timing and apply equally to the frozen baseline
and candidate. They complement the shared guard; they do not replace numerical
checks or establish a general Python security sandbox.


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


## Protected evaluation controls

Controls cover position/sequence-length upper clamps and inactive padding. Hidden-state movement is a bit-exact copy, not an approximate arithmetic operation.

The task-local `_arena_contract.py` and `_arena_replay.py` are protected evaluation code. Original cases, seeds, tolerances, warmups, sample counts, allocations and preparation boundaries remain in `scripts/task_runner.py`. The extra `contract_controls` manifest row is correctness-only. Both the frozen baseline and candidate receive the same checks. The measured graph exposes its real outputs; an untimed replay changes a domain-valid input, recomputes the CPU oracle and restores all input buffers in `finally`. For the zero operator the replay control instead poisons its output. References and snapshots are outside device timing. Failure to observe or replay the measured invocation is an error, never an accepted timing sample.

Additional unscored public-branch controls from PR105: Hidden-state2049 tail and maximum-request257 second block.
These use explicit `control-upstream-*` manifest rows. Original scored inputs,
numerical gates, seeds, warmups and sample counts remain unchanged.
