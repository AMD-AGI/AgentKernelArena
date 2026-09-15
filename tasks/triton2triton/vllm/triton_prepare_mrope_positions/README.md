# triton_prepare_mrope_positions

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_prepare_mrope_positions_kernel` for maximum GPU throughput
while maintaining correctness.

The kernel prepares Multi-dimensional Rotary Position Embedding (M-RoPE) positions
for each request. For prefill requests (num_computed < prefill_len), it reads
pre-computed 3D positions from a lookup table. For decode requests, it computes
positions as orig_pos + mrope_delta. The output is a [3, num_tokens] tensor with
positions for each of the 3 M-RoPE dimensions.

Key optimization opportunities:
- Block size tuning
- Vectorized loads/stores across the 3 dimensions
- Memory coalescing
- Branch optimization for prefill vs decode paths

Constraints:
- Must maintain the same function signature for `prepare_mrope_positions`
- Output must match reference exactly (integer positions)
- Must handle both prefill and decode paths correctly


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

Mixed prefill/decode controls include request remapping and a preserved output tail. The original is_prefill=False scored cases now actually execute decode (computed=50, prefill length=10), matching their existing correctness cases; historical timings for those formerly mislabeled inputs are not comparable.

The task-local `_arena_contract.py` and `_arena_replay.py` are protected evaluation code. Original cases, seeds, tolerances, warmups, sample counts, allocations and preparation boundaries remain in `scripts/task_runner.py`. The extra `contract_controls` manifest row is correctness-only. Both the frozen baseline and candidate receive the same checks. The measured graph exposes its real outputs; an untimed replay changes a domain-valid input, recomputes the CPU oracle and restores all input buffers in `finally`. For the zero operator the replay control instead poisons its output. References and snapshots are outside device timing. Failure to observe or replay the measured invocation is an error, never an accepted timing sample.
