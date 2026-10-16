# triton_linear_attn_decode

The starting candidate is implemented Triton. Improve only the declared kernel symbols and permitted new implementation helpers;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton linear attention decode kernel `_linear_attn_decode_kernel`
for maximum GPU throughput while maintaining numerical correctness.

The kernel performs linear attention decoding for a single token step:
1. Compute kv_outer = k[:, None] * v[None, :]
2. Apply decay: kv_state = kv_outer + exp(-slope) * kv_cache_old
3. Compute output = sum(q[:, None] * kv_state, axis=0)
4. Update KV cache in-place

Key parameters:
- D: query/key dimension (constexpr)
- BLOCK_SIZE: value dimension block size (default 32)
- Uses slot_idx for batched KV cache indexing

Constraints:
- Must maintain the same function signature for `linear_attn_decode_forward`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- The kernel must handle slot_idx=-1 for padding
- Each active batch lane owns a distinct cache slot for this single decode step.
  Active indices must be unique and in `[0, num_slots)`; repeated `-1` padding is
  allowed. Duplicate active slots would race in the parallel kernel while a
  sequential reference would apply two ordered updates, so they are outside this
  task's contract.


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


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

The retained public wrapper requires the value dimension to be divisible by
`BLOCK_SIZE`; all five scored cases satisfy this condition. A `slot_idx=-1`
lane leaves the cache untouched and does not initialize its newly allocated output
lane. Unscored padding controls therefore verify all active output lanes and every
cache slot. All scored cases have active slots and retain complete output checks.

The protected input contract requires `q/k` shaped `[B, H, 1, D]`, `v` shaped
`[B, H, 1, D_v]`, cache shaped `[num_slots, H, D, D_v]`, slope shaped `[H]`, and
int32/int64 slots shaped `[B]`, all on one device. `D` and `BLOCK_SIZE` must be
positive powers of two; `D_v` must be divisible by `BLOCK_SIZE` (default 32).
`scripts/input_contract.py` checks these preconditions before correctness launches,
reference updates, and performance capture. Slot inspection stays outside the
candidate wrapper and device timing. `workloads.json` binds the same preconditions.
The five scored cases retain their original unique `arange(B)` slots, random
draws and timing; unscored controls additionally cover remapping, unused cache
slots, padding and repeated padding. These finite cases do not claim arbitrary
sequence length, duplicate-slot serialization or unsupported geometry coverage.
