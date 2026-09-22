# triton_gumbel_sample

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton Gumbel-max sampling kernel that adds Gumbel noise to logits and finds the argmax per block, used for random sampling with optional temperature scaling.

Constraints:
- Must maintain the same function signature for `gumbel_sample`
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


The seeded integer sampling contract is checked against a protected CPU
Philox-4x32-10/Gumbel-max oracle, rather than candidate self-consistency alone.
It preserves the original two-stage seed/position derivation and float32 uniform
conversion documented in [Triton random.py](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/language/random.py).
The CPU generator is independently tested against
[Random123 known-answer vectors](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors).
All original temperature-zero argmax, fixed-seed determinism, shape and range
checks remain. Output is int64 on the input device and token IDs compare exactly.

Unscored controls add 1031-token vocabularies, noncontiguous row strides,
nonidentity request mappings, mixed zero/nonunit temperatures, both
`apply_temperature` settings, and 64-bit seed/position high bits. The five
original cases and seeds (42+i correctness, 0 performance), 10 warmups and
100 samples remain unchanged. No distribution or tolerance is fitted to the
candidate.

The actual `TimedRun` return is checked numerically. Replay changes logits,
request mapping, temperature, seeds and positions in their existing buffers,
poisons the captured output and replays the same measured call. Inputs remain
read-only and are restored even on failure. The original allocating wrapper and
its timing boundary are preserved; the oracle executes outside device timing.
