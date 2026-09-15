# triton_paged_prefix_prefill

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton paged prefix prefill attention kernel `_fwd_kernel` for
maximum GPU throughput while maintaining numerical correctness.

The kernel implements paged attention for the prefix prefill stage of LLM
inference. It reads context tokens from a paged KV cache (5D K / 4D V layout)
and new query tokens from dense tensors.

Two-phase attention:
1. Query-vs-Context: queries attend to all cached context tokens (no causal mask)
2. Query-vs-Query: queries attend to new query tokens (with causal mask)

Key data structures:
- K cache: [num_blocks, num_kv_heads, head_dim/x, block_size, x] (5D vectorized)
- V cache: [num_blocks, num_kv_heads, head_dim, block_size] (4D)
- Block table (B_Loc): maps logical block indices to physical block IDs

Key optimization opportunities:
- Tile size tuning (BLOCK_M, BLOCK_N, BLOCK_SIZE) for the target GPU
- Memory access coalescing for the 5D K cache layout
- Loop unrolling factors for cache vs request loops
- Warp scheduling and pipeline stages
- Reducing masking overhead at block boundaries
- Prefetching block table entries

Constraints:
- Must maintain the same function signature for `context_attention_fwd`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- Must handle arbitrary context lengths, query lengths, and head dimensions


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


### Protected execution and replay checks

The task computes each oracle from pristine inputs before calling the public
wrapper, checks all returned or supplied outputs (including auxiliary statistics),
and rejects changes to read-only input bytes, tensor metadata, or output/input
aliasing. Shape-only inputs are compared bytewise without assuming finite values.
The wrapper must dispatch the declared genuine Triton JIT kernel; Torch allocation,
same-device casts/copies and views are permitted, while operator computation in
Torch is rejected. References execute outside that dispatch guard. This is a
backend execution contract, not a Python security sandbox.

Performance retains the original cases, 0.01 absolute/relative gates, ten warmups,
100 samples, and shared timing helper. The actual measured graph outputs are
checked against the pristine reference. Outside timing, source data is perturbed
and every output poisoned, then the same captured graph is replayed and fully
compared against a new reference. Inputs are restored afterward. No reference,
poisoning, comparison, or extra GPU check is added inside the timed invocation.
A timing fallback that cannot expose the actual measured outputs fails closed.

Additional correctness-only controls are declared with concrete shapes, dtypes, sequence
lengths, optional arguments and seeds in `workloads.json`; the protected evaluator
requires those parameters to match the actual generator. All original scored
cases remain unchanged. Each added case runs the real public wrapper with the
same 0.01 gates and an unscored captured-replay diagnostic during correctness.
That diagnostic uses ten warmups and 100 samples; it is never a performance row.
The control covers ragged context/query lengths, non-identity physical pages, a
nondefault attention scale, and the wrapper's window or ALiBi path. The oracle
reconstructs context from actual cache bytes and independently forms the masked
attention operation; cached generator outputs cannot substitute for this check.

A second ragged control uses physical block size 24 to exercise the original
non-power-of-two dispatch; its dimensions and device timing remain explicit.

The plain-prefix context page-table load now masks tokens beyond the context
length, matching its existing masked K/V loads. This fixes an out-of-bounds
metadata read for a partial tile with non-power-of-two physical page sizes;
all other original kernel operations remain unchanged.

The official scoring domain remains exactly `perf1` through `perf5`, with their
original inputs, tolerances, timing boundaries, ten warmups and 100 samples.
Every added control declares only `correctness`. Its real measured/replayed
branch is checked inside the correctness action and reported as an explicitly
unscored diagnostic in that case's metrics. The performance action executes
only the five original workloads; neither their weights nor aggregation change.
