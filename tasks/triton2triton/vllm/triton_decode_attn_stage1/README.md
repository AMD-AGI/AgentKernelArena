# triton_decode_attn_stage1

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton decode attention stage1 kernel `_fwd_kernel_stage1` for
maximum GPU throughput while maintaining numerical correctness.

The kernel implements the first stage of split-KV decode attention. Each program
instance handles one (batch, head, kv_split) and computes partial attention
over a contiguous range of KV tokens from a paged KV cache:
- Loads Q vector for current (batch, head)
- Iterates over KV tokens in the assigned split, loading K/V via paged
  Req_to_tokens mapping
- Computes Q @ K^T scaled dot-product with online softmax
- Outputs partial attention output and logsumexp for later reduction

Key optimization opportunities:
- BLOCK_N tuning for the target GPU
- Memory access coalescing for paged KV cache reads
- Warp scheduling and occupancy tuning (num_warps, num_stages)
- Reducing masking overhead
- Loop unrolling

Constraints:
- Must maintain the same function signature for `decode_att_m_fwd`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- Must handle arbitrary sequence lengths, head dimensions, and page sizes


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

The five original scored cases remain unchanged. Additional correctness-only
controls in `workloads.json` execute the same public wrapper and the same
10-warmup/100-sample unscored graph-replay checks, with concrete input shapes,
dtypes, routing and seed. Their exact manifest parameters are checked against
the protected generator. They do not contribute performance rows or change the original scoring case set.
Stage-1 controls use ragged lengths 63 and 37, reversed page routing, and both
zero and positive logit caps. The positive-cap oracle applies tanh before
softmax. Replay changes page-table data and exchanges sequence lengths.

The additional inactive-split control uses lengths 2 and 37. Upstream stage 1
intentionally leaves empty split slots untouched; those slots are initialized
to a distinct finite sentinel and checked byte-for-byte as caller-owned state.
All active output elements, including logsumexp, are poisoned before numerical
checks/replay. No inactive slot is claimed as a kernel write. The original five
cases have only active splits and still require complete output overwrites.

The official scoring domain remains exactly `perf1` through `perf5`, with their
original inputs, tolerances, timing boundaries, ten warmups and 100 samples.
Every added control declares only `correctness`. Its real measured/replayed
branch is checked inside the correctness action and reported as an explicitly
unscored diagnostic in that case's metrics. The performance action executes
only the five original workloads; neither their weights nor aggregation change.
