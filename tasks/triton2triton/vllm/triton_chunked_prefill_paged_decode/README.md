# triton_chunked_prefill_paged_decode

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `kernel_paged_attention_2d` for maximum GPU throughput
while maintaining numerical correctness.

This kernel implements paged attention for the decode path of chunked prefill.
It reads from a 5D K cache [num_blocks, num_kv_heads, head_size//x, block_size, x]
and 4D V cache [num_blocks, num_kv_heads, head_size, block_size], supporting
non-standard physical block sizes and GQA.

Key features:
- 5D K cache addressing with x-factor interleaving
- 4D V cache with slot-innermost layout
- GQA support with padded query-per-kv groups
- Optional sliding window and ALiBi slopes
- Decode-only filtering via query_start_len_ptr

Constraints:
- Must maintain the same function signature for `chunked_prefill_paged_decode`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16


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

Additional scored controls are declared with concrete shapes, dtypes, sequence
lengths, optional arguments and seeds in `workloads.json`; the protected evaluator
requires those parameters to match the actual generator. All original scored
cases remain unchanged. Each added case runs the real public wrapper with the
same 0.01 gates, ten warmups, 100 samples and checked captured-graph replay.
The mixed-query control includes decode and prefill requests, non-identity pages,
ALiBi and a sliding window. With query-length filtering enabled, prefill output
slots are intentionally untouched by this decode kernel; those caller-owned
slots must preserve their finite initial values byte-for-byte. Replay exchanges
query lengths so a different row becomes active. Only required writes are
poisoned. The original FP16 source accepts `k_scale` and `v_scale` as ABI
placeholders but does not apply them; non-unit values explicitly test that
retained behavior. This task does not claim scaled or quantized KV-cache support.
