# triton_flash_prefill_attention

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton flash attention prefill kernel `_fwd_kernel` for maximum
GPU throughput while maintaining numerical correctness.

The kernel implements memory-efficient flash attention for the prefill stage of
LLM inference. It processes variable-length packed sequences with support for:
- Grouped-query attention (GQA) and multi-query attention (MQA)
- Causal masking
- Bidirectional sliding window masking
- Online softmax with exp2 for numerical stability

Key optimization opportunities:
- Tile size tuning (BLOCK_M, BLOCK_N) for the target GPU
- Memory access pattern optimization (coalescing, prefetching)
- Warp scheduling and occupancy tuning (num_warps, num_stages)
- Reducing unnecessary masking overhead for common cases
- Loop unrolling and instruction-level parallelism

Constraints:
- Must maintain the same function signature for `context_attention_fwd`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- The kernel must handle arbitrary sequence lengths and head dimensions


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
The controls cover ragged packed sequences, a nondefault softmax scale, causal
attention and noncausal bidirectional windows. The independent FP32 reference
applies causal/window masks before softmax and compares every output element.
