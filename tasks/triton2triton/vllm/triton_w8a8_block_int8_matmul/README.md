# triton_w8a8_block_int8_matmul

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton W8A8 block INT8 matmul kernel `_w8a8_block_int8_matmul`
for maximum GPU throughput.

The kernel performs block-wise quantized INT8 matrix multiplication where both
A and B have per-block scaling factors.

Constraints:
- Must maintain the same function signature for `w8a8_block_int8_matmul`
- Output must match reference within atol=1e-1, rtol=1e-1


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


## Complete outputs and measured replay

Correctness checks every output element, shape, dtype and device
against the original CPU block-dequantization/matmul oracle, using pristine
operands and scales. All four input tensors must remain unchanged. The original
`atol=rtol=1e-1` rule is retained. Unscored controls add a contiguous higher-rank
A, partial K/N blocks, a 64-by-64 scale block, zero A and FP32/BF16 output.
The INT8 wrapper retains its contiguous A/B requirement.

The five scored shapes, seeds and input magnitudes remain unchanged, including
the larger performance scales/operands. Ten warmups and 100 samples use explicit
device-Event timing for both roles, invoking the original allocating public
operator for every measured sample. Graph `after_sample` can inspect a captured
output after each replay, but replay would not rerun this wrapper and a batched
graph could leave earlier returns unchecked. The protected wrapper checks every
measured output against the reference, then changes A/B/As/Bs, poisons the final
output and checks an eager rerun of the same callable. These checks are outside
timing. Event times are not directly comparable with historical graph-replay
times.
Input buffers are verified read-only and restored even on replay failure.
NaN and numerical mismatches remain failures. The original `torch.allclose`
semantics permit signed FP16 infinity only at exactly the same locations and
with the same signs as the independent reference; large original performance
scales can exceed FP16's finite range. Those reference-overflow counts are
recorded explicitly. Extra, missing or wrong-sign infinities fail, and all
finite elements still use the unchanged `atol=rtol=1e-1` rule. The check does not
substitute an easier timing workload or a different output dtype.

Performance check failures also emit the underlying exception to stderr before
the original harness records its failing timing sentinel. Failed cases remain
failures; diagnostic output does not substitute for a valid measurement.
