# triton_reduce_segments

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `reduce_segments` for maximum GPU throughput
while maintaining numerical correctness.

This kernel performs logsumexp-based reduction of partial attention outputs
from segmented parallel softmax. Each segment has partial output, max, and
expsum values. The kernel combines them:
  overall_max = max(segm_max[0..N])
  rescaled_expsum[i] = expsum[i] * exp(segm_max[i] - overall_max)
  overall_expsum = sum(rescaled_expsum)
  output = sum(segm_output[i] * exp(segm_max[i] - overall_max)) / overall_expsum

Constraints:
- Must maintain the same function signature for `reduce_attention_segments`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16/float32


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

Protected checks enforce the supplied output buffer's shape, dtype and device,
finite values, and the wrapper's return of that same buffer. Both correctness
and performance compare against the original independent logsumexp formula on
pristine inputs; segment data and sequence routing must remain unchanged.
The actual timed output is checked before perturbing the partial outputs, maxima
and exp sums, poisoning the output, and replaying the same timed invocation.
All input/output buffers are restored even on failure. These checks retain all
five original fully active segment workloads, seeds, 1e-2 absolute/relative
tolerances, full-wrapper timing, 10 warmups and 100 samples.

Additional unscored public-branch controls from PR105: Packed variable queries, zero denominator/extreme maxima and FP32 output.
These use explicit `control-upstream-*` manifest rows. Original scored inputs,
numerical gates, seeds, warmups and sample counts remain unchanged.
