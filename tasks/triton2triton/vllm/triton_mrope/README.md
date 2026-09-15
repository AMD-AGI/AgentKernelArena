# triton_mrope

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton multimodal rotary position embedding kernel `_triton_mrope_forward`
for maximum GPU throughput while maintaining numerical correctness.

The kernel applies rotary embeddings with 3-way multimodal sections (T/H/W) for
Qwen2VL-style models. It processes q and k tensors in-place, applying cos/sin
rotation with section-based masking.

Key optimization opportunities:
- Memory access pattern optimization
- Reducing redundant loads
- Vectorized operations
- Warp-level optimizations

Constraints:
- Must maintain the same function signature for `triton_mrope`
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


For these workloads, cos/sin have shape `[3, num_tokens, rotary_dim // 2]`.
Protected checks validate both complete FP16 q/k outputs, including non-rotary
features, at the original `atol=rtol=1e-2`. The contiguous buffers used here must
be updated in place and returned as a pair. Coefficient tensors and section
metadata remain read-only. Original non-interleaved cases retain their original
FP32 reference. Unscored 17-token, 3-query/2-key-head cases cover padded head
counts, changed section sizes, and the interleaved T/H/W coefficient-selection
mode with full rotary dimension. Interleaving selects coefficient axes; it does
not change the half-split rotation pairing. These controls do not claim every
combination of optional layouts and dimensions.

Performance continues to launch the original raw kernel, rather than the public
wrapper, with its original q_tmp/k_tmp preparation callback. Both actual
`TimedRun` outputs are checked. After timing, q/k seeds and cos/sin are changed,
temporary outputs are poisoned, and the same prepared replay is checked against
the reference. Preparation replaces poison with the input seeds required by this
in-place operation. All six buffers are restored in `finally`, including failure
paths. Original five scored cases, seeds 42+i in both phases, allocation/reset
boundaries, 10 warmups and 100 samples remain unchanged.
