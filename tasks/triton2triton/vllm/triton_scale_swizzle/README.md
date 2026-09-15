# triton_scale_swizzle

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton scale swizzle kernel `triton_scale_swizzle` for maximum
GPU throughput while maintaining numerical correctness.

The kernel rearranges tensor data from row-major to block-scaled swizzle format,
suitable for NVIDIA TMEM block scaling.

Constraints:
- Must maintain the same function signature for `triton_mx_block_rearrange`
- Output must match reference exactly (bit-exact for uint8 data)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


## Byte layout and actual timed replay

The protected check compares every byte and the output shape, dtype and device
against the original layout oracle, calculated from a pristine input copy.
Inputs must remain read-only. In addition to the five original scored shapes,
unscored 129-by-5 controls cover zero-padding, signed-byte input and FP8 byte
encodings. This operation rearranges bytes: NaN encodings are preserved exactly,
not interpreted as floating-point numbers. Input remains contiguous as required
by the original wrapper; output has the padded 128-by-4 block shape.

All original seeds, 10 warmups, 100 samples and timing boundaries are retained.
The public wrapper and its output allocation remain the measured unit. After
timing, the actual captured output is compared, input bytes are XOR-perturbed,
and every output byte is poisoned with the complement of its expected value
before the exact measured graph is replayed and checked. Checks and poisoning
are outside timing; original input bytes are restored even after an exception.
