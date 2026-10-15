# triton_awq_dequantize

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton AWQ dequantization kernel `awq_dequantize_kernel` for maximum
GPU throughput while maintaining numerical correctness.

The kernel unpacks 4-bit AWQ quantized weights, applies zero-point subtraction
and scale multiplication to produce dequantized float16 weights.

Key optimization opportunities:
- Kernel tiling within the protected launch grid
- Memory access pattern optimization
- Warp scheduling and occupancy tuning

Constraints:
- Must maintain the same function signature for `awq_dequantize_triton`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- Packed weights, packed zero points and scales are read-only


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

Only `awq_dequantize_kernel` and permitted new implementation helpers in
`source/triton_awq_dequantize.py` are editable. Triton compiler decorators and
autotune configurations remain available under the shared definition policy.
The public wrapper, supported group-size table, imports, output allocation and
launch boundary are protected, along with the reference and all evaluation code.

The protected `_arena_checks.py` validates the exact captured output against the
original unpacking reference with `atol=1e-2, rtol=1e-2`. Correctness and performance
snapshot all three input buffers and compute the reference before candidate
execution. Exact read-only checks reject input changes even when the output is
otherwise correct, and restore the original buffers on success or failure.
After timing, the checker changes
the packed weights (including the sign/high bits), packed zeros and scales,
computes a new pristine reference, poisons the output, and validates the same
graph's replay and read-only inputs. Shape, dtype, device and finiteness are
checked as well. Snapshots, references, checks and restoration run outside timing
with the same policy for baseline and candidate. Original timed inputs, seeds,
cases and timing parameters are unchanged. Unobservable event fallback fails explicitly.

Additional unscored public-branch controls from PR105: FP32/BF16 scale dtypes and signed packed/tail group boundaries.
These use explicit `control-upstream-*` manifest rows. Original scored inputs,
numerical gates, seeds, warmups and sample counts remain unchanged.
