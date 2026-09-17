# triton_mean

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton mean reduction kernel `mean_kernel` for maximum
GPU throughput while maintaining numerical correctness.

The kernel computes the mean along a single dimension by viewing the input as (M, N, K)
where N is the reduction dimension, with each program computing one output element.

Key optimization opportunities:
- Block size tuning for different reduction sizes
- Parallel reduction strategies
- Memory access coalescing

Constraints:
- Must maintain the same function signature for `mean_dim`
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


The original five FP16 shape/dimension cases and atol=rtol=1e-2 gate remain.
Protected checks require the complete output shape, requested dtype, input
device and finite values, using a pristine-input FP32 accumulation reference.
An extra unscored 2x3x5 arithmetic input checks a reduction tail, negative dim,
keepdim=True and FP32 output override through the existing public interface.

The original full wrapper, seeds, 10 warmups and 100 samples are preserved.
Actual captured outputs and a poisoned replay after changing input values must
match the same reference gate; read-only input is checked and restored even if
replay fails. Added checks run outside timing and do not replace scored cases.

The public `mean_dim` wrapper must actually launch the declared `mean_kernel`
through the Triton runtime. Merely defining an unused JIT kernel or replacing it
with a Python object is insufficient. Host allocation, same-device casts/copies
and views are allowed; PyTorch reductions, arithmetic and library computation
(including `torch.mean` and matrix multiplication) are rejected inside candidate
initialization and wrapper execution. The independent numerical reference runs
outside this restriction.

The runtime audit surrounds each actual correctness and benchmark wrapper call,
including initialization/warmup and graph capture. It observes native launch
hooks and runtime object identity without inserting GPU operations. Captured
graph replays retain their original GPU work; the existing output and perturbed
input replay checks still establish numerical correctness. The same audit applies
to the frozen initial baseline and candidate. This is a backend execution
contract, not a Python security sandbox.
