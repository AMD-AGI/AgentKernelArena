# triton_compute_identity

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `compute_identity_kernel` for maximum GPU throughput.
This kernel computes output[i] = sum_k(hidden_states[i] * expert_scales[i,k])
for MoE identity/zero expert handling.

Key optimization opportunities:
- Block size tuning
- Vectorized loads
- Memory coalescing

Constraints:
- Must maintain the same function signature for `compute_identity`
- Output must match reference within atol=1e-2, rtol=1e-2


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks retain the original equation, `hidden_states * sum(expert_scales)`,
and atol=1e-2, rtol=1e-2. They additionally require the returned tensor's declared
shape, dtype, device and finite values, and prevent candidate input mutation from
changing the reference. The actual timed output and captured replay after input
perturbation must satisfy that same rule. Perturbed inputs are restored in
`finally`; original cases, seeds, warmups, samples, kernel and generated helpers
remain unchanged.

Additional unscored public-branch controls from PR105: Odd token counts and signed near-cancelling top-k scales.
They have independent `control-upstream-*` manifest rows; existing scored
inputs, numerical gates and timing remain unchanged.
