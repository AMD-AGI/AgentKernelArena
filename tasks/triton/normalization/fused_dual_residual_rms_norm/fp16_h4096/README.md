# Dual-residual RMSNorm, float16, hidden size 4096

The post-attention norm of SGLang's Grok-style layers: `inner = rmsnorm(x) * weight1`, rounded to float16;
`mid = residual + inner`, rounded to float16 and returned as the new residual; `out = rmsnorm(mid) * weight2`.
Both outputs, `out` and `mid`, are compared. The reference computes in float64 and rounds the intermediate
values to float16 exactly where the operator stores them.

Baseline: SGLang's Triton `fused_dual_residual_rmsnorm_kernel` with its default (non-autotuned) launch
configuration, one program per row holding the whole row. The kernel is copied unchanged; the launcher
is inlined into `run()` writing the preallocated outputs.

Workloads: token counts at hidden size 4096. This sibling of `bf16_h4096` covers the float16 case of the previous task. Decode-sized batches are correctness-only (below the 10 us floor); the
performance cases (1536 to 196608 tokens) cover prefill chunks and batched prefill. The upper end is
bounded by the float64 reference memory. Epsilon 1e-6 is the previous task's value; both weights are
uniform 0.5 to 1.5.

## Error bounds

`out` uses the default `baseline_relative` bound with `max_violation_fraction: 5.0e-5`. The two stages
round the intermediate value to float16, so a correct float32 design (division versus reciprocal square root,
another summation order) sometimes lands the first norm result on the other side of a rounding boundary.
Emulating the baseline's float32 arithmetic and two correct float32 designs on random rows at this width
(4096 to 6144 tokens, the task's input distribution) against the float64 reference, they exceed the
default bound for a fraction of 4e-7 to 5e-6 of the `out` elements (peak 1.8 to 2.2 times the bound, far below
the framework's 16x limit); the baseline itself never does. The allowance is about ten times that fraction, and the
controls (shifted, zeroed, partial, stale, fp8-rounded outputs) violate far more elements.

`mid` uses `sqnr` with `min_db: 90`. `mid = residual + inner` is stored after the same rounding, so the
same boundary flip changes one element of `mid` by one ulp of `inner`. When `residual` nearly cancels
`inner`, that is many ulps of `mid` (1024 times the baseline-relative bound in the same emulation, fractions
1.6e-5 to 2.1e-5), so no `k` or violation fraction can accept a correct float32 design without exceeding the 16x peak limit. The
flips are rare and sign-symmetric, so a whole-output signal-to-noise ratio is the right measure: baseline and correct float32
designs reach 106 to 107 dB in the emulation, an output shifted by 16 ulp (about 24 dB for bfloat16, 42 dB for float16)
and zeroed, partial, stale or fp8-rounded outputs fall far below 90 dB, and a candidate that skips the intermediate
rounding is also rejected.
