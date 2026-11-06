# Fused SwiGLU projection, bfloat16, K 1024, N 2048

The SwiGLU of the HSTU / generative-recommenders model: `out = silu(x @ w_gate) * (x @ w_up)` for
`x [m, 1024]` and two `[1024, 2048]` bfloat16 weights, forward only. Both products are accumulated in
float32 and the activation and product are formed in float32 before one rounding to bfloat16. The
reference is float64, chunked over rows, on the same bfloat16 inputs. Weights are drawn from
N(0, (1/32)^2) so the pre-activations are O(1) and `silu` is exercised on both sides of zero.

- **Weight layout.** The weights are declared and passed as `[k, n]` (stored pre-transposed). The upstream
  wrapper accepts nn.Linear `[n, k]` weights and transposes them on every call; that per-call copy is
  weight preparation that production does once at load time, so it is not part of this task. The old task
  mixed several `(k, n)` pairs (256..1024 by 384..2048); this scenario fixes the largest one,
  `k = 1024`, `n = 2048`. The other pairs (including non-multiples such as `k = 320`, `n = 384`) are different
  scenarios: `k` and `n` are constants of a task, so K/N tail masking is not exercised here, only the row (`m`)
  tail. Both const dims are divisible by every autotune block size by design of this production shape.
- **Baseline.** The portable Triton kernel and autotune configs of the old task (an extraction of
  generative-recommenders' `_swiglu_fwd_kernel`): one launch, a grouped-order tile schedule, two
  accumulators sharing each `x` tile. The upstream vendor-specific warp-specialized path was already
  dropped in that extraction. The old extraction does not record an upstream commit; provenance names the
  commit of that extraction. `triton.autotune` tunes on the first call per `(next_power_of_2(m), n, k)`
  during the evaluator's eager warmups, before graph capture.
- **Workloads.** Row counts 1 to 131072. Performance cases 1024 to 131072 (128x): 4.3 GFLOP of bf16
  work at 1024 rows, estimated above the 10 us floor; 512 rows is estimated below it and is
  correctness-only, as are 1, 3, 48 and 1000 rows.
- **Comparison.** Default `baseline_relative`; no override.
- **Risk.** Timing at 1024 rows is close to the floor; the autotune warmup behaviour and memory at
  131072 rows (0.5 GB output, chunked float64 reference) are unverified without a GPU.
