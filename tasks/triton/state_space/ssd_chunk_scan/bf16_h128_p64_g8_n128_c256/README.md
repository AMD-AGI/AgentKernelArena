# Mamba2 SSD chunk scan, bfloat16, 128 heads x 64 channels, 8 groups, state 128, chunk 256

Output stage of the chunked ("state space duality") Mamba2 prefill. Tokens are cut into chunks of 256; for
every chunk, head and token the operator adds three terms:

```text
out[m, h, :] = exp(dA[m]) * (S_prev @ C[m, g])                          # carried state, decayed
             + sum_{k <= m} cb[m, k] * exp(dA[m] - dA[k]) * dt[k] * x[k, h, :]   # causal in-chunk mixing
             + D[h] * x[m, h, :]                                        # skip
```

with `g = h // 16`, `S_prev = states[c-1, h]` (zero for the first chunk of a sequence). `cb` is the chunk's
`C B^T` Gram matrix, `dA` the chunk-local cumulative decay and `states` the per-chunk states produced by the
chunk state and state passing stages; here they are inputs. The z gate and initial states of the production
kernel are off: vLLM's Mamba2 mixer passes no z to the scan (the gate is applied in its gated norm) and uses
initial states only for prefix-cached continuations.

- **Origin.** vLLM's `_chunk_scan_fwd_kernel`
  (`vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py`, derived from `mamba_ssm`). The shape
  (128 heads of 64 channels, 8 groups, state 128, chunk 256) follows hybrid Mamba2 models such as
  Nemotron-H; the earlier AgentKernelArena task of this operator used a few tiny shapes and float16.
- **Baseline.** The vLLM Triton kernel unchanged apart from inlining `fast_exp`. The upstream autotune (it tunes
  on the first call, which is not capturable and keys on constants only, so it would pick one tile shape for all
  token counts) is replaced by one fixed configuration (tile 64 x 64 x 32, 4 warps, 2 stages), and the launch is
  specialised to this scenario. The fixed tile configuration was chosen without GPU timing; it is a valid but
  not necessarily the fastest upstream configuration.
- **Inputs.** `cb`, `x` and `C` are standard-normal-like (`cb` with std 0.32), `states` has std 0.03 and `D` is
  U(0.05, 0.25), so the carried-state, in-chunk and skip terms have comparable magnitude (about 0.15 each) and an
  error in any term is visible. `D` is smaller than a trained model's (about 1) for that reason. `dt` is
  U(0, 0.1) and `dA_cumsum` its running sum with a per-head negative rate in (0.02, 0.2), computed by
  `inputs.py` (constant `dA_cumsum` in the padding of a partial chunk, as the chunk cumsum stage produces). The
  padding slots of `dt` keep their U(0, 0.1) draw instead of the cumsum stage's zeros: the operator only sums
  over `k <= m` for valid rows `m`, so a correct kernel never uses them (the baseline loads them but multiplies
  them by the masked, zero `x` rows), a candidate cannot rely on them being zero, and the 1-token case is not
  255/256 exact zeros (which the QA input audit rejects as a degenerate draw). The valid-token values are the
  same as with zero padding. `cu_chunk_seqlens` are consecutive 256-token chunks; a new sequence (state reset)
  starts every 8 chunks (2048 tokens).
- **Workloads.** 1024 to 131072 tokens (128x span; about 55 KB of traffic per token, so the smallest case is
  estimated at about 10 to 12 us). Smaller counts are correctness-only: 1 token (axis minimum), 100 (one partial
  chunk), 1000 (non-power-of-two, not a multiple of 16) and 2051 (a partial chunk in a second sequence).
- **Comparison.** Default `baseline_relative` bound against a float64 reference. The baseline rounds the
  decayed `cb` tile and the carried state to bfloat16 before its dots, so its own error, and hence the bound, is
  at the bfloat16 level.
- **Scope.** The earlier task also ran float16 inputs with float32 output and, in unscored controls, the
  optional initial states, z gate and per-channel D. This scenario fixes the serving configuration: bfloat16
  output, a per-head D, and no z or initial states (those two are separate production paths and would be
  sibling scenarios). Sequence resets, partial chunks and grouped heads stay covered by the workloads.
