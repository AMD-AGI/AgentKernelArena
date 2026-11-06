# Mamba2 selective state update (decode), bfloat16, 128 heads x 64 x 128

One decoding step of a Mamba2 layer for a batch of sequences. For sequence `b`, head `h`
(group `g = h // 16`), channel `p`:

```text
dt = softplus(dt[b, h] + dt_bias[h])
state[s, h, p, :] = state[s, h, p, :] * exp(A[h] * dt) + B[b, g, :] * dt * x[b, h, p]     s = state_batch_indices[b]
out[b, h, p]      = (sum_n state[s, h, p, n] * C[b, g, n] + D[h] * x[b, h, p]) * silu(z[b, h, p])
```

The state cache `[batch, 128, 64, 128]` is bfloat16 and updated in place (`mutable`); the sequences
address their slots through a permutation `state_batch_indices`. `x`, `dt`, `B`, `C`, `z` and `out` are
bfloat16; `A`, `dt_bias` and `D` are float32 per-head vectors.

- **Origin.** vLLM's Mamba2 decode path (`selective_state_update` in
  `vllm/model_executor/layers/mamba/ops/mamba_ssm.py`, kernel from Tri Dao's mamba-ssm). Dimensions are
  Nemotron-H's (128 heads, head dim 64, state size 128, 8 groups). `dt_bias`, `A` and `D` follow the
  Mamba2 initialization (`dt` in `[1e-3, 1e-1]` through the inverse softplus, `A = -exp(U(log 1, log 16))`).
- **Baseline.** The existing in-task kernel (an earlier simplified copy of vLLM's, with the same body but
  without the stochastic-rounding argument) with vLLM's default launch heuristic (4 rows, 4 warps for
  state size 128). `run()` expands `A`, `dt`, `dt_bias` and `D` to stride-0 views over the head dimension,
  exactly as vLLM's Mamba2 mixer does, which selects the kernel's `TIE_HDIM` path; the views need no
  PyTorch compute. The earlier task exercised neither `dt_bias`, softplus nor the state indirection.
- **Workloads.** Decode batches 10 to 1024 (102x). The state is the whole traffic: 2 MB read and 2 MB
  written per sequence, so batches below about 10 (1, 3, 8) and the unaligned 100 are
  correctness-only or near the 10 us floor; batch 10 is the tightest performance case (estimated about
  10 us on MI355X) and is the main timing risk for the GPU audit. The largest case holds a 2 GB state.
- **Comparison.** Default `baseline_relative` for `out` and the updated `state` (the state is checked
  against the float32 reference rounded once to bfloat16). The reference computes in float64 in steps of
  16 sequences and returns the state as float32 to keep the largest case small.
