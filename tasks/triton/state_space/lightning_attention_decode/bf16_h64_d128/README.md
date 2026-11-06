# Lightning attention decode, bf16, 64 heads, head size 128

One decode step of MiniMax-Text-01 lightning (linear) attention. Every sequence owns a float32 KV
state per head, `[128, 128]` (key dimension first). For batch entry `b`, its state slot
`s = slot_idx[b]` and head `h` with decay slope `r[h]`:

```text
state[s, h] = k[b, h] (outer) v[b, h] + exp(-r[h]) * state[s, h]      (in place)
out[b, h]   = q[b, h] . state[s, h]                                   (contraction over the key dim)
```

`q`, `k`, `v` are `[batch, 64, 128]` bfloat16; `out` is `[batch, 8192]` bfloat16 (heads
concatenated); `kv_cache` is `[batch, 64, 128, 128]` float32 and is the mutable state, checked after
the call. `slot_idx` is a random permutation of the slots, so the kernel's indirection is exercised
and slots are unique (duplicate active slots are outside the contract: the parallel update would
race). Padding slots (`-1`) exist in production but are not part of this task.

- **Origin.** `_linear_attn_decode_kernel` / `linear_decode_forward_triton` in vLLM's
  `vllm/model_executor/layers/lightning_attn.py` (the same kernel is carried by SGLang). It
  replaces two earlier tasks (an SGLang copy and a vLLM copy with small test shapes). The scenario
  is MiniMax-Text-01 (64 heads, head size 128), the model the kernel was written for. Production
  computes q, k, v in float32 after a SiLU in current vLLM; this task keeps the bfloat16 activations
  of both earlier tasks.
- **Inputs.** `q`, `k`, `v ~ N(0, 1)`; the state history `~ N(0, 0.1)`; `inputs.py` builds the
  slopes of `MiniMaxText01LinearAttention._build_slope_tensor` for 64 heads (0.0029 to 0.68) scaled
  by the layer factor of layer 20 of 80.
- **Baseline.** The vLLM kernel unchanged (one program per sequence, head and 32-wide value block),
  with the output written into the preallocated buffer through its strides. As in production the
  outer product `k * v` is formed in bfloat16 before it is added to the float32 decayed state, so the
  baseline's state and output errors against the float64 reference are those of one bfloat16
  rounding of that product (55.7 dB SQNR on the state in a CPU emulation). The baseline is
  therefore less accurate than a float32 implementation, not more.
- **Comparison.** `out` uses the default `baseline_relative` bound. `kv_cache` uses
  `baseline_relative` with `k = 4` and `max_violation_fraction = 1e-4` (defaults 2 and 0). In the
  rows `state[s, h, d, :]` whose key `k[b, h, d]` is a power of two (about 0.85% of rows for these
  bfloat16 draws) the bfloat16 product is exact, so there the baseline's error is only the float32
  rounding of its fused multiply-add and the row RMS collapses to the float32 noise floor. A correct
  float32 implementation that rounds `decay * state` and the sum separately, or whose `exp(-r)`
  differs from the baseline's by an ulp, exceeds twice that floor in a few of those elements. On
  MI355X a float32 PyTorch evaluation of the reference failed the default bound on `kv_cache` in 7
  of 8 cases (violation fraction 0.83e-6 to 1.31e-6, peak 2.4x at b800; `out` passed with peak
  0.33x). A CPU emulation of the baseline (bfloat16 product, fused multiply-add; it reproduces those
  audit fractions) against float32 implementations with the exact product, on 16 and 24 sequences
  (17M and 25M state elements), gave, all violations inside the exact-product rows:

  | decay vs. baseline | `k = 2` violation fraction | `k = 4` violation fraction (peak) |
  | --- | --- | --- |
  | equal | 0 to 1.3e-6 | 0 (peak up to 0.97x) |
  | 1 ulp off | 3.9e-5 to 5.8e-5 | 1.1e-6 to 3.9e-6 (up to 2.0x) |
  | 2 ulp off | 3.5e-4 | 6.4e-5 to 7.1e-5 (up to 3.7x) |
  | 3 ulp off | 7.9e-4 to 8.1e-4 | 2.6e-4 to 2.7e-4 (rejected) |

  `k = 4` with `1e-4` accepts a float32 implementation whose decay is within 2 ulp of the
  baseline's (an `exp2`-based or fast `exp`), with or without fused multiply-add. The same
  emulation shows that, on `kv_cache` alone, it still rejects a 16-ulp shift of the baseline's
  state (violation fraction 7.6e-3, 76x the allowance), a state stored through bfloat16 (2.3e-2),
  fp8 rounding (0.65), a half-written state (0.50) and a missing decay (0.41).
- **Workloads.** Batches 8 to 800 (100x span). Every sequence reads and writes its 4 MiB state
  (8 MiB of traffic), so the smallest performance case moves about 64 MB and should take the
  baseline over 10 us; fewer sequences are correctness-only (1, 3, 37). The largest case holds
  3.4 GB of state, and its float64 reference output state 6.7 GB.
  The reference was checked against a per-sequence formula on 37 sequences.
- **Timing risk.** The kernel is bandwidth-bound; the 10 us floor is reached at about 6 sequences,
  so the smallest performance case sits close to it.
