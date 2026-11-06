# Chunk-local cumulative sum of vector gates, float32, 32 heads x 128 channels, chunk 64

For gates `g[T, 32, 128]` (float32) the operator writes, for every head and channel, the running sum
within consecutive 64-token chunks:

```text
out[t, h, c] = sum_{u = 64*floor(t/64)}^{t} g[u, h, c]
```

The last chunk is partial when `n_tokens` is not a multiple of 64. This is the `chunk_local_cumsum`
step with vector (per-channel) gates that gated-linear-attention prefill (Kimi Delta Attention style)
runs before its chunked recurrence. Both tensors are contiguous float32.

- **Origin.** Flash-linear-attention's `chunk_local_cumsum_vector` as carried in vLLM
  (`vllm/third_party/flash_linear_attention/ops/cumsum.py`). The shapes (32 heads, 128 channels,
  chunk 64) follow the earlier AgentKernelArena task of the same operator, scaled from its tiny test
  tensor (2 x 128 x 4 x 32) to production head counts; its input was `0.1 * randn`. Here the gates are
  non-positive log-decay values, `U(-1, 0)`, so chunk sums reach about -32.
- **Baseline.** The earlier task's Triton source: each program multiplies a 64 x 64 tile with a
  lower-triangular ones matrix (`tl.dot`, no TF32) and stores it. The upstream autotune over block
  width and warps was replaced by one fixed configuration (block 64, 4 warps), so a first call never
  triggers tuning inside a graph capture. The batch dimension (always 1 in serving here) and the
  reverse mode were removed.
- **Workloads.** Token counts from 2048 to 262144 (128x span; 32 KB of traffic per token, so the
  smallest case takes about 11 us). Sub-2048 batches are correctness-only: 1 token (axis minimum),
  100 (a partial chunk) and 1000 (non-power-of-two, not a multiple of 16).
- **Comparison.** Default `baseline_relative` bound; the reference is a float64 cumulative sum.
