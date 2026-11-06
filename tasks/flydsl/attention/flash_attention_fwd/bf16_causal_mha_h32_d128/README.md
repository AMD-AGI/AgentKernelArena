# Flash attention forward, causal MHA, bfloat16, 32 heads, head_dim 128

Prefill attention over a batch of equal-length sequences: every query attends to all earlier
positions of its own sequence and itself, with scale `1/sqrt(128)`. Tensors use the
`flash_attn_func` layout `[batch, seq_len, heads, head_dim]`; the float64 reference chunks the
queries so even the largest case stays within memory.

- **Baseline:** the FlyDSL MFMA32 flash-attention kernel from ROCm/FlyDSL
  (`kernels/flash_attn_func.py`): `mfma_f32_32x32x16` for both GEMMs, online softmax in registers,
  probabilities fed straight to the second GEMM, XOR-swizzled LDS and DMA-to-LDS K/V prefetch on
  gfx950. For 32 heads it builds a 128-row and a 256-row query-tile variant and picks one by
  `batch * seq_len` (256 rows from 4096 tokens). The probabilities are truncated, not rounded, to
  bfloat16 before the second GEMM; that is part of the baseline's error budget under the default
  `baseline_relative` bound.
- **Baseline changes:** the kernel was previously a builder driven by a harness; `run()` caches the
  built launcher per (heads, head_dim, dtype) with `functools.lru_cache`, launches it on the current
  stream with flat views of the tensors, and the `FLYDSL_FLASH_ATTN_FUNC_*` environment tuning
  switches are fixed to their defaults. `buffer_ops` comes from the installed FlyDSL when it still
  ships that module; otherwise the bundled `baseline/buffer_ops.py`, trimmed from the same FlyDSL
  source and without environment access, is used.
- **Scenario:** the earlier task mixed float16/bfloat16 and 8-64 heads at head_dim 128; this one is
  bfloat16, 32 heads, head_dim 128. The kernel requires `seq_len % 128 == 0`, so `seq_len` is
  declared in multiples of 128. The Triton AITER `mha` and the rocBench Triton flash attention
  (float16, 48 heads, head_dim 64) of the earlier tasks are different scenarios and are not part of
  this one.
- **Workloads:** single prompts of 1k-16k tokens and batches of 4-16 prompts of 2k-8k tokens
  (`batch * seq_len` from 1024 to 131072). The shortest performance case is estimated near 13 us
  from causal flops (about 8.6 GFLOP) and must be confirmed on MI355X. Short sequences (128-2176)
  are correctness-only; (11, 384) and (17, 256) exercise the 256-row variant, including a
  partial query tile.
- **Not verified here:** the kernel needs FlyDSL and gfx950; the buffer_ops fallback is written for
  FlyDSL releases with the typed-iterator pointer API.
