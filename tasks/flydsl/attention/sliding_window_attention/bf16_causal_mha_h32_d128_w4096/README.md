# Sliding-window attention forward, causal MHA, bfloat16, 32 heads, head_dim 128, 4096-token window

Prefill attention over a batch of equal-length sequences in which every query attends only to the previous
`window_left + 1 = 4096` positions of its own sequence (itself included), with scale `1/sqrt(128)`. Tensors use the
`flash_attn_func` layout `[batch, seq_len, heads, head_dim]`; there is no grouped-query sharing (32 query heads, 32 KV
heads). `window_left` is a scalar input fixed to 4095 in every case. The float64 reference computes the exact banded
softmax, chunking the queries by 256 rows and slicing only the keys inside each chunk's band, so even the largest case
stays within memory.

## Scenario and config citations

An OLMo-3 sliding-attention layer. OLMo-3 interleaves sliding and full attention (three sliding layers, then one full
layer); this task is the sliding layer.

- Hugging Face `allenai/Olmo-3-1025-7B` `config.json`: `hidden_size` 4096, `num_attention_heads` 32, `num_key_value_heads` 32 (MHA),
  `sliding_window` 4096, and `layer_types` repeating three `sliding_attention` layers then one `full_attention` (32 layers).
  Corroboration in the pinned sglang `python/sglang/srt/configs/olmo3.py` (commit `887c401e151e2d1b082d74ef8769a23eef5c465c`):
  `Olmo3Config` defaults and the `(i + 1) % 4` layer rule.
- `head_dim` 128 = `hidden_size / num_attention_heads`, and the softmax scale `head_dim ** -0.5`
  (vLLM `vllm/model_executor/models/olmo3.py`, commit `edbc4969a76b25c531f7b9bb16b79984e65ef023`, which also takes the
  per-layer window from `layer_types`).
- `window_left` 4095: a causal sliding window of `sliding_window` tokens keeps the keys at distance `0 .. sliding_window - 1`,
  which vLLM's ROCm attention backend expresses as `(sliding_window - 1, 0)`
  (`vllm/v1/attention/backends/rocm_attn.py`).

The kernel is specialised for `head_dim` 128 and has no attention-sink input, so gpt-oss sliding layers (which use
sinks, see `vllm/model_executor/models/gpt_oss.py`) are not representable. Gemma-2's sliding layers apply logit
soft-capping and a `query_pre_attn_scalar` scale, which the kernel does not implement either.

## Baseline

`baseline/kernel.py` is ROCm/FlyDSL `kernels/attention/swa_gfx950.py` at tag v0.3.2
(`dd8373349e33638c8d158a9ea009b7b28598984a`): an 8-wave GQA kernel in which each wave owns a 32-row query tile
(256 query rows per workgroup), K/V stream through LDS in 64-key tiles by DMA, `mfma_f32_32x32x16_bf16` computes both
products, the softmax is online with a lazy rescale, and only the key tiles inside the band of the workgroup are visited.
The query is pre-multiplied by `log2(e) / sqrt(128)` and rounded to bfloat16, and the probabilities are rounded to
bfloat16 before the second product; both roundings belong to the baseline's error under the default `baseline_relative`
bound. A CPU emulation of that arithmetic (bfloat16 scaled query, truncated bfloat16 probabilities, float32 sums,
bfloat16 output) gives about 50 dB SQNR against the float64 reference, far above the 25 dB sanity floor.

Changes to the upstream file: the import of `_reduction_pair` from the 5600-line `flash_attn_utils.py` now points to the
sibling `reduction_helpers.py`, which holds `_bitcast_i32`, `_bitcast_f32` and `_reduction_pair` copied unchanged
(with that file's Apache-2.0 header), and `run()` was added at the end of `kernel.py`. The launcher is built per
`(batch, heads, KV heads, head_dim, window_left)` and cached with `functools.lru_cache`: the kernel bakes the batch into
its launch grid, so every distinct batch size is its own build. `run()` launches it on the current stream with flat views
of the tensors. The kernel body is unchanged.

The kernel visits whole 64-key tiles and a 256-row query tile per workgroup, so it needs `seq_len` to be a multiple of 256
(declared as `multiple_of: 256`). Upstream exercises it at sequence lengths 4096-16384 with windows 256-4096; shorter
sequences (256-2048) are correctness cases here as well.

## Provenance

- `kernel.py` is FlyDSL project code: `kernels/attention/swa_gfx950.py` of ROCm/FlyDSL, whose repository license is
  Apache-2.0 (`baseline/LICENSE-FlyDSL`). The file was added to FlyDSL in `efd1e999` ("[Kernel][MI350] Add 8 wave GQA
  sliding-window attention kernel (#894)", 2026-08-11) and has no other history. Its header is
  `# SPDX-License-Identifier: MIT` followed by `# FlyDSL GQA sliding window attention kernel`; the SPDX tag names no
  copyright holder, so there is no MIT copyright notice to reproduce. No "adapted from" or URL appears in the file; a
  comment calls the rescale logic "reference block 2/6" without naming the reference.
- `reduction_helpers.py` carries FlyDSL's `SPDX-License-Identifier: Apache-2.0` and
  `Copyright (c) 2025 FlyDSL Project Contributors`.

## Workloads

Prompt lengths of 1k-32k tokens at batch 1, and batches of 2-8 prompts of 2k-16k tokens (`batch * seq_len` from 1024 to
131072, 128x). Single prompts shorter than the window are plain causal attention and the 8k+ cases exercise the band.
Estimated baseline time from the banded flops (`4 * heads * head_dim` per query-key pair) at 1.0 PFLOP/s plus 4 us:
the smallest performance case (1, 1024) has 524k pairs, about 8.6 GFLOP, so about 13 us; the largest (8, 16384) has
58.7M pairs per sequence, about 7.7 TFLOP, so about 7.7 ms. The shortest case runs only 128 workgroups on 256 compute
units, so its real time is likely higher.

Correctness-only cases: the minimum `(1, 256)`, odd batch and sequence `(3, 768)`, batch 17 `(17, 256)`, and `(1, 4352)`
and `(2, 5120)`, whose first keys fall out of the window by 256 and 1024 positions. Q, K and V are N(0, 1): scores have
unit variance, so the attention is neither uniform nor a single spike, and an off-by-one window edge or a leaked
out-of-window key changes every output row beyond 4096 by about 1.5% (one key of roughly 4096 equal weights), several
times the baseline's own roundings.
