# MLA decode, bf16, 128 heads, kv_lora 512 + rope 64, token-granular cache (Gluon baseline)

Absorbed multi-head latent attention for one decode step: each sequence has one query token whose
128 heads are 576 wide ([512 latent | 64 RoPE]); the single latent KV head lives in a flat cache of
576-wide rows addressed per token through `block_tables[batch, ctx]` (page size 1); the value is the
512-wide latent part of the same cache row. The output is bfloat16 `[batch, 128, 512]`.
`softmax_scale` defaults to 0.1352, the DeepSeek-V3 YaRN-scaled `192^-0.5 * mscale^2`. The sibling
task `tasks/triton/attention/mla_decode/bf16_h16_lora512_rope64_p64` is the same operator for 16 heads
(tensor parallel 8) with a 64-token paged cache; this scenario is the single-GPU, 128-head geometry
that the Gluon kernel targets, with its page size of 1.

## Baseline

AITER's Gluon MLA decode kernel `_mla_gluon` vendored from ROCm/aiter at
652f2d4b768239aefd515c929e84817681ff598d, `aiter/ops/triton/gluon/mla_gluon.py` (MIT, `baseline/LICENSE`).
Kept verbatim: the SPDX and copyright header and the kernel body. The `bh64` regime is used: BLOCK_H =
BLOCK_N = 64, bf16 Q and KV, XCD-aware 3-D grid, three-stage async-copy pipeline.

Third-party origin: the upstream header states "Gluon MLA decode kernel originated from FlashMLA
triton kernel" (https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py, MIT,
Copyright (c) 2025 DeepSeek). Its notice is in `baseline/LICENSE-FlashMLA`, the upstream header
comment is preserved in `baseline/kernel.py`, and the origin is recorded in `provenance.modifications`.

Adaptation: the upstream wrapper picks `NUM_KV_SPLITS` in {1, 2, 4} from (batch, nhead) and, when it
exceeds 1, runs a Triton stage-2 reducer (`_mla_softmax_reducev_kernel`). A Gluon baseline may launch
only Gluon kernels, so that reducer is dropped and the scenario is chosen so the split count is 1, where
stage 1 writes the final output directly. With 128 heads (2 head blocks) the split count is 1 for batch
128 and 256 (`base_grid = 8 * 2 * batch / 8 >= 256`), so `batch` is a var axis restricted to
`{128, 256}` (`multiple_of: 128`); smaller batches would need the reducer. `run()` asserts this. The
`mla_gluon` wrapper is replaced by `run()`, which performs the same dispatch (`use_2d_view=True`,
`kv_pe_offset=512`, `kv_scale=1.0`, `WITHIN_2GB` selected from the cache size, so both the
buffer-load and the global-load paths are exercised: the cache exceeds 2 GB from batch 128, ctx 16384).
The arch assertion is replaced by `platform_support`; the XCD count is the upstream constant 8.
`q_nope`/`q_pe` are views of `q`, as in the upstream op test.

## Scenario and workloads

DeepSeek-V3 / R1 `config.json`: `num_attention_heads` 128, `kv_lora_rank` 512,
`qk_rope_head_dim` 64 (`qk_nope_head_dim` 128 plus 64 gives the 192-wide head behind the 0.1352 scale
via `rope_scaling` factor 40, `mscale_all_dim` 1). The upstream op test (`op_tests/test_mla.py`) runs
this kernel with batch 64/128/256, nhead 64/128 and context 16384, which is the basis of the batch
and context ranges.

`batch` (128 or 256) and `ctx` (the longest cached context, 256 to 32768) vary. The generator scatters
cache rows randomly over the whole cache, gives the first sequence exactly `ctx` tokens and the others
uniformly between `max(193, ctx/2)` and `ctx` tokens, so ragged rows and partial 64-token tiles occur
in every case. The floor of 193 is the kernel's requirement (`gl.assume(num_iter >= 3)`; the wrapper
asserts `min_kv_seq_len > 3 * BLOCK_N`), hence the axis minimum of 256 makes the shortest sequences
193..256 tokens. Q is N(0, 0.5), the latent cache N(0, 1), giving logit std about 1.6.

Performance cases (9) span 65536 to 8388608 batch*ctx cache tokens (128x). Cache streamed is about
56 MB to 7.2 GB (sequences average 0.75 ctx), and the kernel is compute-bound at 128 heads (upstream
reports about 560 TFLOPS), so the estimated baseline time runs from about 20 us (batch 128, ctx 512)
to about 3 ms (batch 256, ctx 32768). Six correctness-only cases cover the axis minimum (128, 256), a
non-power-of-two context (333), contexts that are not multiples of 16 (333, 777, 1000, 4097), and
batch 256 with short contexts. The largest cache plus inputs is about 10 GB.

## Reference and bound

`reference.py` gathers each sequence's rows and computes the attention in float64. The default
`baseline_relative` bound applies. The kernel rounds the probabilities to bfloat16 before the PV product
(as the Triton sibling does), which is part of the baseline's own error.

## Not covered

The 64/16-head (`bh64` nhead 64, `bh16bn64`, `bh16bn128`) regimes, multi-token (MTP) queries, the
split-KV path that needs the Triton reducer, `return_lse`, and DeepSeek-V4 sparse prefill (`has_pe=False`)
are different scenarios or operators.
