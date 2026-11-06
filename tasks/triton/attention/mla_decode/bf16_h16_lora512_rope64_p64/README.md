# MLA decode, bf16, 16 heads, kv_lora 512 + rope 64, 64-token pages

Absorbed multi-head latent attention for one decode step: each sequence has one query token whose 16
heads are 576 wide ([512 latent | 64 RoPE]); the single latent KV head lives in a paged cache of
64-token pages addressed by `block_tables`; the value is the 512-wide latent part of the same cache
row. This is the DeepSeek-V3 decode attention at tensor parallel 8 (128 heads / 8). The output is
bfloat16 `[batch, 16, 512]`. `softmax_scale` defaults to 0.1352, the DeepSeek-V3 YaRN-scaled
`192^-0.5 * mscale^2`.

## Baseline

AITER's Triton MLA decode (split-KV attention kernel plus segment reduction) vendored from
ROCm/aiter at 652f2d4b768239aefd515c929e84817681ff598d (MIT; `baseline/LICENSE`). Kept: the decode
kernels, `select_3d_config` and the launch logic. Dropped: the prefill kernel, the gfx1250 Gluon
dispatch, the unused 2-D config. Two deliberate differences from that checkout, both inherited from
the previous version of this task (triton2flydsl/aiter/mla, validated on ROCm 10): the attention loop is
a plain `range()` so the launch's two pipeline stages apply, and gfx950 launches four waves. The
production `max_seqlen_kv` argument is derived from the block-table shape (`max_blocks * page_size`),
so `run()` never reads the device. Intermediate split-segment buffers are allocated per call with
`torch.empty` (allocation only).

## Scenario and workloads

`batch` and `ctx` (the longest cached context) vary; heads, widths and page size are fixed. The
generator scatters physical pages randomly over the cache, gives the first sequence exactly `ctx`
tokens and the others uniformly between `ctx/2` and `ctx`, so full and partial last pages and ragged
rows occur in every case. Q is N(0, 0.5), the latent cache N(0, 1).

Performance cases (8) span 32768 to 4194304 batch*ctx cache tokens (128x) and stream between 38 MB
and 4.8 GB of cache (estimated baseline time 12 us to about 0.9 ms). Smaller shapes stay
correctness-only: the minimum (1, 1), an odd batch, a context one token into a second page, an unaligned
short context, and shapes whose batch (17, 45) and context (333, 777, 1000) are not multiples of 16.

## Reference and bound

`reference.py` gathers each sequence's pages and computes the attention in float64. The default
`baseline_relative` bound applies: the baseline rounds the probabilities to bfloat16 before the PV
product, which is the production numerics and is part of the baseline's own error.

## Not covered

AITER's `mla_decode_rope` (which fuses RoPE) and the FP8-cache 128-head HIP MLA task are different
operators or scenarios and would be separate tasks.

The previous task (triton2flydsl/aiter/mla) also exercised other dimensions of the same kernel:
128 and 32 query heads, latent widths of 128 and 256, 32-token pages and multi-token queries
(`num_tokens_per_seq > 1`, the causal non-decode path). This scenario fixes the DeepSeek-V3 TP8 geometry
(16 heads, 512 + 64, 64-token pages, one query token); those variants are different scenarios and
would be sibling tasks (for example a 128-head TP1 scenario), not additional cases here.
