# FP8 paged MQA logits, DeepSeek-V3.2 indexer (64 heads, head size 128, 64-token preshuffled cache blocks)

For one decode step, every query token of every sequence scores the cached keys of its sequence with the
DeepSeek-V3.2 lightning indexer. For sequence `b`, query token `n` of `next_n`, and key `t` of its context
(`context_lens[b]` tokens, token `t` in slot `t % 64` of cache block `block_tables[b, t // 64]`):

```text
logits[b * next_n + n, t] = sum_h weights[b * next_n + n, h]
                              * relu(dot(q[b, n, h, :], key[b, t, :]) * key_scale[b, t])      t <= context_lens[b] - next_n + n
logits[...] = -inf                                                                              every other column, up to max_model_len
```

`q` and the key codes are `float8_e4m3fn`; `key_scale` is the float32 dequantization scale stored with the key.
Accumulation and the output are float32. The logits feed a later top-k (2048 keys for DeepSeek-V3.2).

## Production origin

AITER's `deepgemm_fp8_paged_mqa_logits` with `Preshuffle=True`, `ChunkK=256`, `WavePerEU=2` and no varctx
schedule, the call vLLM's ROCm sparse-attention indexer makes for 64-token cache blocks
(`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py::rocm_fp8_paged_mqa_logits`; `get_supported_kernel_block_sizes`
is `[1, 64]` on ROCm) and that AITER's FlyDSL op test (`op_tests/test_flydsl_pa_mqa_logits_fp4.py`,
`_bench_gluon_fp8`) reproduces as the end-to-end decode convention. The interface follows DeepGEMM's
`fp8_paged_mqa_logits` (no DeepGEMM code is used). Per block the cache holds `64 * 128` fp8 codes followed by 64
float32 scales (132 bytes per token). The sibling for 1-token blocks and the pure-Triton kernel is
`tasks/triton/attention/paged_mqa_logits/fp8_h64_d128_bs1`.

## Layouts

`kv_cache[num_blocks, 64, 1, 132]` uint8, per block: 8192 data bytes, then 256 scale bytes. The task's `layout`
transform applies AITER's `shuffle_weight(layout=(16, 16))` to the data bytes of every block, as the benchmark and
the op test do before the call: the 64 token rows are cut into tiles of 16 tokens x 32 bytes and, inside a block,
the byte of (token `t`, dim `d`) sits at `(t // 16) * 2048 + (d // 16) * 256 + (t % 16) * 16 + d % 16`. The
reference reads the natural token-major order; `inputs.py` was checked on the CPU against the kernel's own
`offset_k_fixed` address arithmetic (identical gathered operand). The scales are not shuffled. `block_tables` has
`max_model_len / 64` columns of int32 block ids; unlisted entries are 0.

## Scenario and constant dimensions

- DeepSeek-V3.2 `config.json` (Hugging Face): `index_n_heads` 64, `index_head_dim` 128, `index_topk` 2048,
  `max_position_embeddings` 163840. vLLM reads the same fields in `model_executor/models/deepseek_v2.py`
  (`n_head = config.index_n_heads  # 64`, `head_dim = config.index_head_dim  # 128`), and its indexer metadata
  comment states "For DeepSeek-V3.2, the max_model_len is 163840" with 132 bytes per cached token.
- `num_heads` 64 and `head_size` 128 are those fields; `max_model_len` 163840 is the width of every logits row;
  `block_size` 64 is vLLM's ROCm block size for the preshuffled kernel.

## Workloads

Var axes: `batch_size` 1-128 decoding sequences, `next_n` 1 or 2 query tokens per sequence (2 is one MTP draft
token), `ctx` 128-114688, the longest context of the batch. The longest sequence has exactly `ctx` tokens and
the others are uniform between `max(128, ctx/2)` and `ctx` (the AITER benchmark draws contexts within +-50% of the
average), so lengths are ragged and not multiples of a block or a 256-key chunk. Each sequence owns
`ceil(ctx / 64)` distinct cache blocks scattered randomly over the cache. `ctx` stops at 114688 because the
kernel addresses the cache with 32-bit element offsets (`block * 8448` must stay below 2^31, 254 k blocks) and the
largest case holds 229 k blocks (1.94 GB).

Performance cases (9) span 229376 to 29360128 in `batch_size * next_n * ctx` (128x). Estimated baseline time per
case, from the bytes streamed (every `next_n` row reads the keys again) plus the logits prefill, at 4.5 TB/s plus
4 us: about 10 us for the smallest (two 112k-token sequences) and 0.73 ms for the largest (128 sequences at 112k
with a draft token). The smallest case is close to the 10 us floor. Tiny and
single-sequence shapes are correctness-only. Boundary correctness cases: the axis minimum (`b1_n1_c128`), a
non-power-of-two batch (`b3_n2_c1000`) and contexts that are not multiples of 16 (777, 1000, 4097).

Inputs follow the model: `q` is a standard normal bfloat16 activation quantized to fp8 per head (`amax / 448`)
and the quantization scale joins the head weight, `weights = weights_proj(x) * q_scale * head_size**-0.5 *
num_heads**-0.5`; cached keys are standard normal bfloat16 activations quantized per token (`amax / 448`). The
projection output is drawn from N(0.5, 0.5), mostly positive: trained head weights are not public, and a positive
mean gives the summed score the nonzero mean a trained indexer has, so that a global scale error (the QA
controls' 16-ulp shift) is caught by the mean-error test.

## Baseline

`baseline/pa_mqa_logits_gluon.py` vendors `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` and the helpers it
uses from AITER's `aiter/ops/triton/gluon/pa_mqa_logits.py` unchanged (MIT, `baseline/LICENSE`, the file's SPDX and
copyright header kept); the other kernels of the file are dropped, and the one aiter import
(`get_cdna_version`) is replaced by its gfx950 value, 4. `baseline/kernel.py` is the launch code of
`deepgemm_fp8_paged_mqa_logits` (`attention/pa_mqa_logits.py`) for that path; `SplitKV` is
`(max(1, CUs // (batch * next_n)) + 4) // 5 * 5 * 2` with the compute-unit count read from the device
properties (256 on MI355X).

- Upstream compiles the kernel with `triton.compile` (and an AOT metadata wrapper) and launches the compiled
  object. The audit counts launches through the Gluon JIT, so `run()` launches the same kernel as
  `kernel[grid](...)` with the same options (4 warps, `waves_per_eu` 2, 2 stages, `ARCH` gfx950, `CDNA_VERSION` 4).
  The divisibility and `pointer_range` attributes upstream sets are replaced by the JIT's own argument
  specialization; the buffer-load kernel does not depend on them for correctness.
- The varctx variant (`..._preshuffle_varctx`) needs a schedule kernel that AITER writes in Triton, and a Gluon
  baseline may launch only Gluon kernels, so the non-varctx grid is used (the convention of the AITER op test and
  of vLLM).
- The kernel writes only the chunks that cover each context (up to the next multiple of 64 keys); production
  gives it a logits buffer that was filled with -inf. The evaluator poisons the output with NaN, so `run()` begins
  with `logits.fill_(-inf)`, the same prefill AITER's own FP4 wrapper performs. `fill_` is a preparation
  operation the candidate rules allow (`audit.PREPARATION_OPS`).
- No third-party origin was found for the vendored files: neither the AITER files, the `gluon/README.md` nor the
  first commits of the kernel (`fe927e8d3`, "Feat: pa_mqa_logits performance optimization & support kv_preshuffle
  + blocksize16/64") name one. The file header is AITER's (`MIT`, Copyright AMD). Its comment about the AMD Gluon
  extension refers to a toolchain, not to code.

## Comparison

`baseline_relative` with the defaults. The baseline's error is dominated by the fp8 MFMA accumulation, as in the
sibling `tasks/triton/attention/fp8_mqa_logits` (about 50 unit roundoffs rms measured on MI355X for this
instruction), so an exactly accumulating candidate stays far inside twice that. The -inf columns must match
exactly. Every context is at least 127 keys long, so each row has many elements to estimate the baseline's own
error. No bound was tuned.
