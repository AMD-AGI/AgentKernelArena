# FP8 paged MQA logits, DeepSeek-V3.2 indexer (64 heads, head size 128, one token per cache block)

For one decode step, every query token of every sequence scores the cached keys of its sequence with the
DeepSeek-V3.2 lightning indexer. For sequence `b`, query token `n` of `next_n`, and key `t` of its context
(`context_lens[b]` tokens, token `t` stored in cache block `block_tables[b, t]`):

```text
logits[b * next_n + n, t] = sum_h weights[b * next_n + n, h]
                              * relu(dot(q[b, n, h, :], key[b, t, :]) * key_scale[b, t])      t <= context_lens[b] - next_n + n
logits[...] = -inf                                                                              every other column, up to max_model_len
```

`q` and the key codes are `float8_e4m3fn`; `key_scale` is the float32 dequantization scale stored with the key.
Accumulation and the output are float32. The logits feed a later top-k (2048 keys for DeepSeek-V3.2).

## Production origin

AITER's `deepgemm_fp8_paged_mqa_logits`, which vLLM's ROCm sparse-attention indexer calls
(`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py::rocm_fp8_paged_mqa_logits`, `ChunkK=256`, `WavePerEU=2`,
`Preshuffle` only for block sizes above 1) and SGLang's DSA backend uses for the same model. The interface
follows DeepGEMM's `fp8_paged_mqa_logits` (no DeepGEMM code is used). The cache layout is vLLM's: per block
`block_size * 128` fp8 codes followed by `block_size` float32 scales (132 bytes per token, `head_dim + 4`).
vLLM's ROCm indexer supports cache blocks of 1 and 64 tokens (`get_supported_kernel_block_sizes`); this task
is the 1-token scenario, whose AITER kernel is the pure-Triton one. The 64-token scenario, with the
preshuffled Gluon kernel, is `tasks/gluon/attention/paged_mqa_logits/fp8_h64_d128_bs64`.

## Scenario and constant dimensions

- DeepSeek-V3.2 `config.json` (Hugging Face): `index_n_heads` 64, `index_head_dim` 128, `index_topk` 2048,
  `max_position_embeddings` 163840. vLLM reads the same fields in `model_executor/models/deepseek_v2.py`
  (`n_head = config.index_n_heads  # 64`, `head_dim = config.index_head_dim  # 128`), and its indexer metadata
  comment states "For DeepSeek-V3.2, the max_model_len is 163840" with 132 bytes per cached token.
- `num_heads` 64 and `head_size` 128 are those fields; `max_model_len` 163840 is the width of every logits row
  (vLLM sizes the logits buffer `[batch * next_n, max_model_len]`); `block_size` 1.

## Workloads

Var axes: `batch_size` 1-128 decoding sequences, `next_n` 1 or 2 query tokens per sequence (2 is one MTP draft
token; the second row of a sequence is causally masked by one key), `ctx` 128-114688, the longest context of the
batch. Context lengths: the longest sequence has exactly `ctx` tokens and the others are uniform between
`max(128, ctx/2)` and `ctx` (the AITER benchmark draws contexts within +-50% of the average the same way), so
lengths are ragged and not multiples of any tile. Each sequence owns distinct cache blocks scattered randomly over
the whole cache; unlisted block-table entries are 0, as in a serving block table. `ctx` stops at 114688 because
the baseline gathers keys with 32-bit offsets (`block * 132` must stay below 2^31, about 16.2 M cached tokens)
and the largest case holds `batch_size * ctx` = 14.7 M of them (1.9 GB).

Performance cases (9) span 229376 to 29360128 in `batch_size * next_n * ctx` (128x), from two 112k-token
sequences to 128 sequences at 112k with a draft token. Estimated baseline time per case, from the bytes the
kernel streams (every `next_n` row reads the keys again) plus the logits prefill, at 4.5 TB/s plus 4 us: about
10 us for the smallest and 0.75 ms for the largest. The smallest case is close to the 10 us floor. Single-token, tiny and single-sequence shapes (for example one 112k-token sequence, about 8 us)
are correctness-only. Boundary correctness cases: the axis minimum (`b1_n1_c128`), a non-power-of-two batch
(`b3_n2_c1000`) and contexts that are not multiples of 16 (777, 1000, 4097).

Inputs follow the model: `q` is a standard normal bfloat16 activation quantized to fp8 per head (`amax / 448`)
and the quantization scale joins the head weight, `weights = weights_proj(x) * q_scale * head_size**-0.5 *
num_heads**-0.5` as the DeepSeek indexer computes it; cached keys are standard normal bfloat16 activations
quantized per token (`amax / 448`). The projection output is drawn from N(0.5, 0.5), mostly positive: trained
head weights are not public, and a positive mean gives the summed score the nonzero mean a trained indexer has,
so that a global scale error (the QA controls' 16-ulp shift) shows up in the mean-error test instead of hiding
inside a zero-mean sum.

## Baseline

`baseline/pa_mqa_logits_kernel.py` vendors `_deepgemm_fp8_paged_mqa_logits` (the block-table kernel of
`_triton_kernels/attention/pa_mqa_logits.py`) and its `_sum_combine` helper unchanged, with the file's SPDX
and copyright header (MIT, `baseline/LICENSE`). `baseline/kernel.py` is the launch code of the pure-Triton
branch of `deepgemm_fp8_paged_mqa_logits` (`attention/pa_mqa_logits.py`) as `run()`; `SplitKV` is
`(max(1, CUs // (batch * next_n)) + 4) // 5 * 5 * 2` with the compute-unit count read from the device
properties (256 on MI355X).

- On Triton 3.5 and newer AITER compiles its Gluon kernel for this call, not this Triton kernel; the Triton
  kernel is the fallback branch of the same wrapper, so it is the Triton baseline here. The Gluon scenario of
  the same operator covers the Gluon path.
- The kernel writes only the chunks that cover each context (everything up to the next multiple of 256 keys);
  production gives it a logits buffer that was filled with -inf. The evaluator poisons the output with NaN, so
  `run()` begins with `logits.fill_(-inf)`, the same prefill AITER's own FP4 wrapper performs. `fill_` is a
  preparation operation the candidate rules allow (`audit.PREPARATION_OPS`); a candidate may instead write the
  -inf columns from its kernel.
- No third-party origin was found for the vendored files: neither the two AITER files, the directory READMEs
  nor the first commits of the files (`d36601a9d`, "[Feat] Deepgemm FP8 paged_mqa_logits") name one. They have
  AITER's SPDX header (`MIT`, Copyright AMD).

## Comparison

`baseline_relative` with the defaults. The baseline's error is dominated by the fp8 MFMA accumulation, as in the
sibling `tasks/triton/attention/fp8_mqa_logits`, whose MI355X audit measured about 50 unit roundoffs rms for this
instruction; an exactly accumulating candidate stays far inside twice that. The -inf columns must match exactly.
Every context is at least 127 keys long, so each row has many elements to estimate the baseline's own error
(the sibling needed a violation allowance only for rows of a single key). No bound was tuned.
