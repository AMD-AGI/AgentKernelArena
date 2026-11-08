# Paged MXFP4 MQA logits, DeepSeek-V4 FP4 indexer (64 heads, head size 128, 64-key cache blocks)

For one decode step, every query token of every sequence scores the cached keys of its sequence with the
lightning indexer, using MXFP4 queries and keys. Both are packed e2m1 vectors (two codes per byte, element `2i`
in the low nibble) with one E8M0 power-of-two scale per 32 elements: `dequant(x)[i] = e2m1(x[i]) *
2 ** (scale[i // 32] - 127)`. For sequence `b`, query token `n` of `next_n`, and key `t` (`context_lens[b]`
indexed keys, key `t` in slot `t % 64` of cache block `block_tables[b, t // 64]`):

```text
logits[b * next_n + n, t] = weight_scale * sum_h weights[b * next_n + n, h]
                              * relu(dot(dequant(q[b, n, h, :]), dequant(key[b, t, :])))      t <= context_lens[b] - next_n + n
logits[...] = -inf                                                                             every other column, up to max_seq_len
```

Accumulation and the output are float32. The logits feed a later top-k.

## Production origin

AITER's FlyDSL decode kernel `flydsl_pa_mqa_logits_fp4` (`aiter/ops/flydsl/kernels/pa_mqa_logits_fp4.py`), the
`mfma_scale_f32_16x16x128_f8f6f4` indexer that SGLang's DeepSeek-V4 backend runs on ROCm
(`kernels/ops/attention/dsv4/fp4_indexer_hip.py::aiter_fp4_paged_mqa_logits`, `block_k=256`, `kv_block_size=64`,
`num_warps=4`). The DeepSeek-V4 indexer scores the 4x-compressed key sequence (`compress_ratio=4`), so `ctx` and
`max_seq_len` count indexed (compressed) keys. The queries are quantized by AITER's `rope_rotate_activation`
(RoPE, Hadamard rotation, MXFP4 with shuffled scales); keys are written into the preshuffled paged cache by the
indexer's cache write; the schedule below is built once per step. All three are production layouts, supplied
here as inputs.

## Layouts

The generator draws natural layouts and the task's `layout` transform produces the kernel ABI (AITER's
`create_paged_preshuffle_kv_fp4` and its preshuffled query scales; `inputs.py` was checked on the CPU against the
kernel's own addressing, which reproduces the reference exactly):

- `q_fp4[batch, next_n, 64, 64]` uint8: head `h`, 16-byte chunk `kq` holds elements `32 kq .. 32 kq + 31`.
- `q_scale[batch, next_n, 1, 4, 16, 4]` uint8: the scale of head `m * 16 + r`, group `kq` is at `[.., 0, kq, r, m]`.
- `kv_cache[num_blocks, 1, 4, 64, 16]` uint8: chunk `kq` of token `t` of a block at `[block, 0, kq, t, :]`.
- `kv_scale[num_blocks, 1, 4, 64]` uint8: the scale of group `kq` of token `t = 16 j + l` at
  `[block, 0, kq, 4 l + j]` (the four 16-token tiles one thread reads are interleaved into one dword).
- `weights[batch * next_n, 64]` bfloat16; `weight_scale` = `head_size ** -0.5 * num_heads ** -0.5` = 0.0110485
  (SGLang's `weight_scale = softmax_scale * n_heads ** -0.5`).
- `block_tables[batch, max_seq_len / 64 + 4]` int32: the row carries 4 guard pages (SGLang's `_guarded_pages`
  padding, covering the kernel's one-chunk lookahead); unlisted entries and guards are 0.
- `cta_info[total_ctas, 4]` int32, rows `[batch_packed, chunk_start, chunk_count, ctx_len]`: the persistent-grid
  schedule. `inputs.py` builds it with a torch port of AITER's `_varctx_cta_info_kernel`
  (`compute_varctx_schedule`) for `total_ctas = next_n * min(batch * ceil(max_seq_len / 256), max(1024,
  4 * batch))`, SGLang's `_decode_cta_count` (one CTA per 256-key chunk until the grid reaches four CTAs per
  compute unit). The port was checked on the CPU to cover every (row, chunk) exactly once. A candidate may use it or
  schedule the work itself.

## Scenario and constant dimensions

- `num_heads` 64 and `head_size` 128: DeepSeek-V4-Flash `config.json` (Hugging Face): `index_n_heads` 64,
  `index_head_dim` 128. As corroboration of the serving choices (not config fields), vLLM's DeepSeek-V4 indexer
  (`vllm/models/deepseek_v4/attention.py`) has an MXFP4 indexer cache (`use_fp4_indexer_cache`), SGLang's
  `fp4_indexer_hip.py` fixes `_GROUP_SIZE = 32` (the MXFP4 block), `_KV_BLOCK_SIZE = 64` and the `weight_scale`
  of `dsv4/indexer.py`, and AITER's kernel defaults are 64 heads and head size 128.
- `block_size` 64 and the E8M0 group of 32: the kernel ABI above. `max_seq_len` 131072: the width of the logits
  rows, the `--ctx 131072` example of AITER's op test (512k raw tokens at 4x compression).

## Workloads

Var axes: `batch_size` 1-256 decoding sequences, `next_n` 1 or 2 (AITER's op test runs both; SGLang calls the
kernel with one), `ctx` 128-131072, the longest context of the batch. The longest sequence has exactly `ctx` keys
and the others are uniform between `max(128, ctx/2)` and `ctx` (AITER's `_make_varctx` draws between 25% and 75% of
the maximum), so lengths are ragged and not multiples of a block or chunk. Each sequence owns `ceil(ctx / 64)`
distinct cache blocks scattered randomly over the cache. The cache is limited to 2^19 blocks (`batch_size * ctx` of
at most 33.5 M keys, 2.3 GB) because the kernel forms byte offsets in 32 bits (`block * 4096` below 2^31); the
declared corner (256, 131072) is exactly that limit.

Performance cases (9) span 524288 to 67108864 in `batch_size * next_n * ctx` (128x). Estimated baseline time per
case, from the bytes the kernel streams (68 bytes per key, re-read for every `next_n` row) plus the logits prefill,
at 4.5 TB/s plus 4 us: about 11 us for the smallest (4 sequences at 128k) and 0.9 ms for the largest (256 sequences
at 128k with a draft token). The smallest case is close to the 10 us floor. Tiny
and single-sequence shapes (one 128k sequence, about 6 us) are correctness-only. Boundary correctness cases: the
axis minimum (`b1_n1_c128`), a non-power-of-two batch (`b3_n2_c1000`) and contexts that are not multiples of 16
(777, 1000, 4097).

Queries and keys are standard normal bfloat16 activations quantized to MXFP4 (`scale = 2 ** ceil(log2(amax / 6))`
per 32 elements, round to nearest e2m1), as the kernel's op test quantizes. The head weights are bfloat16 from
N(0.5, 0.5), mostly positive: trained head weights are not public, and a positive mean gives the summed score the
nonzero mean a trained indexer has, so that a global scale error (the QA controls' 16-ulp shift) is caught by the
mean-error test.

## Baseline

`baseline/pa_mqa_logits_fp4.py` vendors `build_pa_mqa_logits_fp4_module` and `compile_pa_mqa_logits_fp4` from
ROCm/aiter `aiter/ops/flydsl/kernels/pa_mqa_logits_fp4.py` unchanged (MIT, `baseline/LICENSE`, the file's SPDX and
copyright header kept) except for the imports. `baseline/kernel.py` is the launch of `flydsl_pa_mqa_logits_fp4`
for a supplied `cta_info` as `run()`, the launcher cached per shape by the vendored `lru_cache`, on the current
stream.

- Not vendored: the file's Triton schedule kernel and its host wrappers. A FlyDSL baseline may not import
  Triton, and production builds the schedule once per decode step for all layers, so it is the task input
  `cta_info`. The baseline therefore does not read `context_lens` (the schedule carries each context length);
  the reference does.
- The kernel stores only in-context keys. AITER's wrapper pre-fills the logits with -inf when it builds the
  schedule itself, so `run()` begins with `logits.fill_(-inf)` (the evaluator poisons the output with NaN).
  `fill_` is a preparation operation the candidate rules allow (`audit.PREPARATION_OPS`).
- `baseline/buffer_ops.py` is AITER's copy of FlyDSL's `kernels/common/buffer_ops.py` that the kernel imports.

Third-party origin: `buffer_ops.py` carries FlyDSL's header (`SPDX-License-Identifier: Apache-2.0`, `Copyright (c)
2025 FlyDSL Project Contributors`, `Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.`) and states its
upstream, ROCm/FlyDSL `kernels/common/buffer_ops.py` (PR 880). It is recorded in `provenance.modifications`; its
Apache-2.0 notice is `baseline/LICENSE-FlyDSL`. The kernel file has no third-party notice. No origin note was
found in the kernel's first commits (`3c0976b1b`, "[FLYDSL] Support paged mqa logits fp4 kernel").

## FlyDSL 0.3.2 adaptation

AITER pins `flydsl==0.3.0`; the evaluation image has 0.3.2. Every FlyDSL API the kernel and `buffer_ops` use was
compared with `python/flydsl` at v0.3.2 (`flyc.kernel`/`flyc.jit`, `fx.Tensor`/`Int32`/`Int64`/`Float32`/`Stream`,
`fx.Vector` and its `from_elements`/`filled`/`bitcast`/`maximumf`, `fx.make_copy_atom`, `fx.rocdl.make_buffer_tensor`
and `BufferCopy{32,64,128}b`, `fx.slice`/`logical_divide`/`make_layout`, `fx.MemRefType.get`/`LayoutType.get`/
`AddressSpace.Register`, `fx.memref_alloca`/`memref_load_vec`/`copy_atom_call`, `rocdl.mfma_scale_f32_16x16x128_f8f6f4`
with the nine-operand list, `rocdl.ds_bpermute`, `range(..., init=...)` loops): all exist with the signatures the
kernel uses. One port was needed: in `buffer_ops.buffer_load` and `buffer_store` the cache-policy operand is
passed as the keyword-only `aux` integer attribute of `rocdl.RawPtrBufferLoadOp` / `RawPtrBufferStoreOp` (the
0.3.2 op constructors take no positional aux value), and the environment-variable architecture lookup was
removed (`is_rdna_arch(None)` detects the GPU). The same two edits are in the repository's other FlyDSL tasks that
vendor AITER's `buffer_ops`; no change to the kernel algorithm.

## Comparison

`baseline_relative` with `k` = 2 and `max_violation_fraction: 1.0e-3` (no other override). fp4 and E8M0 products are
exact in float32, so the baseline's error is the accumulation of the scaled MFMA and the float32 head sum, which
is close to a float32 evaluation's own error. A correct candidate that accumulates in another order therefore
exceeds twice the baseline's per-row error on a few elements. On MI355X a float32
PyTorch evaluation of the reference fails the default bound on 12 cases, e.g. `b17_n2_c4097` at violation fraction
1.28e-4, peak 1.24x the bound. A CPU model (baseline = float32 with K split in four chunks and heads summed
sequentially, probe = float32 matmul with a tree head sum) gives 0 to 6e-5 and peaks 0.8 to 1.2x over five cases,
the same behavior. `1e-3` is about 8x the worst value measured on MI355X, leaving room for the unmeasured cases; it allows
1 element in 1000, and elements admitted must still lie within 16x the bound, so skipped or zeroed elements are
not admitted. The other controls are unaffected: zeros, half-written and no-op outputs differ at the -inf columns
(non-finite elements must match exactly, independent of the allowance); a 16-ulp scale shift of a float32 output
violates nearly every element in the model and is also rejected by the mean-error test (bias z about 0.96 against a
limit of 5 / sqrt(n), since all scores share a positive mean); fp8 rounding violates every element by up to
5e5x the bound. Both were checked with the model above at `1e-3`. `k` is unchanged.
