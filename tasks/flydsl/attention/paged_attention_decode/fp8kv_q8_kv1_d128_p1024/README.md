# Paged attention decode, FP8 KV cache (per-token scales), 8 query heads / 1 KV head, head size 128, 1024-token pages

Single-token (decode) grouped-query attention over a paged FP8 KV cache. For each sequence `s` and
query head `h`,

```text
out[s, h] = softmax(softmax_scale * q[s, h] . K_s^T) . V_s
K_s[t] = key_code[t] * key_scale[t],  V_s[t] = value_code[t] * value_scale[t]
```

over the first `context_lengths[s]` cached tokens; the eight query heads share the single KV head,
`softmax_scale = 128 ** -0.5`, and the result is rounded to bfloat16. The float64 reference
dequantizes the codes exactly and is otherwise exact attention.

## Production origin

AITER's persistent ("PS") paged-attention decode path, which serves FP8 KV caches in 1024-token pages
with per-token quantization (`op_tests/test_pa_ps.py`, `pa_persistent_fwd`). This task's baseline is
the FlyDSL port of that path, `kernels/pa_decode_fp8.py` from ROCm/FlyDSL. Eight query heads per KV
head and head size 128 are the per-rank attention of GQA 8:1 models such as Llama-3.1-70B at TP 8 or
Qwen3-30B-A3B at TP 4. The cache is written the way AITER's `pertoken_quant` writes it: per token and
head, `scale = max|x| / 448` over the head dimension and `code = float8_e4m3fn(x / scale)`.

## Interface and layouts

- `query[num_seqs, 8, 128]` bfloat16: one decode token per sequence.
- `key_cache[pages, 1, 8, 1024, 16]` float8_e4m3fn: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, d // 16, slot, d % 16]` (16-byte key vectors).
- `value_cache[pages, 1, 64, 128, 16]` float8_e4m3fn: element `d` of the token in `(page, slot)` is
  at `[page, kv_head, slot // 16, d, slot % 16]` (the PS kernel's transposed value layout).
- `key_scale`, `value_scale[pages, 1, 1024, 1]` float32: the per-token, per-head scales.
- `context_lengths[num_seqs]`, `kv_indptr[num_seqs + 1]`, `kv_page_indices[pages]`: token `t` of
  sequence `s` is in page `kv_page_indices[kv_indptr[s] + t // 1024]`, slot `t % 1024`. Every
  sequence owns `ceil(max_seq_len / 1024)` scattered pages and lists the first
  `ceil(context_lengths[s] / 1024)`. The longest sequence has `max_seq_len` tokens, the others between
  half of it and all of it. Slots past a sequence's length and the pages it does not list hold
  stale but finite codes and scales; they must not reach the result.
- The worklist (`work_indptr`, `work_info`, `reduce_indptr`, `reduce_final_map`,
  `reduce_partial_map`): what AITER's `get_pa_metadata_v1` (`csrc/kernels/mla/metadata/v1_2_pa_device.cuh`,
  query length 1, one KV head) builds for 256 compute units, followed by `pa_decode_fp8.py`'s
  block-split expansion. Compute unit `c` handles work rows `work_indptr[c] .. work_indptr[c+1]-1`;
  a row `[sequence, slot, q_start, q_end, kv_start, kv_end, 0, q_head_range]` covers
  `kv_page_indices[kv_start:kv_end]` of one sequence, and its four 256-token sub-tile partials are rows
  `slot * 4 + sub_tile` of the partial buffers; sequence group `g` merges the partial rows
  `reduce_partial_map[reduce_indptr[g]:reduce_indptr[g+1]]` into output token `reduce_final_map[g, 0]`.
  Production builds this once per decode step and shares it across all layers, so it is an input
  rather than timed work. `inputs.py` ports both the scheduler and the expansion. A candidate may
  use it or schedule the work itself.

## Workloads

Batch sizes 1-512 and contexts up to 128k: the AITER PS test's batches (64, 128 = CUs / 2, 158) and
contexts (to 10240) extended to long-context decode. The largest case, 256 sequences of up to 64k
tokens (about 12.6M cached tokens, 3.3 GB of FP8 KV per layer), fits the KV capacity of
Qwen3-30B-A3B at TP 4 on one MI355X. `s3_l1027` and `s81_l1027` are the batches of the upstream
FlyDSL test. Batches whose whole cache is a few hundred pages finish in a few microseconds and are
correctness-only. The smallest timed case, one 128k request, splits the sequence across 128 compute
units, so its 512 partials make the reduce stage a large part of its time.

## Baseline

`baseline/kernel.py` vendors `compile_pa_decode_ps` and its helpers unchanged from ROCm/FlyDSL
`kernels/pa_decode_fp8.py` at commit `28a18d328b4882c999864b2df2f8f9fe3fcc8b47` (`dpp_utils.py` from
`kernels/dpp_utils.py`). `run()` is the persistent-scheduling branch of `pa_decode_ps_launch` (no
sliding window, bf16 queries, per-token KV scales): one PS kernel launch with grid
`(256 compute units, 1, 4 sub-tiles)` that walks the worklist, then the reduce. The upstream launch
calls two AITER operators that cannot be vendored as FlyDSL source:

- `get_pa_metadata_v1` (a HIP kernel) and the host-side block-split expansion build the worklist;
  both are ported to `inputs.py` and the result is passed in.
- `pa_reduce_v1` merges the partials. It is replaced by `compile_pa_decode_ps_reduce`, a task-written
  FlyDSL kernel with the same merge, `out = sum_i exp(lse_i - lse) * partial_i`: one workgroup per
  (sequence group, query head), one thread per head-dimension element, 64 partial LSEs read per wave
  step with a running maximum across steps. It is likely slower than AITER's reduce for sequences
  split many ways.

`buffer_ops.py`, `vector.py` and `meta.py` are FlyDSL's `python/flydsl/expr` helpers from the same
commit, used only when the installed FlyDSL no longer ships them: the descriptor flags come from the
detected architecture instead of an environment variable, `vector.py` re-exports only the two raw ops
the kernel uses instead of a star import, and `meta.py` attaches unknown MLIR locations instead of
reading the Python stack.

## Accuracy and the comparison bound

The baseline quantizes each query head to FP8 (`scale = max|q| / 240`) and the probabilities to FP8
before the PV product (normalized by the largest value scale of each 256-token sub-tile), so its
output is about 28 dB SQNR against the exact reference (estimated on the CPU; the default sanity floor
is 25 dB). The default `baseline_relative` bound is kept. It accepts two kinds of
implementation. One reproduces the baseline's FP8 quantization: the same per-head query scale and
the same per-sub-tile probability normalization. The other keeps both the query and the
probabilities in bfloat16 or better, which a CPU emulation placed at 0.14 of the bound.

A partial change fails. In the same emulation, FP8 queries with float32 probabilities peaked at
1.1-1.2 times the bound, and bfloat16 queries with FP8 probabilities at 1.15-1.5 times. An FP8
implementation that changes the granularity also fails, for example by normalizing probabilities
per 64 or 1024 tokens. Its errors are as large as the baseline's but fall at different elements,
and it needs `k` of 4-6 to pass. An output rounded to FP8, which the QA controls must reject,
already passes at `k: 6`, and at `k: 4` it is rejected by a margin of only 10-15%. The bound
therefore cannot be widened for these variants.

## Scope

The original FlyDSL test also ran 16 query heads per KV head, multi-token (MTP) decode with 2 and
4 query tokens per sequence, and a per-tensor KV scale. This task fixes 8 query heads, one query
token and per-token scales. The per-tensor cache is the sibling task
`paged_attention_decode/fp8kv_pertensor_q8_kv1_d128_p1024`. The 16-head and multi-token variants
change the interface (query group size, query rows and causal mask), so they would be separate
scenarios.

## Not verified here

The kernels need FlyDSL and gfx950. The worklist port, the reference and the baseline's numerics were
checked on the CPU against an emulation of the PS kernel and reduce, but the reduce kernel and the
vendored launch have not been compiled or timed. The timing of the smallest case (one 128k request) is
an estimate that the MI355X audit must confirm.
