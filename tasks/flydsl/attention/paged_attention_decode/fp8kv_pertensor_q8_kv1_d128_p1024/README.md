# Paged attention decode, FP8 KV cache (per-tensor scales), 8 query heads / 1 KV head, head size 128, 1024-token pages

Single-token (decode) grouped-query attention over a paged FP8 KV cache with one scale per tensor.
For each sequence `s` and query head `h`,

```text
out[s, h] = softmax(softmax_scale * q[s, h] . K_s^T) . V_s
K_s[t] = key_code[t] * key_scale[0],  V_s[t] = value_code[t] * value_scale[0]
```

over the first `context_lengths[s]` cached tokens; the eight query heads share the single KV head,
`softmax_scale = 128 ** -0.5`, and the result is rounded to bfloat16. The float64 reference
dequantizes the codes exactly and is otherwise exact attention.

This is the per-tensor sibling of `paged_attention_decode/fp8kv_q8_kv1_d128_p1024` (per-token
scales). The kernel, worklist and workloads are the same; only the scale layout and the
baseline's probability quantization differ.

## Production origin

AITER's persistent ("PS") paged-attention decode path serves FP8 KV caches in 1024-token pages with
per-token or per-tensor quantization (`op_tests/test_pa_ps.py`, `pa_persistent_fwd`). This task's
baseline is the FlyDSL port of that path, `kernels/pa_decode_fp8.py` from ROCm/FlyDSL, whose
upstream test runs both quantization modes. The cache is quantized the way AITER's
`per_tensor_quant` does it: `scale = max|x| / 448` over the whole key (or value) cache and
`code = float8_e4m3fn(x / scale)`. A per-tensor FP8 cache is the common production setup, with
one key and one value scale per layer. Eight query heads per KV head and head size 128 are the
per-rank attention of GQA 8:1 models such as Llama-3.1-70B at TP 8 or Qwen3-30B-A3B at TP 4.

## Interface and layouts

- `query[num_seqs, 8, 128]` bfloat16: one decode token per sequence.
- `key_cache[pages, 1, 8, 1024, 16]` float8_e4m3fn: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, d // 16, slot, d % 16]` (16-byte key vectors).
- `value_cache[pages, 1, 64, 128, 16]` float8_e4m3fn: element `d` of the token in `(page, slot)` is
  at `[page, kv_head, slot // 16, d, slot % 16]` (the PS kernel's transposed value layout).
- `key_scale[1]`, `value_scale[1]` float32: the per-tensor scales.
- `context_lengths[num_seqs]`, `kv_indptr[num_seqs + 1]`, `kv_page_indices[pages]`: token `t` of
  sequence `s` is in page `kv_page_indices[kv_indptr[s] + t // 1024]`, slot `t % 1024`. Every
  sequence owns `ceil(max_seq_len / 1024)` scattered pages and lists the first
  `ceil(context_lengths[s] / 1024)`. The longest sequence has `max_seq_len` tokens, the others between
  half of it and all of it. Slots past a sequence's length and the pages it does not list hold
  stale but finite codes; they must not reach the result.
- The worklist (`work_indptr`, `work_info`, `reduce_indptr`, `reduce_final_map`,
  `reduce_partial_map`): what AITER's `get_pa_metadata_v1` (`csrc/kernels/mla/metadata/v1_2_pa_device.cuh`,
  query length 1, one KV head) builds for 256 compute units, followed by `pa_decode_fp8.py`'s
  block-split expansion. Compute unit `c` handles work rows `work_indptr[c] .. work_indptr[c+1]-1`.
  A row `[sequence, slot, q_start, q_end, kv_start, kv_end, 0, q_head_range]` covers
  `kv_page_indices[kv_start:kv_end]` of one sequence. Its four 256-token sub-tile partials are rows
  `slot * 4 + sub_tile` of the partial buffers. Sequence group `g` merges the partial rows
  `reduce_partial_map[reduce_indptr[g]:reduce_indptr[g+1]]` into output token `reduce_final_map[g, 0]`.
  Production builds this once per decode step and shares it across all layers, so it is an input
  rather than timed work. `inputs.py` ports both the scheduler and the expansion. A candidate may
  use it or schedule the work itself.

## Workloads

The workloads match the per-token sibling: batch sizes 1-512 and contexts up to 128k. They take the
AITER PS test's batches (64, 128 = CUs / 2, 158) and contexts (to 10240) and extend them to
long-context decode. The largest case is 256 sequences of up to 64k tokens, about 12.6M cached
tokens and 3.2 GB of FP8 KV per layer. `s3_l1027` and `s81_l1027` are the batches of the upstream
FlyDSL test. Batches whose whole cache is a few hundred pages finish in a few microseconds and are
correctness-only. The smallest timed case is one 128k request. It splits the sequence across 128
compute units, and its 512 partials make the reduce stage a large part of its time.

## Baseline

`baseline/kernel.py` vendors `compile_pa_decode_ps` and its helpers unchanged from ROCm/FlyDSL
`kernels/pa_decode_fp8.py` at commit `28a18d328b4882c999864b2df2f8f9fe3fcc8b47`; `dpp_utils.py`
comes from `kernels/dpp_utils.py`. `run()` is the persistent-scheduling branch of
`pa_decode_ps_launch` with no sliding window and bf16 queries. Scales of shape `[1]` select the
kernel's per-tensor path, exactly as upstream. It launches the PS kernel once, with grid
`(256 compute units, 1, 4 sub-tiles)`, to walk the worklist, then launches the reduce. The upstream
launch calls two AITER operators that cannot be vendored as FlyDSL source:

- `get_pa_metadata_v1` (a HIP kernel) and the host-side block-split expansion build the worklist.
  Both are ported to `inputs.py`, and the result is passed in.
- `pa_reduce_v1` merges the partials. It is replaced by `compile_pa_decode_ps_reduce`, a task-written
  FlyDSL kernel with the same merge, `out = sum_i exp(lse_i - lse) * partial_i`. It runs one
  workgroup per (sequence group, query head) and one thread per head-dimension element. Each wave
  step reads 64 partial LSEs, and a running maximum carries across steps. It is likely slower than
  AITER's reduce for sequences split many ways.

`baseline/` is identical to the per-token sibling's apart from the `run()` docstring.
`buffer_ops.py`, `vector.py` and `meta.py` are FlyDSL's `python/flydsl/expr` helpers from the same
commit, used only when the installed FlyDSL no longer ships them. Three changes were made: the
descriptor flags come from the detected architecture instead of an environment variable;
`vector.py` re-exports only the two raw ops the kernel uses instead of a star import; and `meta.py`
attaches unknown MLIR locations instead of reading the Python stack.

## Accuracy and the comparison bound

The baseline quantizes each query head to FP8 (`scale = max|q| / 240`). In the per-tensor path it
also quantizes the probabilities directly to FP8 in `[0, 1]`, relative to the maximum of each
256-token sub-tile, before the PV product. A CPU emulation of this path on the task's inputs gave
29-30 dB SQNR against the exact reference; the default sanity floor is 25 dB.

The default `baseline_relative` bound is kept. It accepts an implementation that reproduces the
baseline's FP8 quantization. It also accepts one that keeps the query and the probabilities in
bfloat16 or better, which the emulation placed at 0.11-0.14 of the bound. Normalizing the FP8
probabilities per 64 tokens instead peaked at 1.2-1.5 times the bound and fails. As in the
per-token sibling, the bound cannot be widened for such variants without also accepting an output
rounded to FP8 (2.0-2.5 times the bound at the default `k`), which the QA controls must reject.

## Not verified here

The kernels need FlyDSL and gfx950. The worklist port and the reference were checked on the CPU,
and the baseline's per-tensor numerics against an emulation of the PS kernel and the reduce. The
reduce kernel and the vendored launch have not been compiled or timed. The timing of the smallest
case (one 128k request) is an estimate that the MI355X audit must confirm.
