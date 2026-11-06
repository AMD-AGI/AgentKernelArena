# Sliding-window paged decode attention, FP8 KV cache, 16 query heads / 1 KV head, head size 128, window 256

Single-token (decode) grouped-query attention with a sliding window over a paged FP8 KV cache.
For sequence `s` with context length `L` (the decoded token is at position `L - 1`) and query head `h`,

```text
out[s, h] = softmax(scale * q[s, h] . K_w^T) . V_w,   w = max(0, L - 1 - sliding_window) .. L - 1
```

so every query attends to the keys at distance at most `sliding_window` from itself
(`sliding_window + 1` keys once `L > sliding_window`). All 16 query heads share the single KV head.
Keys and values are `float8_e4m3fn` codes; the reference dequantizes them with the per-tensor
multipliers `key_scale[0]` and `value_scale[0]` and computes in float64. The result is rounded to
bfloat16. `sliding_window` is 256 and `scale` is `128 ** -0.5` in every case.

## Production origin

The operator and baseline come from FlyDSL's sliding-window paged-attention decode kernel
(`kernels/pa_decode_swa.py` in ROCm/FlyDSL), as carried by the AgentKernelArena task
`flydsl2flydsl/pa_decode_swa_kernel`. Its geometry is fixed by the kernel: query groups of 16
heads, head size 128, 1024-token KV pages processed in 256-token tiles, a bfloat16 query and an FP8
cache with per-tensor scales. The source task measured windows of 256 and 512 and 1, 2 or 4 KV
heads (16, 32 or 64 query heads); this scenario fixes the window at 256 and one KV head per rank
(16 query heads per rank). The window and the head counts are scenario constants (the kernel is
compiled per window), so the source task's 512-window and multi-KV-head shapes belong to sibling
scenarios of this operator rather than to this task's workloads; its batch sizes and context
lengths are kept here.

The source task stored FNUZ (`float8_e4m3fnuz`) caches and converted the loaded bytes to the OCP
encoding inside the kernel on gfx950. This task targets gfx950 only and stores OCP
`float8_e4m3fn` caches, the encoding gfx950's FP8 MFMA instructions and its serving frameworks use,
so that conversion is removed from the baseline.

## Interface and layouts

- `query[num_seqs, 16, 128]` (bfloat16): one decode token per sequence.
- `key_cache[num_blocks, 1, 8, 1024, 16]`: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, d // 16, slot, d % 16]` (16-byte FP8 key vectors).
- `value_cache[num_blocks, 1, 128, 1024]`: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, d, slot]`.
- `block_tables[num_seqs, ceil(max_context_len / 1024)]`: token `t` of sequence `s` is in page
  `block_tables[s, t // 1024]`, slot `t % 1024`.
- `context_lengths[num_seqs]`: the longest sequence has `max_context_len` tokens, the others between
  half of it and all of it, as in a steady-state decode batch.
- `key_scale[1]`, `value_scale[1]`: per-tensor dequantization multipliers (absolute maximum / 448
  of the standard-normal keys and values the codes were quantized from).

A sequence keeps only the pages that hold its window (one or two); earlier block-table entries
point at the null page 0, as vLLM's V1 KV-cache manager does when sliding-window pages are freed.
Each sequence owns two pages dealt in random order, so its pages are scattered. The null page and
pages no sequence references hold NaN (vLLM keeps its null page zero, and freed pages keep stale
data; NaN makes any read of them that reaches the result visible), so an implementation must not
let them reach the result. The slots of a referenced page
outside the window (older tokens, or stale data after the last token) are finite and must not
affect the result either. The baseline reads whole 256-token tiles of referenced pages and masks
the positions outside the window.

## Workloads

The window bounds the work per sequence (two 256-token tiles), so the kernel time grows with the
batch size; the context length moves the window across tiles and pages and sets the block-table
width. The source task's batches (1, 2, 4 and 8 sequences at 1K contexts, one sequence at 2K)
take only a few microseconds and are kept as correctness-only cases, together with the boundary
shapes (a
one-token context, contexts at most one window long, windows crossing a page boundary, a 128K
context). Timed batches span 256 to 4096 concurrent sequences with contexts of 1K to 32K tokens;
their var-axis size spans 128x, the batch alone 16x. Batches of a few hundred sequences are the
smallest expected to stay above the 10 us timing floor (about 140 KB of cache, query and partial
outputs per sequence, over two launches).

## Baseline

`baseline/kernel.py` is the two-stage FlyDSL decode: `compile_pa_decode_sw` launches one
workgroup per (sequence, KV head, 256-token tile of the window), quantizes each query head to FP8
with a per-head dynamic scale, runs the QK and PV products on FP8 MFMA instructions with FP8
probabilities and writes per-tile softmax statistics and bfloat16 partial outputs;
`compile_pa_decode_sw_reduce` combines the tiles. `run()` allocates the partial buffers per call
and launches both kernels on the current stream. `buffer_ops.py`, `vector.py` and `meta.py` are the
FlyDSL helper modules the kernel was written against, bundled because the runtime's FlyDSL no longer
ships `flydsl.expr.buffer_ops`; `dpp_utils.py` holds the kernel's DPP lane-exchange helpers.

The FP8 query and probabilities are the baseline's own approximation: on CPU emulation of its
numerics against this task's inputs its output has an SQNR of about 29 dB against the float64
reference (about 27 dB even if the hardware flushed FP8-subnormal probabilities to zero), above
the default 25 dB baseline sanity floor, and the default `baseline_relative` bound applies. A candidate may compute more accurately.
