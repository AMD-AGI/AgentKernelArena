# DeepSeek-V4 sparse FP8 MLA decode, C128A layer (sliding window + C128 pool)

The decode attention of a DeepSeek-V4 C128A layer, as served through the
`flash_mla_with_kvcache` interface with an FP8 KV cache, attention sinks and an
extra KV pool. Each request has one query token with 64 heads of width 512
(448 NoPE + 64 RoPE). It attends to two sets of latent KV rows, each row being
both key and value:

- the sliding window: the nonnegative slots among the first `sparse_lens[b]`
  entries of `sparse_indices[b, 0]`, in the main pool `kv_cache` (256 slots per
  page);
- the compressed history: the nonnegative slots among the first
  `extra_sparse_lens[b]` entries of `extra_sparse_indices[b, 0]`, in the C128
  pool `extra_kv_cache` (2 slots per page, one compressed entry per 128 tokens).

A slot is `page * page_size + position`; `-1` inside a prefix selects nothing and
entries past a prefix are ignored. With `logit = sm_scale * q . kv`:

```text
output[b, 0, h] = sum_s exp(logit[h, s]) kv[s] / (sum_s exp(logit[h, s]) + exp(sinks[h]))   (bfloat16)
lse[b, 0, h]    = log(sum_s exp(logit[h, s]))                                                (float32, sink excluded)
```

A request without any KV entry has a zero output and `lse = +inf`.

**Packed pools.** A pool is `[pages, page_size, 1, 584]` bytes, held in `float8_e4m3fn`
storage as `flash_mla_with_kvcache` receives it. A page holds
`page_size` payloads of 576 bytes (448 OCP E4M3 codes, then 64 bfloat16 RoPE
values), followed by `page_size` scale records of 8 bytes (7 UE8M0 exponents, one
per 64 codes, and a padding byte). NoPE value `d` decodes to
`e4m3(code[d]) * 2^(scale[d // 64] - 127)`. This is the format's definition used by
the DSv4 references in vLLM (`tests/kernels/attention/test_rocm_triton_attn_dsv4.py`)
and SGLang (`dsv4/dequant_k_cache.py`). SGLang's ROCm TileLang kernel
(`dsa/tilelang_kernel.py::dpsk_v4_fp8_attention_fwd`) instead assembles the bfloat16
bits directly. That is the same value for every normal code and differs only for
subnormal codes, whose magnitude is below 2^-6 of the group scale (under 7e-5 of
the group maximum). Production-quantized rows almost never contain such codes, so
both decodings pass.

## Baseline

`baseline/kernel.py` is vLLM's ROCm DeepSeek-V4 sparse decode
(`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` at
`edbc4969a76b25c531f7b9bb16b79984e65ef023`, Apache-2.0): the gfx942/gfx950
split-K path of `_rocm_sparse_attn_decode_ragged_triton`. It has a partial
kernel (16 heads x 32 KV entries per `tl.dot` tile, bfloat16 softmax weights, fp32
accumulators) and a reduce kernel that applies the sink, plus the gfx950-tuned
split heuristic. The modifications, marked `task:` in the source:

- The partial kernel reads the dense index tables and their lengths directly.
  Upstream's dense entry point first packs them into ragged `(indices, indptr)`
  form with `torch.clamp`/`torch.cumsum` and a pack kernel. That would be
  PyTorch computation inside `run()`, and it selects the same entries in the same
  order. The split heuristic still sees the table widths as average lengths, as
  upstream's dense path does.
- The reduce kernel also writes the log-sum-exp (sink excluded, `+inf` for an
  empty row), which `flash_mla_with_kvcache` returns.
- Both pools are fixed to OCP E4M3. The fallback kernel for un-tuned
  architectures and the vLLM platform plumbing are removed.

The earlier version of this task used SGLang's TileLang kernel as its baseline.
A TileLang baseline cannot serve as a candidate starting point or a positive
control. The SM120 Triton kernel in SGLang
(`flash_mla_sm120_triton.py`) was also considered and not used: it runs one
program per (request, head), so it re-reads every KV row 64 times without
matrix instructions; it merges the two pools and applies the sink with PyTorch
operations; and its LSE includes the sink. SGLang's ROCm Triton backend
(`nsa_triton_decode`, dual-scope path) is also a production option, but it is not
used here for three reasons. It autotunes over ten configurations. For pools
above 2 GB, which the C4A scenario's long-context pools exceed, it toggles the
global `triton.knobs.amd.use_buffer_ops`. It has a dual-scope kernel, a split-K
kernel and three combine kernels. vLLM's two gfx950-tuned kernels serve both
extra-pool scenarios unchanged at every pool size.

## Workloads

The var axes are `batch` and `ctx`, the longest context in the batch. Request
contexts are drawn between `ctx/2` and `ctx`, and row 0 has exactly `ctx`. From
its context `c`, a request's tables are modelled on how the serving metadata
builds them:

- window: the last `min(c, 128)` tokens, in a 256-slot ring page of the main pool
  (`slot = page * 256 + position % 256`);
- compressed history: all `c // 128` compressed tokens, in order, through the
  request's own extra-pool pages (`ceil(ctx / 256)` pages per request).

Pages are handed out in shuffled order, and the pools are sized to the batch.
Pool contents are production-quantized: per 64-value group,
`scale = 2^ceil(log2(amax / 448))` (as the DSv4 cache writers compute it). The
groups have different magnitudes (log2 spread of ±2), and the RoPE tail is
bfloat16.

Rows 1-10 of every batch, where present, exercise the rest of the index contract:

| Row | Pattern |
| --- | --- |
| 1 | Empty |
| 2 | `-1` holes inside both prefixes |
| 3 | Legal slots past both prefixes |
| 4 | Arbitrary, possibly repeated, slots anywhere in the pools |
| 5 | Empty window |
| 6 | Compressed prefix of length 1 whose only entry is `-1` |
| 7-10 | Window prefix cut to 1, 63, 65 or 127 entries, the rest of the window left past it, next to the unchanged compressed prefix |

Correctness cases cover the boundary lengths:

- window lengths 1-128, and a partial window (1, 63, 65, 127) next to a non-empty
  compressed prefix (rows 7-10; in production a window shorter than 128 only
  occurs before the first compressed entry, but the index contract allows it);
- compressed-entry counts 0, 1, 2, 16-33, 64-128, 256-512, 8191, 8255, and the
  full width of 8256.

Performance cases range from short-context high-concurrency decode
(1024 x 256 tokens) to 1M-token contexts (32 x 1,048,576), with batch x context
spanning 128x. Small batches move too little KV for the 10 µs timing floor
(about 27 MB), so they are correctness-only.

`sinks` are drawn uniformly from [-2, 12], the range the LSE covers from one to
8k entries, so the sink changes the output in every length regime. Queries are
standard normal.

## Comparison bounds

- `output`: `baseline_relative` with `k = 3` and `max_violation_fraction = 0.001`.
  The baseline rounds softmax weights to bfloat16 relative to each tile's running
  maximum. Its per-element error therefore depends on the tiling, and an equally
  accurate kernel with other tiles differs from it element by element. A CPU
  emulation of the baseline's arithmetic measured the share of elements outside
  the default `k = 2` bound:
  - other tile sizes or split counts: up to 6e-5;
  - SGLang's TileLang algorithm (bfloat16 partial outputs per split): about 2e-3.

  At `k = 3`, the first stays below 2e-6 and the second at about 3e-4. In every
  case that exercises it, each wrong implementation tried falls outside the bound
  on about 0.5% of elements or more, and usually on 30-90%: ignoring the sink, dropping the last entry, reading past a
  prefix, treating `-1` as slot 0, using the wrong scale group, or a 1% scale
  error.
- `lse`: `tolerance` with `atol = 1e-4` and `rtol = 1e-5`. The log-sum-exp is an
  fp32 reduction over up to 8384 terms. Its error, about 1e-6, depends on the
  summation order, so a bound relative to the baseline would reject exact
  reorderings. Semantic errors exceed this tolerance by orders of magnitude: a sink
  included in the LSE, a missing pool, a wrong scale, or a dropped entry.
