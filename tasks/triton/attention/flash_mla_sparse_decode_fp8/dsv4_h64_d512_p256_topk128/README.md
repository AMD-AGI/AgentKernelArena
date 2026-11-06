# DSv4 sparse FlashMLA over the packed FP8 sliding-window pool (64 heads, d512, page 256, top-k 128)

DeepSeek-V4 attention reads its key/value cache from a packed FP8 pool. Each query row `b` has 64
heads of dimension 512 and an index table of 128 pool slots with a valid prefix length
`sparse_lens[b]`; the nonnegative entries of the prefix are attended, `-1` selects nothing, and entries
past the prefix must be ignored. A slot decodes to one 512-wide vector that is both key and value.
With `logit[h, s] = sm_scale * q[b, 0, h] . kv[s]` and per-head sink logits:

```text
output[b, 0, h] = sum_s exp(logit[h, s]) kv[s] / (sum_s exp(logit[h, s]) + exp(sinks[h]))   (bfloat16)
lse[b, 0, h] = log(sum_s exp(logit[h, s]))                     (float32, the sink excluded)
```

A row that selects nothing has a zero output and `lse = +inf`.

**Packed pool.** `kv_cache` is `[2019, 256, 1, 584]` float8_e4m3fn storage. A page holds 256 payloads of
576 bytes (448 E4M3 codes, then 64 bfloat16 rope values) followed by 256 scale records of 8 bytes (7
UE8M0 exponents, one per 64 codes, and a pad byte). Dimension `d < 448` of a slot is
`fp8(code) * 2^(scale - 127)`; dimensions 448-511 are the rope values unchanged. This is the
dequantization of SGLang's DSv4 cache (`dsv4/dequant_k_cache.py`); the pool is written the way its
quantizer (`dsv4/quant_k_cache.py`) writes keys: scale `2^ceil(log2(amax / 448))` per 64-wide tile.

## Production origin and scenario

SGLang's ROCm DSv4 backend (`deepseek_v4_backend_hip_radix.py`) calls `flash_mla_with_kvcache` with
one query row per token: a decode request is one row, and a prefill chunk is expanded causally so every
prompt token is a row whose sliding-window table covers its own last `min(position + 1, 128)` tokens.
The row count therefore runs from 1 (single-request decode) to the prefill chunk size. This scenario is
the main sliding-window pool alone (the compressed extra pools are other scenarios), with the source
workload's page size 256 and 2019-page pool; the current SGLang HIP backend views its window pool in
128-slot pages, which changes only the slot-to-page arithmetic.

## Baseline

SGLang's ROCm Triton backend for this call (`nsa_triton_decode`, selected by
`SGLANG_HACK_FLASHMLA_BACKEND=triton`), single-scope path: one fused kernel per (64-head block, row)
that gathers 64 slots at a time, dequantizes in registers, runs eight 64-wide `tl.dot`s for the logits
and eight for the output with probabilities rounded to bfloat16, and applies the sink and the empty-row
rules in its epilogue. `_process_kv_block_aggressive` and `_fused_gather_attn_dsv4_kernel` are unchanged.
Changes: the gfx942 FP8 probe is fixed to gfx950's OCP e4m3fn; `@triton.autotune` (BLOCK_H 16/64/128,
BLOCK_N 32/64/128, 4/8 warps) is replaced by its `BLOCK_H=64, BLOCK_N=64`, 8-warp configuration,
because the autotuner benchmarks with PyTorch statistics inside the first call, which the candidate
audit rejects; the split-K path (top-k >= 8192) and the >2 GB buffer-ops switch, neither of which
this scenario reaches, are dropped; outputs are written into the preallocated tensors.

The default ROCm backend is TileLang (`dsa/tilelang_kernel.py::dpsk_v4_fp8_attention_fwd`, the
previous task's baseline). It was not vendored: a TileLang baseline cannot be copied as a candidate
(no candidate-path positive control), and it decodes codes by assembling bfloat16 bits, which equals
the FP8 dequantization for normal codes but maps zero and subnormal codes to `2^(scale - 134) * (1 +
m/8)` instead. The SM120 Triton kernel named in the migration plan (`flash_mla_sm120_triton.py`) was not
used either: it targets NVIDIA SM120, runs one program per (row, head) without matrix instructions,
and applies the sink with PyTorch operations that also fold it into the returned LSE (an empty row
returns the sink logit instead of `+inf`).

## Inputs and workloads

`inputs.py` writes every pool slot (bfloat16 keys with varying token and tile magnitudes, quantized as
above). A case of at most 4096 rows (the largest decode batch of the source workload) is a decode
batch, one request per row, whose index tables are drawn as the source task drew them: uniform legal
slots over the whole pool. About 87% of its rows use the full 128-entry window, 12% are shorter
(sequence starts) and 1% are empty. A larger case is a prefill chunk expanded the way the ROCm backend
expands it: the chunk is split into prompts of 256-8192 tokens (half of them continue an already
cached prefix of up to 8192 tokens), each prompt's tokens occupy its own shuffled pages, and the row of
the token at position `p` lists the slots of positions `p, p - 1, ..., p - 127` (`-1` for negative
positions) with length `min(p + 1, 128)`, so neighbouring rows share 127 of their 128 slots, as in
production. In every case half of the short rows keep legal slots past their length, which must not
contribute, and one row in 16 has `-1` holes inside its prefix. Rows 0-7 are fixed boundary rows of
uniform legal slots with the source task's boundary lengths: 128, 0 (legal tail), 1 (whose only entry
is `-1`), 64, 63 and 127 (legal tails), 65 (`-1` tail) and 128 with every third entry `-1`. Queries
are standard normal; sinks are normal with standard deviation 2 so the sink term is visible in full
windows, not only in short rows.

Performance cases are decode batches of 160-4096 rows and prefill chunks of 8192 and 16384 tokens.
A decode row moves about 0.2 MB (64 KB of queries, 75 KB of KV records, 64 KB of output), so 160
rows are estimated near 11 us; a prefill row reuses its neighbours' KV records from cache and moves
about 0.13 MB, so 16384 rows are estimated near 0.5 ms. Smaller decode batches are correctness-only.
The source task's per-length cases (0, 1, 63-65, 127, 128, holes, legal tails, mixed rows) are covered
by the boundary rows and the per-row patterns of every case.

## Error bounds

**`output`** uses `baseline_relative` with `k = 5` and `max_violation_fraction = 1e-6` (defaults 2
and 0). The baseline's own error is dominated by rounding the probabilities to bfloat16 and by float32
accumulation order, and any other correct tiling of the online softmax produces independent noise of
the same size. A CPU emulation of the baseline's arithmetic, compared with the same kernel at
`BLOCK_N = 32` and `128` (both autotune choices), exceeded the bound at `k = 2` and `k = 3` (peaks
1.1-1.6x, up to 5e-5 of the elements) and reached 0.79-0.93x of it at `k = 4` over 2048 rows; at
`k = 5` the peak was 0.66-0.81x. Split-K variants (two or four partial softmax states combined in
float32, as decode kernels do for small batches) peaked at 0.65-0.66x at `k = 5`. The violation
allowance (at most 537 of the 537M output elements of the largest case, none below 1M elements) covers
the tail of the 16384-row cases without admitting a wrong (row, head), which spans 512 elements. In the
same emulation the bound rejects a dropped sink, ignored lengths, `-1` read as slot 0, a 1% scale
error, a 16-ulp output shift and FP8-rounded outputs. On MI355X a float32 PyTorch evaluation of the
reference peaked at 0.17x of this bound in every case.

**`lse`** uses `baseline_relative` with `k = 8` and `max_violation_fraction = 5e-4`. The LSE error is
the softmax-weighted mean of the logit errors plus the float32 rounding of the exponentials, the sum and
the logarithm. The logit errors come from accumulating 512-wide dot products in float32 and depend on
the accumulation order: the baseline adds bf16 products with matrix instructions over eight 64-wide
tiles, and a correct candidate that accumulates in another order has independent logit errors, which
short windows (and rows dominated by one logit) do not average out. With the earlier `k = 5` and `1e-6`,
the MI355X QA audit found that a float32 PyTorch evaluation of the reference (one float32 matrix product
for the logits, then `torch.logsumexp`) failed the LSE bound in 9 of the 14 cases (violation fractions
1.9e-4 to 3.6e-4, peaks 1.06-2.43x). A CPU emulation (6656 rows from the task's input generator on a
160-page pool: 4096 decode rows and 2560 prefill rows of 128-1024-token prompts) reproduced this with
three models of the baseline's accumulation (exact 16- or 32-product groups rounded into a float32
accumulator, or a float32 chain per 64-wide tile): with `k = 5` and `1e-6`, a sequential 512-step
float32 FMA chain failed everywhere (fractions 3.3e-4 to 7.4e-4, peaks 1.5-2.6x), four-product float32
groups (a float32 matrix-instruction order) failed everywhere (0.9e-4 to 2.9e-4), and even another
grouping of the bf16 products than the baseline's failed in some runs (up to 9.3e-5). At `k = 8` those
fractions fall to at most 1.2e-4 (peak 1.82x), 3.1e-5 and 7.7e-6, and logits rounded once from exact dot
products stay below 0.6x. The allowance of 5e-4 is 4x the worst of them and above the audit probe's
worst fraction at `k = 5`, which can only shrink at `k = 8`; it admits less than one wrong row (64
elements) in every case up to 1024 rows. In the same emulation the LSE bound alone still rejects a
16-ulp shift (every element, peak 4.1-5.1x), the sink folded into the LSE, LSE in log2 units, bf16- or
FP8-rounded LSE, a 1e-4 relative error in `sm_scale` (every element), `-1` read as slot 0 (5-7% of the
elements), ignored lengths (0.2-7% of the elements, and a finite LSE for boundary row 1) and `-inf`
instead of `+inf` for an empty row.

## Not verified here

The MI355X QA audit of the previous package, which differed from this one only in the LSE bound
(`k = 5`, `max_violation_fraction = 1e-6`) and this README, passed every check, including the baseline
compile, correctness and performance runs, the 10 us timing floor and the control suite; its only
calibration warning is the LSE finding addressed above. The LSE bound change and its emulation were
checked on CPU only; this package needs a fresh audit. Risks: the fixed launch configuration is not
tuned for every row count.
