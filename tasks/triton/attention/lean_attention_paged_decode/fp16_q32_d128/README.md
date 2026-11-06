# Lean attention, paged decode (32 heads, head size 128, float16)

Decode attention in which each request has 16 query rows per head (the speculative-decode / grouped
query tile that lean attention was written for) and attends over its whole KV history, stored as
16-token blocks addressed through a per-head block table. `sm_scale` is 0.5. Output is float16.

**Production origin.** AITER's `lean_atten_paged` (`persistent_lean_attention_paged`), a Triton port of
Lean Attention (stream-K persistent decode) with paged KV access. The shapes follow AITER's decode
benchmark (32 heads, head size 128, 16 query rows, block size 16, contexts 512 to 32k).

**Scenario.** One head per KV head (no GQA sharing), so the cache is `[32, tokens, 128]`. `batch` requests
are packed along the token axis; `inputs.py` gives them ragged lengths (weights 1, 1.5, 2, 1, ...)
that sum to `batch * ctx_len`, and each head gets its own random permutation of the blocks. A request's
keys are the blocks the head's table assigns to it. The reference gathers them explicitly in float64.

**Workloads.** Performance cases are the decode steps whose K/V traffic (16 KiB per token) takes the
baseline longer than 10 us: 4096 to 524288 packed tokens (128x). Shorter contexts are correctness-only.
The largest case holds an 8.6 GB KV cache. Risk: the 4096-token single request moves about 67 MB of
K/V traffic (roughly 15 us at HBM bandwidth), the closest case to the 10 us floor.

**Baseline.** AITER's persistent stream-K Triton kernel, vendored. Changes: the partial-result scratch
(`Mp`, `Lp`, `Op`) and the lock array, which the old harness pre-allocated, are allocated and zeroed inside
`run()` on every call (the locks must start at zero each launch), so their cost is part of the baseline
and of every candidate. The grid is `min(CU count, tile count)` programs: the upstream split arithmetic
divides by `max_tiles_per_wg - 1` and fails when there are fewer tiles than programs, which only tiny
shapes hit.

The kernel body is the copy of AITER's kernel that the previous version of this task carried, with comments and the
module docstring removed, `host_block`/`finishing_block` written as boolean expressions, and `acc += tl.dot(p, v)`
in place of `tl.dot(p, v, acc=acc)` (same math). The launch uses `num_warps=4` as the previous task did (the AITER
wrapper passes `waves_per_eu`, i.e. 2, as `num_warps`).

**Coverage.** The previous task also ran head size 64 and 64 to 128 heads with 64-token blocks in correctness; those
are different scenarios (other head size, block size 64) and are not part of this one.
