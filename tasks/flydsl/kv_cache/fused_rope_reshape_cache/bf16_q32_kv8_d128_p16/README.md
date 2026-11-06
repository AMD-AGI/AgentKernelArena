# Fused RoPE and paged KV-cache write, bfloat16, 32 query / 8 KV heads, head size 128, block size 16

One launch applies NeoX-style (rotate-half) rotary embedding to the queries and keys of a batch,
writes the rotated queries and keys to `q_out` / `k_out`, and scatters the rotated key and the
unrotated value into a paged KV cache (flash layout `[num_blocks, block_size, kv_heads, head_dim]`)
at `slot_mapping`. cos/sin tables hold the first-half angles (`reuse_freqs_front_part`). About one
token in sixteen is padding (slot -1) and writes nothing; both caches are `mutable`, start at zero
and are compared in full, so stray writes to other slots fail.

Baseline: FlyDSL's `fused_rope_cache_kernel` (Apache-2.0), restricted to this configuration. The
fp8-cache path, non-flash layout and scale arguments are removed. Positions and slot indices are
read through the layout API from int32 storage viewed as float32 and bitcast back (the same
technique the FlyDSL top-k kernel uses), because the runtime image's FlyDSL has no raw
`buffer_ops` module. This index-load change is the only functional difference to upstream and has
been checked on a CPU only for syntax and the static candidate rules; GPU validation is pending.

Workloads: decode batches and short prefills (below ~1250 tokens, under the 10 us timing floor at
about 26 KB of traffic per token) are correctness-only. Timed cases are prefill chunks from 1280 to
131072 tokens (a 128K-token prompt), which gives the required 100x span; the cache holds 262144
slots so every case draws unique slots from a larger pool. Positions are random in the table, not
sequential, to exercise the cos/sin gather.

Compare: `q_out`, `k_out` and `key_cache` use the default bound. `value_cache` uses `exact`: values are
copied unrotated from `v`, so any difference is a defect, not rounding.
