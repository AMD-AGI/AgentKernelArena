# Fused QKV split + NeoX RoPE, bfloat16, 32 query / 8 KV heads, head size 128

One launch splits the fused QKV projection output `qkv[T, 6144]` into `q[T, 32, 128]`, `k[T, 8, 128]`
and `v[T, 8, 128]`, and applies NeoX-style (rotate-half) rotary embedding to `q` and `k`. The cos/sin
tables hold the first-half angles (`reuse_freqs_front_part`) and rows are gathered by `positions[t]`
(random positions in a 131072-row table, so the gather is exercised). `v` is copied unrotated.

Baseline: ROCm/aiter `fused_qkv_split_qk_rope` (MIT, `baseline/LICENSE`), Triton. The kernel and the
rotate helpers it imports are inlined into one file; the launch wrapper is replaced by `run()`, which
fixes the scenario's flags (NeoX, no NoPE, front-part reuse, positions, no offsets). The GPT-J and NoPE
branches stay in the kernel as compile-time-dead code. The previous task covered 1200 flag combinations;
this one fixes the production GQA configuration (the Qwen3-style 32/8/128 head layout) and varies only
the token count.

Workloads: about 24 KB of traffic per token, so decode batches and short prefills (below ~1250 tokens)
are correctness-only; timed cases are prefill chunks from 1280 to 131072 tokens.

Compare: `q` and `k` use the default bound. `v` uses `exact`: it is a pure copy, so any difference is a
defect, not rounding. The reference computes in float64 from the same bfloat16 cos/sin tables.
