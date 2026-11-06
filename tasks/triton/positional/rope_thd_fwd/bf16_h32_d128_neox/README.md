# Variable-length (thd) NeoX RoPE forward, bfloat16, 32 heads, head size 128

Rotary position embedding on packed variable-length activations `x[T, 32, 128]`. `cu_seqlens[9]` splits the
`T` tokens into 8 sequences; the token at position `p` inside its sequence is rotated with the angles
`freqs[p, 0, 0, :]` (64 angles, one per rotated pair, shared by both halves of the head). cos and sin are
computed from the float32 angle table inside the operator, NeoX (rotate-half) style, over the whole head,
and the result is rounded to bfloat16.

Baseline: ROCm/aiter `rope_thd_fwd` (MIT, `baseline/LICENSE`), Triton. The thd forward kernel and the
rotate helpers it uses are inlined into one file; the launch wrapper is replaced by `run()`, which fixes the
scenario's flags (NeoX, no NoPE, front-part frequency reuse, out of place). The GPT-J, NoPE and in-place
branches stay in the kernel as compile-time-dead code. The upstream launch grid is `(num_seqs, heads,
ceil(T / 32))`, with programs beyond a sequence's length exiting immediately; this is kept as is.

The previous task covered four small fixed cu_seqlens lists (1024 or 2048 tokens, 8/16/32 heads, head size
64 or 128) and compared against the AITER HIP `rope_thd` operator. This one fixes the production
32-head, head size 128 layout and varies the total token count. The old 8- and 16-head and head size 64
cases are not carried over: per-token traffic at those layouts (2 to 8 KB) puts every size below the 10 us
timing floor until the token count is unrealistically large, so they could not form a timed scenario.

Inputs: `x` is normal; `cu_seqlens` has random cut points per case (non-empty, ragged sequences; the 8-token
case is one token per sequence); `freqs` is the uncached `pos * 10000^(-j/64)` table in float32 with 262144
rows (the longest supported batch). The old task used a bfloat16 angle table; float32 is used here because
bfloat16 cannot represent angles at long positions.

Workloads: 16 KB of traffic per token, so decode batches and short prefills (below ~2000 tokens) are
correctness-only; timed cases are 2048 to 262144 tokens (128x span), up to a 256K-token long-context prefill.

Reference: float64 per sequence from the same float32 angles, returned as float32 (a float64 copy of the
largest case would need 8.6 GB; float32 is far finer than the bfloat16 output). Compare: default
`baseline_relative` bound.
