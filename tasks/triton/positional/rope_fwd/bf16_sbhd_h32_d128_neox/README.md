# RoPE forward, bfloat16, sequence-first layout, 32 heads, head size 128, NeoX style

Rotary position embedding of `x[S, B, H, D]` (4 sequences, 32 heads, head size 128) with the whole
head rotated in NeoX style (rotate-half). `freqs[S, 1, 1, D/2]` holds the angle of every position and
frequency pair, in float32 (standard schedule, angle = position * 10000^(-k / 64)); the same angles
serve both halves of the head (`reuse_freqs_front_part`). cos and sin are computed from the angles
inside the operator, so a candidate pays for the trigonometry as the baseline does. The reference
works in float64 from the same float32 angles.

Baseline: AITER's Triton `rope_fwd` (MIT), kernel `_rope_kernel_sbhd_fwd` and its launch logic
(32 positions per program, one program per (batch, head, position block)), inlined unchanged into
one file. The kernel also contains GPT-J and NoPE branches; `run()` selects NeoX, front-part reuse and
no NoPE part. The old tasks enumerated rotate style, frequency reuse, NoPE side and rotary
fraction as 48 correctness rows; here those are not varied; GPT-J and half-rotary NeoX are the sibling scenarios
`bf16_sbhd_h32_d128_gptj` and `bf16_sbhd_h32_d128_neox_rot64`. NoPE-first and non-reused
(full-width) frequencies are not covered. Heads were raised from the upstream test's 8 to 32 so the
smallest timed case clears the 10 us floor.

Inputs: positions start at a random offset in [1, 4096) drawn from the case generator (a decode past a
KV cache), so the angles are full-precision float32 values that vary across frequencies even for one
position, not exact zeros.

Workloads: sequence length is the only varying axis. Sequences below about 512 positions (about
64 KB of traffic per position against the 10 us timing floor) are correctness-only, as are unaligned
lengths 33, 100 and 1000. Timed cases run from 512 to 65536 positions (128x). The reference
processes positions in chunks to bound its float64 temporaries.

Compare: default `baseline_relative` bound.
