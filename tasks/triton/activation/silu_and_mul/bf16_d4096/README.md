# SiLU-and-mul, bfloat16, d = 4096

`out[t] = silu(x[t, :4096]) * x[t, 4096:]` over `x` of shape `[n_tokens, 8192]`: the gated activation
of a fused gate/up projection in a SwiGLU MLP. The SiLU is computed in float32 and truncated to
bfloat16 before the multiply with the up half, which is the numerics of the AITER kernel; the
float64 reference does not round the intermediate, and the default `baseline_relative` bound absorbs
the baseline's own two roundings.

**Baseline.** AITER's `fused_silu_mul` Triton kernel and its tuned tile dispatch (`_pick_block_n`,
`_pick_block_m`, `_pick_num_warps`), taken from `aiter/ops/triton/activation.py` and
`_triton_kernels/activation.py` at the recorded commit (MIT; `baseline/LICENSE`). The kernel is
identical to upstream; logging and argument assertions were removed. The kernel text came through the
earlier `triton2flydsl/aiter/fused_silu_mul` task, which had ported it unchanged. The optional
GPT-OSS clamp of the earlier `torch2flydsl/silu_and_mul_kernel` task was always disabled in its cases
(`limit = 0`) and is not part of this scenario.

**Workloads.** The old tasks used `n = 4096` (d = 2048) at 1 to 4096 rows, and a `[512, 8192]` model
input. This scenario fixes the d = 4096 case. At 24 KB of traffic per token, about 1100 tokens are
needed to pass the 10 us floor, so decode batches (1 to 1000 tokens) are correctness-only. Timed
cases run from 2048 to 262144 tokens (prefill chunks up to large aggregate batches; the 131072 and
262144 sizes are aggregate and not a single serving step) to satisfy the 100x span. Inputs are
`normal(0, 2)`.

**Reference memory.** The reference is evaluated in chunks of 8192 rows into a float64 output.

**Sibling scenarios.** The earlier tasks also covered float16 input and other widths (d = 512, 3200
and non-multiples of the column tile). Those are kept as `../fp16_d3200` and `../bf16_d512`.
