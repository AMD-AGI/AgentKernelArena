# Fused router, sigmoid top-1 with a shared expert, bfloat16, hidden 5120, 128 experts

The Llama-4 (Maverick) router: `scores = sigmoid(x @ w)` over 128 routed experts, one expert per token
(`argmax`, lowest id on ties), plus a fused shared expert column. Output row `[selected id, 128]` with
weights `[sigmoid score, 1.0]`, as in AITER's `routing_sigmoid_top1` with `fused_shared_experts=True`
(int32 ids, float32 weights). Token counts range from decode batches to batched prefill; the router reads
10 KB of activations per token, so only batches of 4096 tokens or more are timed and smaller ones are
correctness-only (the baseline is expected to run below the 10 us floor for them).

## Baseline

ROCm/aiter `aiter/ops/triton/_triton_kernels/moe/moe_routing_sigmoid_top1_fused.py` at commit
`652f2d4b768239aefd515c929e84817681ff598d` (MIT, `baseline/LICENSE`), with the gfx950 N128 tuning table of
`aiter/ops/triton/configs/moe/gfx950-MOE_ROUTING_SIGMOID_TOPK1.json` and the M-bucket dispatch of
`aiter/ops/triton/moe/moe_routing_sigmoid_top1_fused.py`. Three changes:

- The output stores are masked to the valid rows. Upstream stores unmasked, so the last program writes past
  the output when `n_tokens` is not a multiple of `BLOCK_M`.
- Each program rebases `X_ptr` on its first row with a 64-bit offset (`pid_m.to(tl.int64) * BLOCK_M *
  stride_xm`), and the in-block row offsets stay 32-bit. Upstream forms `offs_m * stride_xm` in 32 bits:
  Triton types the stride argument as `i32` and `offs_m` is `i32`. That product wraps once
  `n_tokens * 5120` exceeds 2^31 elements, i.e. from row 419431. In the MI355X audit of the kernel without
  this change, the 524288-token case read wrong activations for exactly those rows: 19.8% of rows got a wrong
  id, which is 20% x 127/128, and every smaller case passed. The arithmetic and the tiling are unchanged.
- The wrapper is `run()` writing into preallocated outputs.

A candidate must likewise address activations beyond 2^31 elements. The output offsets (`n_tokens * 2`)
stay well inside 32 bits.

## Inputs and bounds

`inputs.py` draws activations as multiples of 1/8 in [-1, 1] and weights as multiples of 1/64 in [-1/8, 1/8],
all exact in bfloat16. Each logit is then an exact multiple of 1/512 independent of accumulation order
(logit std about 0.64), so the float32 kernel and the float64 reference rank experts identically; distinct
logits are at least 1/512 apart, far above the float32 sigmoid error, and exactly equal logits tie to the
lowest id in both. `topk_ids` therefore compares `exact`; an inexact grid would let near-tied scores flip
an id legitimately. This replaces the original suite's "any near-argmax expert is acceptable" rule.

`topk_weights` uses the fixed `tolerance` bound (`atol=1e-6`, `rtol=1e-5`) rather than the default
baseline-relative bound: the sigmoid scores carry only float32 evaluation error (a few ulps, depending on the
`exp` implementation), whereas the default bound scales with the baseline's own error, which is zero for
many rows here. The bound still rejects any lower-precision (bfloat16 or float16) score by three orders of
magnitude. The shared-expert weight must be exactly 1.0.

## Workloads

Performance cases span 4096 to 524288 tokens (128x). Large counts are beyond a single serving batch but are
the only way to keep the span at least 100x while every case stays above the timing floor; the activations
of the largest case occupy 5.4 GB. The reference works in row chunks to bound its float64 memory.

## Scope

The original suite also covered float16 inputs, other expert counts (16 to 64), other hidden sizes (2048 to
4096) and the variant without the shared-expert column (`topk_ids` of shape `[n_tokens, 1]`). A scenario fixes
dtype, constants and output layout, so those are separate scenarios (the same baseline kernel serves them);
this task covers the Llama-4 production configuration only.
